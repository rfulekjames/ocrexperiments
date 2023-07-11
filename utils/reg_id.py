import re
import os
from .misc import get_last_dot_index, tmp_folder
import subprocess
import pytesseract
from PIL import Image as pillow_image
from .alignment import (
    get_three_alignment,
    get_two_alignment,
    recover_from_aligned_candidates,
)
from rapidfuzz.process import extractOne
from rapidfuzz.fuzz import ratio, WRatio
from rapidfuzz.utils import default_process
import json
import boto3

s3 = boto3.resource("s3")
client = s3.meta.client  # used for interacting with client for convenience

tables_bucket = 'centene-test'
# ----------------Load regID table values-----------------------
response = client.get_object(Bucket=tables_bucket, Key="twenty_codes.json")
content = response["Body"].read().decode("utf-8")
twenty_digit_codes = json.loads(content)
twenty_digit_codes.sort(key=lambda x: x[-4:])  # order by year descending

response = client.get_object(Bucket=tables_bucket, Key="approval_codes.json")
content = response["Body"].read().decode("utf-8")
reg_id_approved = json.loads(content)
reg_id_approved
# -----------------Ouput, Queries and Rules ---------------


approval_re = re.compile(
    r"[(A-Z0-9_]+[_ ].*?(?:MATERIALS?|MODIFIED|MODIFIED_2023|ACCEPTED|SPN|)\d{7,8}|[A-Z0-9]+_[0-9A-Z]+_[A-Z0-9]+_[A-Z]+"
)
# approval_re = re.compile(r'[A-Z][A-Z0-9]+[_ ].*\d\d\d\d\d\d\d\d')
# twenty_re_old = re.compile(r'[A-Z][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9]+[ _.A-Z]\d{3,4}')
twenty_re = re.compile(
    r"[A-Z][A-Z][OSC0-9][A-Z][A-Z][A-Z][A-Z0-9][A-Z0-9][A-Z0-9]+[ _.][A-Z\d]{3,4}$"
)


# --------------REG ID EXTRACTION ------------------

TESSERACT_CONF_MULTIPLICATION_FACTOR = 1.1
TESSERACT_CODE_HI_CONFIDENCE_THRESHOLD = 100
TESSERACT_CODE_CONFIDENCE_THRESHOLD = 70

TEXTRACT_CODE_HI_CONFIDENCE_THRESHOLD = 93
TEXTRACT_CODE_CONFIDENCE_THRESHOLD = 70

DEFAULT_EXTRACTION_CUTOFF = 87
LOW_EXTRACTION_CUTOFF = 69

EXTRACTION_PREFIX = "******"


def multistage_extraction(get_text_function_list, reg_expressions):
    """
    multistage_extraction extracts text in multiple stages defined by arguments
    :param get_text_function_list:  list of functions to extract text
    :param reg_expressions: list of regular expressions to get a match, if None the raw extracted text is returned with highest confidence
    :return: List: None if no match found for the corresponding regular expression and a pair
    of the matched text with confidence otherwise.
    """
    matches_with_confidence = (
        len(reg_expressions) * [None] if reg_expressions else [None]
    )
    for get_text_function in get_text_function_list[::-1]:
        # we try all the extraction stages in the given order
        text, confidence = get_text_function()
        if len(text) == 0:
            continue
        print(f"Tesseract raw: {text}")
        # if confidence is 0 it often means the text is extracted Ok
        confidence = (
            confidence
            if confidence > 0.00001
            else TESSERACT_CODE_CONFIDENCE_THRESHOLD + 0.00001
        )
        # adjust confidence since it is lower than for textract
        confidence = min(100, confidence * TESSERACT_CONF_MULTIPLICATION_FACTOR)
        # print(f"Tesseract raw: {text}")
        text = text.replace(" _", "_")
        text = text.replace("_ ", "_")
        if reg_expressions:
            matches_found = [reg_exp.findall(text) for reg_exp in reg_expressions]
        else:
            matches_found = [text]
        # we overwrite previous matches (for each regular expression match or just the single text depending
        # on whether regular expression was given at the input)
        # if either the previous match is None or the confidence is higher than it was previously
        matches_with_confidence = [
            ((match[0] if reg_expressions else match), confidence)
            if (
                not matches_with_confidence[i]
                or confidence > matches_with_confidence[i][1]
            )
            and match
            else matches_with_confidence[i]
            for i, match in enumerate(matches_found)
        ]
        if all(matches_with_confidence):
            break
    if reg_expressions:
        print("Tesseract matches:")
    else:
        print("Tesseract extraction (no regex):")
    print(
        "\n".join(
            [
                f"Tesseract match {match[0]} with confidence {match[1]}"
                if match
                else "  with confidence {match[1]}"
                if match
                else ""
                for match in matches_with_confidence
            ]
        )
    )
    return matches_with_confidence


def get_tesseract_text_with_confidence(file, cfg):
    """
    gets text extracted by tesseract with average confidence over words
    :param file: path to the image file
    :cfg: config string for tesseract
    :return: extracted_text, confidence
    """
    data = pytesseract.image_to_data(
        pillow_image.open(file),
        config=cfg,
        output_type=pytesseract.Output.DICT,
    )
    return get_tesseract_text_from_data_with_confidence(data)


def get_tesseract_text_from_data_with_confidence(data):
    """
    gets concatenated words from data object returned by tesseract separated by spaces
    with confidence
    :param data: data object returned by tesseract
    :return: extracted_text, confidence
    """
    words, confidence, total_chars = [], 0, 0
    for i, word_text in enumerate(data["text"]):
        if data["conf"][i] != -1:
            confidence += int(data["conf"][i]) * len(word_text)
            total_chars += len(word_text)
            words.append(word_text)
    average_confidence = confidence / total_chars if total_chars else 0
    return " ".join(words), average_confidence


def get_codes_from_tesseract(ln, file, reg_expressions):
    """
    get_codes_from_tesseract extracts text based on reg_exp match
    it considers the bounding box specificed by ln in the image in the file.
    Does a little preprocessing that thickens the letters.
    :param ln:  line returned from Textract
    :param file: filename of the bottom part image
    :param reg_expressions: compiled regular expressions to match codes
    :return: List: None if no match found for the corresponding regular expression and the matched code otherwise.
    """

    def get_error_output():
        return [None] * len(reg_expressions) if reg_expressions else [None]

    file_in_folder = os.path.join(tmp_folder, file)
    last_dot_index = get_last_dot_index(file_in_folder)
    file_in_folder_morphed = (
        f"{file_in_folder[:last_dot_index]}-morphed{file_in_folder[last_dot_index:]}"
    )
    file_in_folder_cropped = (
        f"{file_in_folder[:last_dot_index]}-cropped{file_in_folder[last_dot_index:]}"
    )
    file_in_folder_cropped_morphed = f"{file_in_folder[:last_dot_index]}-cropped-morphed{file_in_folder[last_dot_index:]}"
    file_in_folder_cropped_2 = (
        f"{file_in_folder[:last_dot_index]}-cropped2{file_in_folder[last_dot_index:]}"
    )
    file_in_folder_cropped_morphed_2 = f"{file_in_folder[:last_dot_index]}-cropped-morphed2{file_in_folder[last_dot_index:]}"
    try:
        image = pillow_image.open(file_in_folder)
    except Exception as e:
        print(f"Code extraction failed: {e}")
        return get_error_output()
    width, height = image.size
    box = ln[2]["BoundingBox"]
    left = width * box["Left"]
    top = height * box["Top"]
    box_width = width * box["Width"]
    box_height = height * box["Height"]
    bounding_box = (left, top, left + box_width, top + box_height)
    image2 = image.crop(bounding_box)
    try:
        image2.save(file_in_folder_cropped_2)
    except (ValueError, OSError) as e:
        print(f"Code extraction failed: {e}")
        return get_error_output()
    # Need to make the bounding box larger a bit since Textract returns it too small
    bounding_box = (left - 20, top - 20, left + box_width + 20, top + box_height + 20)
    image = image.crop(bounding_box)
    try:
        image.save(file_in_folder_cropped)
    except (ValueError, OSError) as e:
        print(f"Code extraction failed: {e}")
        return get_error_output()
    # print(f'Bounding Box from {file_in_folder}:')
    # print(bounding_box)
    cfg = '-c tessedit_char_whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ "'

    def stage1_extraction():
        try:
            subprocess.run(
                [
                    "convert",
                    file_in_folder_cropped,
                    "-morphology",
                    "Erode",
                    "Disk:1.0",
                    "-compress",
                    "group4",
                    file_in_folder_cropped_morphed,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Code extraction failed: {e}")
            return "", 0
        return get_tesseract_text_with_confidence(file_in_folder_cropped_morphed, cfg)

    def stage2_extraction():
        try:
            subprocess.run(
                [
                    "convert",
                    file_in_folder_cropped_2,
                    "-morphology",
                    "Erode",
                    "Disk:1.0",
                    "-compress",
                    "group4",
                    file_in_folder_cropped_morphed_2,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Code extraction failed: {e}")
            return "", 0
        return get_tesseract_text_with_confidence(file_in_folder_cropped_morphed_2, cfg)

    def stage3_extraction():
        return get_tesseract_text_with_confidence(file_in_folder_cropped, cfg)

    def stage4_extraction():
        return get_tesseract_text_with_confidence(file_in_folder_cropped_2, cfg)

    get_text_function_list = [
        stage1_extraction,
        stage3_extraction,
        stage2_extraction,
        stage4_extraction,
    ]

    return multistage_extraction(get_text_function_list, reg_expressions)


def average_tesseract_textract_confidence(tesseract_conf, textract_conf):
    return (tesseract_conf + textract_conf) / 2


def get_reg_id_part(
    ln_first,
    ln_conf,
    ln_alternative,
    code,
    code_conf,
    code_name,
    table,
    extractOne_cutoff=DEFAULT_EXTRACTION_CUTOFF,
):
    """
    Returns extracted app_code or twenty_code.
    Steps 1. and 3. here: https://www.notion.so/Reg_Id-extraction-logic-eea5f0bd597b4970ac257af38fcaea13
    """
    if (ln_first or ln_alternative) and code == "":
        # it seems plausible (by multiple observation) that 0 confidence of tesseract
        # is in fact usually quite correct
        if ln_alternative:
            print(f"Tesseract {code_name} is {ln_alternative[0]}")
        else:
            print(f"No Tesseract extraction of {code_name}")
        if ln_first:
            print(f"Textract {code_name} is {ln_first[0]}")
        else:
            print(f"No Textract extraction of {code_name}")

        match_info, first_match_info, alternative_match_info, conf = None, None, None, 0
        # The processor removes all non alphanumeric characters from the ends, trims whitespace from the ends,
        # converting all characters to lower case, then does the comparison
        if ln_first:
            first_match_info = extractOne(
                ln_first[0],
                table,
                scorer=ratio,
                score_cutoff=extractOne_cutoff,
                processor=default_process,
            )

        if ln_alternative:
            alternative_match_info = extractOne(
                ln_alternative[0],
                table,
                scorer=ratio,
                score_cutoff=extractOne_cutoff,
                processor=default_process,
            )

        if extractOne_cutoff < DEFAULT_EXTRACTION_CUTOFF:
            # fallback part (Step 3.)
            # if both table lookup matches agree return the table result
            if (
                first_match_info
                and alternative_match_info
                and first_match_info[0] == alternative_match_info[0]
            ):
                code = first_match_info[0]
                code_conf = (first_match_info[1] + alternative_match_info[1]) / 2
                print(
                    f"{EXTRACTION_PREFIX} Table used for {code_name}: {code} (both matches agreed)"
                )
                print(f"Average table lookup score is {round(code_conf, 2)}")
            elif first_match_info or alternative_match_info:
                code = (
                    first_match_info[0]
                    if first_match_info
                    else alternative_match_info[0]
                )
                code_conf = (
                    first_match_info[1]
                    if first_match_info
                    else alternative_match_info[1]
                )
                print(
                    f"{EXTRACTION_PREFIX} Table used for {code_name}: {code} (one match agreed)"
                )
                print(f"Table match score is {code_conf}")
            return code, code_conf

        # pick table lookup match_info to be either first_match_info or alternative_match_info based on the similarity score
        # or take the one that exists if the other has no match.
        if first_match_info and alternative_match_info:
            match_info, extracted_code, conf = (
                (
                    first_match_info,
                    ln_first[0],
                    ln_conf,
                )
                if first_match_info[1] >= alternative_match_info[1]
                else (
                    alternative_match_info,
                    ln_alternative[0],
                    ln_alternative[1],
                )
            )
        elif first_match_info or alternative_match_info:
            match_info, extracted_code, conf = (
                (
                    first_match_info,
                    ln_first[0],
                    ln_conf,
                )
                if first_match_info
                else (
                    alternative_match_info,
                    ln_alternative[0],
                    ln_alternative[1],
                )
            )

        # Check if close match to result in table
        # search through table for best match, cutoff improves speed...if no
        # score for entry in table lookup lower than cutoff then further processing
        # on that entry stopped. If all items in table below cutoff then highest
        # score among them returned
        # If match above cutoff found then format close to table format
        if match_info:
            print(f"Table {code_name} is {match_info[0]} with score {match_info[1]}")
            ################### Tesseract Extraction and Alignment with Recovery ##########################################
            # print(f"extracted_code (Textract possibly Tesseract): {extracted_code},\n ln_alternative (Tesseract): {ln_alternative},\n match_info[0]: {match_info[0]}")
            extracted_info_without_spaces = extracted_code.replace(" ", "")
            match_without_spaces = match_info[0].replace(" ", "")
            # If table match and code differ only in spaces return table code.
            if match_without_spaces == extracted_info_without_spaces:
                # Step 1.a)
                code = match_info[0]
                code_conf = conf
                print(f"{EXTRACTION_PREFIX} Table used for {code_name}: {code}")
                print(f"Line confidence is {code_conf}")
                print(f"Table match score is {match_info[1]}")
                return code, code_conf

            # is extracted string very close to the table lookup string?
            is_good_match = are_closely_aligned(
                match_info[0],
                extracted_code,
                max_allowed_length_difference_between_strings=4,
            )
            # if the table code is far from the extracted code and
            # textract and tesseract codes are close to each other
            # and textract has high confidence return textract
            # (meant to cover the case when a code is missing in the table)
            if (
                not is_good_match
                and ln_conf > TEXTRACT_CODE_HI_CONFIDENCE_THRESHOLD
                and ln_first
                and ln_alternative
                and are_closely_aligned(
                    ln_first[0],
                    ln_alternative[0],
                )
            ):
                # Step 1.b)
                code = ln_first[0]
                code_conf = ln_conf
                print(f"{EXTRACTION_PREFIX} Textract {code_name} used {code}")
                print(f"Textract line confidence is {code_conf}")
            elif (
                ln_alternative
                and ln_first
                and ln_conf > TEXTRACT_CODE_CONFIDENCE_THRESHOLD
                and ln_alternative[1] > TESSERACT_CODE_CONFIDENCE_THRESHOLD
            ):
                # Step 1.c)
                (
                    code,
                    code_conf,
                ) = recover_from_aligned_candidates(
                    average_tesseract_textract_confidence(ln_alternative[1], ln_conf),
                    *get_three_alignment(
                        ln_first[0],
                        ln_alternative[0],
                        match_info[0],
                        gap_penalty=-10,
                        half_match_score=-5,
                    ),
                )
                print(
                    "Recovered output from three sources textract/tesseract/table lookup"
                )
                print(f"Tesseract code is {ln_alternative[0]}")
                print(f"Textract code is {ln_first[0]}")
                print(f"Table code is {match_info[0]}")
                print(f"Line confidence is {ln_conf}")
                print(
                    f"{EXTRACTION_PREFIX} Recovered {code_name}: {code} with confidence {code_conf}"
                )
            # Tesseract has super high confidence so use it
            elif (
                ln_alternative
                and ln_alternative[1] > TESSERACT_CODE_HI_CONFIDENCE_THRESHOLD
            ):
                # Step 1.d)
                code = ln_alternative[0]
                code_conf = ln_alternative[1]
                print(f"{EXTRACTION_PREFIX} Tesseract {code_name} used {code}")
                print(f"Tesseract line confidence is {code_conf}")
            else:
                # Step 1.e)
                # Either textract missed, or tesseract, but not both (as we have table match),
                # # or their confidence is not high, so use tableso use table
                code = match_info[0]
                # previous product with table score was too low in testing, going with line confidence as this is typically already low when
                # one of both codes not found
                code_conf = ln_conf
                print(f"{EXTRACTION_PREFIX} Table used for {code_name}: {code}")
                print(f"Line confidence is {code_conf}")
                print(f"Table match score is {match_info[1]}")
    return code, code_conf


def are_closely_aligned(
    string1,
    string2,
    max_allowed_length_difference_between_strings=3,
    max_allowed_deviation=2,
):
    """
    :param string1
    :param string2
    :return bool of whether the difference of the lengths of string1 and string2
    is at most max_allowed_length_difference_between_strings, and
    the length of the two_alignment (from get_two_alignment(.))
    is at most max_allowed_deviation away from the average length of string1 and string2
    """
    if (
        not abs(len(string1) - len(string2))
        <= max_allowed_length_difference_between_strings
    ):
        return False
    two_alignment = get_two_alignment(string1, string2)[0]
    return (
        abs((len(string1) + len(string2)) / 2 - len(two_alignment))
        <= max_allowed_deviation
    )


def get_reg_id(response_json, file):
    """To obtain the reg id using AWS Textract Detect Document Text.
    Outputs a tuple: string consisting of regulatory approval id and confidence score.
    """
    # Default values
    twenty_code, twenty_max_conf = "", 0
    approval_code, approval_max_conf = "", 0

    approval_code_back_up, twenty_code_back_up = None, None
    if "Blocks" in response_json:
        # Get all lines
        lines = [
            (item["Text"], item["Confidence"], item["Geometry"])
            for item in response_json["Blocks"]
            if item["BlockType"] == "LINE"
        ]
        # Examine last lines
        for ln in lines:
            # Get line text and confidence
            ln_text = ln[0].upper()
            ln_conf = ln[1]
            print(f"Textract raw {ln[0]} with confidence {ln[1]}")
            ln_text = ln_text.replace(" _", "_").replace("_ ", "_")
            # print(ln_text)
            # print(ln_text)
            # print(ln_conf)
            # Check if approval code in line
            ln_approval = approval_re.findall(ln_text)
            # Check if 20 code in line
            ln_twenty = twenty_re.findall(ln_text)
            ln_approval_alternative, ln_twenty_alternative = map(
                lambda code: (code[0].upper(), code[1]) if code else None,
                get_codes_from_tesseract(ln, file, (approval_re, twenty_re)),
            )
            # ln_twenty2 = twenty_re2.findall(ln_text)
            # If approval code found and we have not found one yet
            # How to choose cutoff? We have done limited experimentation. This is a guess that seemed to perform well in the handful of tests we did
            if ln_approval:
                ln_approval = (
                    fix_wrong_substring(
                        ln_approval[0], known_bad_approval_prefix_reg_exps
                    ),
                )
            if ln_approval_alternative:
                ln_approval_alternative = (
                    fix_wrong_substring(
                        ln_approval_alternative[0],
                        known_bad_approval_prefix_reg_exps,
                    ),
                    ln_approval_alternative[1],
                )
            approval_code, approval_max_conf = get_reg_id_part(
                ln_approval,
                ln_conf,
                ln_approval_alternative,
                approval_code,
                approval_max_conf,
                "app_code",
                reg_id_approved,
            )

            if ln_twenty:
                aux = fix_wrong_substring(
                    ln_twenty[0], known_bad_twenty_prefix_reg_exps
                )

                ln_twenty = (
                    fix_wrong_substring(
                        aux[::-1], known_bad_twenty_reg_reverse_prefix_exps
                    )[::-1],
                )
            if ln_twenty_alternative:
                aux = fix_wrong_substring(
                    ln_twenty_alternative[0], known_bad_twenty_prefix_reg_exps
                )
                ln_twenty_alternative = (
                    fix_wrong_substring(
                        aux[::-1],
                        known_bad_twenty_reg_reverse_prefix_exps,
                    )[::-1],
                    ln_twenty_alternative[1],
                )
            twenty_code, twenty_max_conf = get_reg_id_part(
                ln_twenty,
                ln_conf,
                ln_twenty_alternative,
                twenty_code,
                twenty_max_conf,
                "twenty_code",
                twenty_digit_codes,
            )
            approval_code_back_up = get_code_backup(
                ln_conf, ln_approval, ln_approval_alternative, approval_code_back_up
            )
            twenty_code_back_up = get_code_backup(
                ln_conf, ln_twenty, ln_twenty_alternative, twenty_code_back_up
            )
            if (twenty_max_conf > 0) and (approval_max_conf > 0):
                break

        if approval_code_back_up and approval_max_conf == 0:
            approval_code, approval_max_conf = (
                approval_code_back_up[0],
                approval_code_back_up[1],
            )
            print(
                f"{EXTRACTION_PREFIX} Using approval code backup {approval_code} with confidence {approval_max_conf}"
            )
        if twenty_code_back_up and twenty_max_conf == 0:
            twenty_code, twenty_max_conf = (
                twenty_code_back_up[0],
                twenty_code_back_up[1],
            )
            print(
                f"{EXTRACTION_PREFIX} Using twenty code backup {twenty_code} with confidence {twenty_max_conf}"
            )

        # fallback ignoring regular expression match and just using table lookup:
        # we lower the score cutoff for rapidfuzz (extractOne) lookup and
        # only return anything if both tess/text-ract give us the same table lookup value
        if approval_max_conf == 0 or twenty_max_conf == 0:
            for ln in lines:
                ln_text = ln[0].upper()
                ln_conf = ln[1]
                print(f"Textract raw line: {ln_text} with confidence {ln_conf}")
                ln_alternative = list(
                    map(
                        lambda code: (code[0].upper(), code[1]) if code else None,
                        get_codes_from_tesseract(ln, file, None),
                    )
                )

                approval_code, approval_max_conf = get_reg_id_part(
                    [ln_text],
                    ln_conf,
                    ln_alternative[0],
                    approval_code,
                    approval_max_conf,
                    "app_code_fallback",
                    reg_id_approved,
                    extractOne_cutoff=LOW_EXTRACTION_CUTOFF,
                )
                twenty_code, twenty_max_conf = get_reg_id_part(
                    [ln_text],
                    ln_conf,
                    ln_alternative[0],
                    twenty_code,
                    twenty_max_conf,
                    "twenty_code_fallback",
                    twenty_digit_codes,
                    extractOne_cutoff=LOW_EXTRACTION_CUTOFF,
                )

                if (twenty_max_conf > 0) and (approval_max_conf > 0):
                    break

    # # applying some rules by following the pattern given by the table codes we know at this point
    # # to fix the reg_id
    # approval_code = fix_wrong_substring(
    #     approval_code, known_bad_approval_prefix_reg_exps
    # )
    # twenty_code = fix_wrong_substring(twenty_code, known_bad_twenty_prefix_reg_exps)
    # Take min of the two conf. levels as the confidence overall that way
    # code only above cutoff if both parts are above cutoff
    if (twenty_max_conf > 0) and (approval_max_conf > 0):
        confidence_reg_id = min(twenty_max_conf, approval_max_conf)
    else:  # take the average so that if both are zero then it is 0, we use 0 to determine when no codes found (e.g. CILT)
        confidence_reg_id = (twenty_max_conf + approval_max_conf) / 2

    reg_id_match = " ".join((approval_code, twenty_code))

    # applying some rules by following the pattern given by the table codes we know at this point
    # to fix the reg_id
    reg_id_match = delimit_known_words_by_spaces(reg_id_match)
    reg_id_match = double__elimination(reg_id_match)

    return reg_id_match, confidence_reg_id


def double__elimination(string):
    double__index = string.find("__")
    while double__index > -1:
        string = string.replace("__", "_")
        double__index = string.find("__")
    return string


def fill_in_missing_underscores(string1, string2):
    string = ""
    string1 = string1.strip()
    string2 = string2.strip()
    if len(string1) != len(string2):
        return None
    for i, ch in enumerate(string1):
        chars = {ch, string2[i]}
        if ch == string2[i] or (" " in chars and "_" in chars):
            string += ch if ch == "_" else string2[i]
        else:
            return None
    return string


def get_code_backup(ln_conf, ln, ln_alternative, code_back_up):
    """
    Returns a code that would be used as a back up if Step 1. fails
    in Step 2. here: https://www.notion.so/Reg_Id-extraction-logic-eea5f0bd597b4970ac257af38fcaea13
    """
    if ln_alternative and ln:
        filled_in_string = fill_in_missing_underscores(ln[0], ln_alternative[0])
        if filled_in_string:
            ln = (filled_in_string,)
            ln_alternative = (filled_in_string, ln_alternative[1])
        code_candidate = None
        if ln_alternative[0] != ln[0]:
            if are_closely_aligned(ln[0], ln_alternative[0]):
                code_candidate, conf = (
                    (ln[0], ln_conf)
                    if ln_conf > ln_alternative[1]
                    else (
                        ln_alternative[0],
                        ln_alternative[1],
                    )
                )
        else:
            code_candidate, conf = (
                ln[0],
                average_tesseract_textract_confidence(ln_conf, ln_alternative[1]),
            )

        if code_candidate:
            backup_candidate = (
                code_candidate,
                conf,
            )
            code_back_up = (
                code_back_up
                if code_back_up and code_back_up[1] > backup_candidate[1]
                else backup_candidate
            )

    return code_back_up


known_words = {"INTERNAL", "APPROVED", "ACCEPTED"}


def delimit_known_words_by_spaces(string):
    for known_word in known_words:
        start_index_of_known_word = string.find(known_word)
        if start_index_of_known_word > 0:
            if string[start_index_of_known_word - 1] != " ":
                string = f"{string[:start_index_of_known_word]} {string[start_index_of_known_word:]}"
                start_index_of_known_word += 1
            index_after_string = start_index_of_known_word + len(known_word)
            if index_after_string < len(string) and string[index_after_string] != " ":
                string = f"{string[:index_after_string]} {string[index_after_string:]}"
    return string


# Sometimes (in some batches 1-5%) the prefix NA0 of twenty code is
# extracted as NAO. To have a unified logic with known_bad_approval_prefix_reg_exps
# we use 2 capture groups
known_bad_twenty_prefix_reg_exps = {
    r"(^NAO)(.+)": "NA0",
}

known_bad_twenty_reg_reverse_prefix_exps = {
    r"(^[0O]{2,5}[_ ]*)(.+)": "0000_",
}


# Sometimes (in some batches 1-5%) the prefix Y0020 of  approval code is not
# extracted correctly, e.g., it is extracted as 20, 0020, YO020, YU020, Y00Z0, etc.
known_bad_approval_prefix_reg_exps = {
    r"(^[YO02UZ][O02UZ]{1,4}[_ ]+)([^_ ])": "Y0020_",
    r"(^[YO02UZ][O02UZ]{1,4})([W])": "Y0020_",
}


# reg_exps_with_replacement needs to capture groups
def fix_wrong_substring(string, reg_exps_with_replacement):
    string = string.strip()
    for reg_exp in reg_exps_with_replacement:
        if (
            string[: len(reg_exps_with_replacement[reg_exp])]
            != reg_exps_with_replacement[reg_exp]
        ):
            string = re.sub(reg_exp, rf"{reg_exps_with_replacement[reg_exp]}\2", string)
        # print(string)
    return string
