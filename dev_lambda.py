import boto3
from decimal import Decimal
import json
import urllib.request
import urllib.parse
import urllib.error
import math
import re
from rapidfuzz.process import extractOne
from rapidfuzz.fuzz import ratio, WRatio
from rapidfuzz.utils import default_process
import subprocess
import os
from PIL import Image as pillow_image
import pytesseract
from itertools import product
from collections import deque
import pandas as pd


print("Loading function")

test_bucket_name = "centene-test"
output_bucket_name = "centene-test-out"

session = boto3.Session(profile_name="default")
textract = boto3.client("textract")
s3 = boto3.resource("s3")
client = s3.meta.client  # used for interacting with client for convenience

region = boto3.session.Session().region_name

# a2i=boto3.client('sagemaker-a2i-runtime', region_name=region)

# Must be different from trigger bucket
# Lambda IAM role only has write permission to this bucket
tables_bucket = test_bucket_name
output_bucket = output_bucket_name
data_bucket = test_bucket_name


approval_re = re.compile(
    r"[(A-Z0-9_]+[_ ].*?(?:MATERIALS?|MODIFIED|MODIFIED_2023|ACCEPTED|SPN|)\d{7,8}|[A-Z0-9]+_[0-9A-Z]+_[A-Z0-9]+_[A-Z]+"
)
# approval_re = re.compile(r'[A-Z][A-Z0-9]+[_ ].*\d\d\d\d\d\d\d\d')
# twenty_re_old = re.compile(r'[A-Z][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9]+[ _.A-Z]\d{3,4}')
twenty_re = re.compile(
    r"[A-Z][A-Z][OSC0-9][A-Z][A-Z][A-Z][A-Z0-9][A-Z0-9][A-Z0-9]+[ _.][A-Z\d]{3,4}$"
)


# os.environ["BUCKET"] = data_bucket
# os.environ["REGION"] = region
# role = sm.get_execution_role()

# ----------------Load regID table values-----------------------
response = client.get_object(Bucket=tables_bucket, Key="twenty_codes.json")
content = response["Body"].read().decode("utf-8")
twenty_digit_codes = json.loads(content)
twenty_digit_codes.sort(key=lambda x: x[-4:])  # order by year descending

response = client.get_object(Bucket=tables_bucket, Key="approval_codes_corrected.json")
content = response["Body"].read().decode("utf-8")
reg_id_approved = json.loads(content)
reg_id_approved
# -----------------Ouput, Queries and Rules ---------------
# JSON structure to hold the extraction result


queries = [
    {"Text": "Who is the person?", "Alias": "ADDRESSEE"},
    {"Text": "What is the street address of the person?", "Alias": "STREET_ADDRESS"},
    {"Text": "What is the city of the person?", "Alias": "CITY"},
    {"Text": "What is the state of the person?", "Alias": "STATE"},
    {"Text": "What is the zip code of the person?", "Alias": "ZIP_CODE_4"},
]
# confidence_threshold = 101 #For manual verification of all docs
confidence_threshold = 97  # cutoff for automatic verification
rules = [
    {
        "description": f"ADDRESSEE confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ADDRESSEE",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"ADDRESS_LINE_1 confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ADDRESS_LINE_1",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"ADDRESS_LINE_2 confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ADDRESS_LINE_2",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"CITY confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "CITY",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"STATE confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "STATE",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"ZIP_CODE_4 confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "ZIP_CODE_4",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
    {
        "description": f"REGULATORY_APPROVAL_ID confidence score should be greater than or equal to {confidence_threshold}",
        "field_name": "REGULATORY_APPROVAL_ID",
        "field_name_regex": None,  # support Regex: '_confidence$',
        "condition_category": "Confidence",
        "condition_type": "ConfidenceThreshold",
        "condition_setting": confidence_threshold,
    },
]


# ----------------Image processing -----------------------------------------
def split_tif(filepath):
    print(filepath)
    command_version = ["convert", "-version"]
    subprocess.run(command_version)

    last_dot_index = get_last_dot_index(filepath)
    # command = ["convert", filepath, filepath[:last_dot_index] + "-%02d.tif"]
    command2 = [
        "convert",
        filepath,
        "-crop",
        "100%x100%",
        "+repage",
        "-write",
        filepath[:last_dot_index] + "-%02d.tif",
        "null:",
    ]

    try:
        subprocess.run(command2, check=True)
        print("Image splitting completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")

    # list the TIFF individual pages
    page_files = []

    # Get a list of the generated page files
    for file in os.listdir("tmp"):
        if file.endswith(".tif"):
            page_files.append("tmp/" + file)
    print(page_files)
    # Combine the individual pages into a single PNG file
    cilt_name = "tmp/cilt.png"
    combine_command = ["convert"] + page_files + ["-append", cilt_name]
    try:
        # Execute the command to combine the pages into a PNG file
        subprocess.run(combine_command, check=True)
        print("Image CILT combining completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image CILT combining failed: {e}")


def get_last_dot_index(filepath):
    last_dot_index = filepath[::-1].find(".")
    return -last_dot_index - 1 if last_dot_index != -1 else 0


def extract_bottom(input_path, output_path):
    command = [
        "convert",
        input_path,
        "-gravity",
        "south",
        "-crop",
        "100%x15%",
        output_path,
    ]

    try:
        subprocess.run(command, check=True)
        print("Bottom extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")


def extract_top(input_path, output_path):
    command = [
        "convert",
        input_path,
        "-gravity",
        "north",
        "-crop",
        "100%x85%",
        output_path,
    ]
    try:
        subprocess.run(command, check=True)
        print("Top extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")


def fix_orientation(file):
    try:
        image = pillow_image.open(file)
    except Exception as e:
        print(f"Unable to open file: {e}")   
        return
    newdata=pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    if newdata['rotate']:
        command = [
            "convert",
            file,
            "-rotate",
            str(newdata['rotate']),
            file,
        ]
        try:
            subprocess.run(command, check=True)
            print("Page rotated successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Page rotation failed: {e}")


def split_into_pages_top_and_bottom(filepath):
    # given the filepath of the original multipage .TIF file
    # we split it into pages and for each page, further split
    # into top and bottom.
    split_tif(filepath)
    # iterate over pages
    files = os.listdir("tmp")
    files = [file for file in files if '-' and '.tif' in file]
    files.sort()
    for file in files[:4]:
        fix_orientation(f'tmp/{file}')
        last_dot_index = get_last_dot_index(filepath)
        print("Splitting the following page into the two files below:")
        bottom_filename = (
            f"tmp/{file[:last_dot_index]}-bottom{file[last_dot_index:]}"
        )
        print(bottom_filename)
        top_filename = f"tmp/{file[:last_dot_index]}-top{file[last_dot_index:]}"
        print(top_filename)
        extract_bottom("tmp/" + file, bottom_filename)
        extract_top("tmp/" + file, top_filename)



def add_border(filepath):
    input_file = filepath
    output_file = filepath[: get_last_dot_index(filepath)] + "-border.png"

    command = [
        "convert",
        input_file,
        "-bordercolor",
        "lime",
        "-border",
        "5x5",
        output_file,
    ]
    try:
        subprocess.run(command, check=True)
        print("Border addition completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")


def recombine(top_path, bottom_path):
    # add border to both top and bottom image
    add_border(top_path)
    add_border(bottom_path)

    # append bottom to top and compress as png
    command = [
        "convert",
        top_path[: get_last_dot_index(top_path)] + "-border.png",
        bottom_path[: get_last_dot_index(bottom_path)] + "-border.png",
        "-append",
        "-define",
        "png:compression-filter=5",
        "-define",
        "png:compression-level=9",
        "-define",
        "png:compression-strategy=1",
        "tmp/image-for-A2I.png",
    ]
    try:
        subprocess.run(command, check=True)
        print("Image append completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")

    os.remove(
        top_path[: get_last_dot_index(top_path)] + "-border.png"
    )  # Remove the intermediate files
    os.remove(bottom_path[: get_last_dot_index(bottom_path)] + "-border.png")


# --------------- Helper Functions to call textract APIs ------------------
# copy


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
    try:
        data = pytesseract.image_to_data(
            pillow_image.open(file),
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        print(f"Code extraction failed: {e}")   
        return '', 0
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
        return [None]*len(reg_expressions) if reg_expressions else [None]
    file_in_folder = f"tmp/{file}"
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
            return '', 0
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
            return '', 0
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


TESSERACT_CONF_MULTIPLICATION_FACTOR = 1.1
TESSERACT_CODE_HI_CONFIDENCE_THRESHOLD = 100
TESSERACT_CODE_CONFIDENCE_THRESHOLD = 70

TEXTRACT_CODE_HI_CONFIDENCE_THRESHOLD = 93
TEXTRACT_CODE_CONFIDENCE_THRESHOLD = 70

DEFAULT_EXTRACTION_CUTOFF = 87
LOW_EXTRACTION_CUTOFF = 69

EXTRACTION_PREFIX = "******"


def get_reg_id_part(
    ln_first,
    ln_conf,
    ln_alternative,
    code,
    code_conf,
    code_name,
    extraction_dict,
    table,
    extractOne_cutoff=DEFAULT_EXTRACTION_CUTOFF,
):
    if (ln_first or ln_alternative) and code == "":
        # it seems plausible (by multiple observation) that 0 confidence of tesseract
        # is in fact usually quite correct
        if ln_alternative:
            print(f"Tesseract {code_name} is {ln_alternative[0]}")
            extraction_dict["history"].append(
                f"Tesseract {code_name} is {ln_alternative[0]}"
            )
        else:
            print(f"No Tesseract extraction of {code_name}")
        if ln_first:
            print(f"Textract {code_name} is {ln_first[0]}")
            extraction_dict["history"].append(f"Textract {code_name} is {ln_first[0]}")
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

        # fallback part (last resort)
        if extractOne_cutoff < DEFAULT_EXTRACTION_CUTOFF:
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
                extraction_dict[code_name] = code
                extraction_dict[
                    f'{code_name.replace("_fallback", "")}_type'
                ] = "fallback-1"
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
                extraction_dict[code_name] = code
                extraction_dict[f"{code_name.replace('_fallback', '')}_type"] = "fallback-2"
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
            extraction_dict[f"{code_name}_table_lookup"] = match_info[0]
            extracted_info_without_spaces = extracted_code.replace(" ", "")
            match_without_spaces = match_info[0].replace(" ", "")
            # If table match and code differ only in spaces return table code.
            if match_without_spaces == extracted_info_without_spaces:
                code = match_info[0]
                code_conf = conf
                print(f"{EXTRACTION_PREFIX} Table used for {code_name}: {code}")
                print(f"Line confidence is {code_conf}")
                print(f"Table match score is {match_info[1]}")
                extraction_dict[f"{code_name}_type"] = "100-match-mod-spaces"
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
                code = ln_first[0]
                code_conf = ln_conf
                print(f"{EXTRACTION_PREFIX} Textract {code_name} used {code}")
                print(f"Textract line confidence is {code_conf}")
                extraction_dict[f"textract_conf_{code_name}"] = round(ln_conf, 2)
                extraction_dict[f"{code_name}_textract"] = extracted_code
                extraction_dict[f"{code_name}_type"] = "hi-textract"
            elif (
                ln_alternative
                and ln_first
                and ln_conf > TEXTRACT_CODE_CONFIDENCE_THRESHOLD
                and ln_alternative[1] > TESSERACT_CODE_CONFIDENCE_THRESHOLD
            ):
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
                extraction_dict[f"textract_conf_{code_name}"] = round(ln_conf, 2)
                extraction_dict[f"tesseract_conf_{code_name}"] = round(
                    ln_alternative[1], 2
                )
                extraction_dict[f"{code_name}_textract"] = extracted_code
                extraction_dict[f"{code_name}_tesseract"] = ln_alternative[0]
                extraction_dict[f"{code_name}_type"] = "recovery"
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
                code = ln_alternative[0]
                code_conf = ln_alternative[1]
                print(f"{EXTRACTION_PREFIX} Tesseract {code_name} used {code}")
                print(f"Tesseract line confidence is {code_conf}")
                extraction_dict[f"tesseract_conf_{code_name}"] = ln_alternative[1]
                extraction_dict[f"{code_name}_tesseract"] = ln_alternative[0]
                extraction_dict[f"{code_name}_type"] = "hi-tesseract"
            else:
                # Either textract missed, or tesseract, but not both (as we have table match),
                # # or their confidence is not high, so use tableso use table
                code = match_info[0]
                # previous product with table score was too low in testing, going with line confidence as this is typically already low when
                # one of both codes not found
                code_conf = ln_conf
                print(f"{EXTRACTION_PREFIX} Table used for {code_name}: {code}")
                print(f"Line confidence is {code_conf}")
                print(f"Table match score is {match_info[1]}")
                extraction_dict[f"{code_name}_type"] = "table"
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
    extraction_dict = {}
    extraction_dict["history"] = []

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
                extraction_dict,
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
                extraction_dict,
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
            extraction_dict["approval_back_up"] = approval_code
            extraction_dict[f"app_code_type"] = "back-up"

        if twenty_code_back_up and twenty_max_conf == 0:
            twenty_code, twenty_max_conf = (
                twenty_code_back_up[0],
                twenty_code_back_up[1],
            )
            print(
                f"{EXTRACTION_PREFIX} Using twenty code backup {twenty_code} with confidence {twenty_max_conf}"
            )
            extraction_dict["twenty_back_up"] = twenty_code
            extraction_dict[f"twenty_code_type"] = "back-up"

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
                    extraction_dict,
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
                    extraction_dict,
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

    return reg_id_match, confidence_reg_id, extraction_dict


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


# def fix_wrong_prefix(string, known_bad_prefixes):
#     changed = True
#     string = string.strip()
#     while changed:
#         changed = False
#         for known_bad_prefix in known_bad_prefixes:
#             if not string.find(known_bad_prefix):
#                 string = string.replace(
#                     known_bad_prefix, known_bad_prefixes[known_bad_prefix], 1
#                 )
#                 changed = True
#     return string


# print(fix_wrong_substring("00_W", known_bad_approval_prefix_reg_exps))


def detect_text(bucket, key):
    response = textract.detect_document_text(
        Document={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    return response


def euclidean_distance(point1, point2):
    # Calculate the squared differences of coordinates
    squared_diffs = [(x - y) ** 2 for x, y in zip(point1, point2)]

    # Sum the squared differences and take the square root
    distance = math.sqrt(sum(squared_diffs))

    return distance


def get_next_line(query_string, response_json):
    """To obtain the next line in the text after the query string, we search
    through lines to find lower left x value of the query box. If distance
    between lower left of `query_string` and upper left of following line is
    close, we return it.
    Outputs a string consisting of the next line.
    """
    query_x = 0
    upper_left = 0
    next_line = None

    # Check that query_string not null
    if query_string and ("Blocks" in response_json):
        for item in response_json["Blocks"]:
            if item["BlockType"] == "LINE":
                # Search through lines to find line corresponding
                # to query string, record lower_left, then continue until
                # line found with upper_left approximately equal to it.
                # query_x == 0 indicates it was not found yet
                if query_x != 0:
                    # check if upper left of line equals lower_left of query
                    line_x = item["Geometry"]["Polygon"][0]["X"]
                    line_y = item["Geometry"]["Polygon"][0]["Y"]
                    distance = euclidean_distance((query_x, query_y), (line_x, line_y))
                    if distance <= 0.008:
                        next_line = item["Text"]
                        break  # exit loop and return

                # Check if we found line corresponding to input string
                # Sometimes query result can miss characters such as umlaut
                # May assume query_result is non-empty but may have dropped
                # a character from the actual line_string
                line_string = item["Text"]

                # check if query chars are substring of line string
                diff = len(line_string) - len(query_string)
                if 0 <= diff <= 1 and all(x in line_string for x in query_string):
                    # get lower left
                    query_x = item["Geometry"]["Polygon"][3]["X"]
                    query_y = item["Geometry"]["Polygon"][3]["Y"]

    return next_line


def get_city(address_line2_string):
    city = None
    if address_line2_string:
        pattern = r"\b[A-Z]{2}\b"
        no_state = re.sub(pattern, "", address_line2_string)
        pattern = "[0-9]{5}(?:-[0-9]{4})?"
        no_zip_no_state = re.sub(pattern, "", no_state)
        city = no_zip_no_state.rstrip()
        city = city.rstrip(",")
    return city


def get_state(address_line2_string):
    state_string = None
    if address_line2_string:
        pattern = r"\b[A-Z]{2}\b"
        result = re.findall(pattern, address_line2_string)
        if result:
            state_string = result[-1]  # get last in case two letter city
    return state_string


def get_zip(address_line2_string):
    zip_string = None
    if address_line2_string:
        # Extract ZIP code from string
        # in case of bad characters, etc.
        pattern = "[0-9]{5}(?:-[0-9]{4})?"
        formatted_zip = re.findall(pattern, address_line2_string)  # returns only zip
        if formatted_zip:
            zip_string = formatted_zip[0]
    return zip_string


def get_query_results(ref_id, response_json):
    """Given id and response, search for
    query results for that id and return results"""

    for b in response_json["Blocks"]:
        if b["BlockType"] == "QUERY_RESULT" and b["Id"] == ref_id:
            return {
                "value": b.get("Text"),
                "confidence": b.get("Confidence"),
                "block": b,
            }
    return None


def parse_response_to_json(response_json):
    """Update query_output dictionary above with response information
    in format required for Condition to check it
    Input: response JSON from textract API
    """
    # if response not null
    if response_json:
        # for each query/alias parse results from response
        for b in response_json["Blocks"]:
            if b["BlockType"] == "QUERY" and "Relationships" in b:
                ref_id = b["Relationships"][0]["Ids"][0]  # record id
                results = get_query_results(ref_id, response_json)
                q_alias = b["Query"]["Alias"]  # record alias
                query_output[q_alias] = results

    return query_output


def save_dict_to_s3(dict_obj, bucket_name, file_name):
    """Saves dict_obj json key value pairs obtained from the queries,
    to the target S3 bucket bucket_name under file_name.
    input1: python dictionary object
    input2: bucket name in form of string
    input3: filename in form of string
    """
    try:
        s3.Object(bucket_name, file_name).put(Body=json.dumps(dict_obj))
        status = f"{file_name} saved to s3://{bucket_name}"
    except Exception as e:
        status = f"Failed to save: {e}"

    return status


# --------------- Condition class --------------
from enum import Enum
import re


class Condition:
    _data = None
    _conditions = None
    _result = None

    def __init__(self, data, conditions):
        self._data = data
        self._conditions = conditions

    def check(self, field_name, obj):
        r, s = [], []
        for c in self._conditions:
            # Matching field_name or field_name_regex
            condition_setting = c.get("condition_setting")
            if c["field_name"] == field_name or (
                c.get("field_name") is None
                and c.get("field_name_regex") is not None
                and re.search(c.get("field_name_regex"), field_name)
            ):
                field_value, block = None, None
                if obj is not None:
                    field_value = obj.get("value")
                    block = obj.get("block")
                    confidence = obj.get("confidence")

                if c["condition_type"] == "Required" and (
                    obj is None or field_value is None or len(str(field_value)) == 0
                ):
                    r.append(
                        {
                            "message": f"The required field [{field_name}] is missing.",
                            "field_name": field_name,
                            "field_value": field_value,
                            "condition_type": str(c["condition_type"]),
                            "condition_setting": condition_setting,
                            "condition_category": c["condition_category"],
                            "block": block,
                        }
                    )
                elif (
                    c["condition_type"] == "ConfidenceThreshold"
                    and obj is not None
                    and c["condition_setting"] is not None
                    and float(confidence) < float(c["condition_setting"])
                ):
                    r.append(
                        {
                            "message": f"The field [{field_name}] confidence score {confidence} is LOWER than the threshold {c['condition_setting']}",
                            "field_name": field_name,
                            "field_value": field_value,
                            "condition_type": str(c["condition_type"]),
                            "condition_setting": condition_setting,
                            "condition_category": c["condition_category"],
                            "block": block,
                        }
                    )

                elif (
                    field_value is not None
                    and c["condition_type"] == "ValueRegex"
                    and condition_setting is not None
                    and re.search(condition_setting, str(field_value)) is None
                ):
                    r.append(
                        {
                            "message": f"{c['description']}",
                            "field_name": field_name,
                            "field_value": field_value,
                            "condition_type": str(c["condition_type"]),
                            "condition_setting": condition_setting,
                            "condition_category": c["condition_category"],
                            "block": block,
                        }
                    )

                # field has condition defined and sastified
                s.append(
                    {
                        "message": f"The field [{field_name}] confidence score is {confidence}.",
                        "field_name": field_name,
                        "field_value": field_value,
                        "condition_type": str(c["condition_type"]),
                        "condition_setting": condition_setting,
                        "condition_category": c["condition_category"],
                        "block": block,
                    }
                )

        return r, s

    def check_all(self):
        if self._data is None or self._conditions is None:
            return None
        # if rule missed, this list holds list
        broken_conditions = []
        rules_checked = []
        broken_conditions_with_all_fields_displayed = []
        # iterate through rules_data dictionary
        # key is a field, value/obj is a
        # dictionary, for instance name_rule_data
        # is the dictionary for ADDRESSEE
        for key, obj in self._data.items():
            value = None
            if obj is not None:
                value = obj.get("value")

            if value is not None and type(value) == str:
                value = value.replace(" ", "")
            # Check if this field passed rules:
            # if so, r and s are both []
            # if not, r is list of one or more of the dictionaries
            # seen in the check function
            r, s = self.check(key, obj)
            # If field missed rule
            if r and len(r) > 0:
                # append to bc list
                broken_conditions += r
            # If rule checked for field
            if s and len(s) > 0:
                rules_checked += s
            # If field missed or not
            # append to b_c_w_a_f_d
            if s and len(s) > 0:
                if r and len(r) > 0:
                    # If rule missed on this field, record it
                    broken_conditions_with_all_fields_displayed += r
                else:
                    # If no rule missed, record field
                    broken_conditions_with_all_fields_displayed += s

        # apply index
        idx = 0
        # iterate through dictionaries
        for r in broken_conditions_with_all_fields_displayed:
            idx += 1
            r["index"] = idx
        # If at least one rule missed, display with it all fields
        # otherwise broken c
        if broken_conditions:
            broken_conditions = broken_conditions_with_all_fields_displayed
        return broken_conditions, rules_checked


# --------------- Dictionaries for data storage ------------------
query_output = {
    "ADDRESSEE": None,
    "STREET_ADDRESS": None,
    "CITY": None,
    "STATE": None,
    "ZIP_CODE_4": None,
}
# Centene fields format for saving to S3
centene_format = {
    "RICOH_DCN": None,
    "REGULATORY_APPROVAL_ID": None,
    "ADDRESSEE": None,
    "ADDRESS_LINE_1": None,
    "ADDRESS_LINE_2": None,
    "CITY": None,
    "STATE": None,
    "ZIP_CODE_4": None,
    "BATCH": None,
    "BATCH_PREFIX": None,
}


# --------------- Main handler ------------------
def lambda_handler(event, context):
    for filename in os.listdir("tmp"):
        # if filename.endswith('.tif'):  # Check if the file ends with ".tif"
        file_path = os.path.join("tmp", filename)  # Get the full file path
        os.remove(file_path)  # Remove the file
        print(f"Removed file: {filename}")

    global extraction_log_df
    """Demonstrates S3 trigger that uses
    textract APIs to detect text, query text in S3 Object.
    """
    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    print(key)
    # ------------Image processing on TIF ----------------
    ## save TIF
    local_file_name = "tmp/{}".format(os.path.basename(key))
    # s3.download_file(Bucket=bucket, Key=key, Filename=local_file_name)
    # inFile = open(local_file_name, "r")
    print(f"local_file_name is: {local_file_name}")

    # Download file from S3 to local tmp directory
    try:
        s3.Object(bucket, key).download_file(local_file_name)
        print(f"File downloaded successfully from S3.")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")

    split_into_pages_top_and_bottom(local_file_name)

    # directory = 'tmp'  # Replace with your directory path
    # print('currently in tmp')
    # for filename in os.listdir(directory):
    #    print(filename)

    # Insert 'file' key pair
    # Split the string by '/'
    parts = key.split("/")

    # Get the text after the final '/'
    ricoh_dcn = parts[-1]

    ##########################################################
    # centene_format['RICOH_DCN'] = parts[-1][:-4] #strip .T
    # centene_format['BATCH'] = parts[-2]
    # centene_format['BATCH_PREFIX'] = parts[-3]
    ##########################################################
    # Save to S3 in correct place in output bucket

    # Get Regid

    bottom_list = []
    top_list = []
    reg_id_conf = 0
    files = os.listdir("tmp")
    for file in files:
        # print('Here is one file:')
        # print(file)
        if ".tif" in file:
            print("Save to output bucket this:")
            s3_filename = f"{key.rstrip(parts[-1])}{file}"

            # s3_filename = f'{parts[-4]}/{parts[-3]}/{parts[-2]}/{file}'
            print(f"s3_filename is {s3_filename}")
            # Read the local file as bytes
            # with open(local_file_name, 'rb') as f:
            #    file_bytes = f.read()
            ##s3.upload_fileobj(
            ##    fileobj=file_bytes,
            #    bucket=output_bucket,
            #    key=s3_filename
            #    )
            # s3.Object(output_bucket, s3_filename).put(Body=file_bytes)
            # s3.put_object(
            #    Body=file_bytes,
            #    Bucket=output_bucket,
            #    Key=s3_filename
            #    )
            client.upload_file("tmp/" + file, output_bucket, s3_filename)
            if "bottom" in file:
                bottom_list.append(s3_filename)
            if "top" in file:
                top_list.append(s3_filename)
    # iterate over pages bottom
    bottom_response = []
    # set regid file to be cilt if None, if found update to be that
    reg_id_file = None
    bottom_list = sorted(bottom_list, key=lambda x: x[-13:])
    print(f"Bottom list is: {bottom_list}")

    with open(f"outputs/output-{output_suffix}.txt", "a") as f:
        f.write(f"\n*******{key}*****************")

    for file in bottom_list:
        # Call the analyze_document API

        print(file)

        try:
            # Call the analyze_document API
            print(f"calling Textract on {file}")
            bottom_response = textract.detect_document_text(
                Document={"S3Object": {"Bucket": output_bucket, "Name": file}}
            )

        except Exception as e:
            print(e)

        if bottom_response:
            print(f"Textract called on {file} sucessfully.")
            # Use document knowledge that RegID at bottom and certain format to grab it
            file_name = file.split("/")[-1]
            reg_id, reg_id_conf, extration_dict = get_reg_id(bottom_response, file_name)
            if reg_id_conf > 0:
                print(f"RegID code found in {file}.")
                print(reg_id, reg_id_conf)
                # save location
                reg_id_file = file

                centene_format["REGULATORY_APPROVAL_ID"] = reg_id

                with open(f"outputs/output-{output_suffix}.txt", "a") as f:
                    f.write(f"******{file_name}****\n")
                    f.write(f"{reg_id} .....with confidence {reg_id_conf}")
                    extration_dict["doc"] = file_name
                    extration_dict["RegId"] = reg_id
                    extration_dict["RegIdConf"] = reg_id_conf
                    extraction_log_df = pd.concat(
                        [extraction_log_df, pd.DataFrame([extration_dict])],
                        ignore_index=True,
                    )

                break

    # TOP PROCESSING TEMPORARILY COMMENTED OUT
    ################################################################################################
    # In case no name or no regid found, make sure regid blank
    # if not reg_id_file:
    #     centene_format["REGULATORY_APPROVAL_ID"] = ""

    # query_response = []
    # # Define to be cilt if none
    # name_file = None
    # name_conf = 0
    # top_list = sorted(top_list, key=lambda x: x[-13:])
    # print(f"Top list is: {top_list}")

    # for file in top_list:
    #     print(f"Top file is : {file}")
    #     try:
    #         # Calls textract detect_document_text API to detect text in the document S3 object
    #         # text_response = detect_text(bucket, key)
    #         print(f"Calling Textract on {file}")
    #         # Calls textract analyze_document API to query S3 object
    #         query_response = textract.analyze_document(
    #             Document={"S3Object": {"Bucket": output_bucket, "Name": file}},
    #             FeatureTypes=["QUERIES"],
    #             QueriesConfig={"Queries": queries},
    #         )
    #     except Exception as e:
    #         print(e)
    #         print(
    #             "Error processing object {} from bucket {}. ".format(key, bucket)
    #             + "Make sure your object and bucket exist and your bucket is in the same region as this function."
    #         )
    #         # Save error to file
    #         # status = save_dict_to_s3(centene_format, output_bucket,output_file_name)
    #         # print(status)
    #         raise e
    #     if query_response:
    #         # Store json of parsed response of first page
    #         query_data = parse_response_to_json(query_response)

    #         # Check if query name available
    #         if query_data["ADDRESSEE"]:
    #             print(f"Name found on {file}")
    #             name_file = file
    #             name = query_data["ADDRESSEE"]["value"]
    #             name_conf = query_data["ADDRESSEE"]["confidence"]

    #             # Function get_next_line searches for name in lines, if not found, returns null
    #             # If ad1 returns null, then query name wrong or does not exist in document lines
    #             ad1 = get_next_line(name, query_response)

    #             if (
    #                 ad1
    #             ):  # If ad1 not null, query name is correct, so use it to get fields
    #                 ad2 = get_next_line(ad1, query_response)
    #                 ad3 = get_next_line(ad2, query_response)

    #                 if not ad3:  # If no 3rd address line
    #                     centene_format["ADDRESSEE"] = name
    #                     centene_format["ADDRESS_LINE_1"] = ad1
    #                     centene_format["ADDRESS_LINE_2"] = ""
    #                     centene_format["CITY"] = get_city(ad2)
    #                     centene_format["STATE"] = get_state(ad2)
    #                     centene_format["ZIP_CODE_4"] = get_zip(ad2)
    #                 else:  # If there is a 3rd address line
    #                     centene_format["ADDRESSEE"] = name
    #                     centene_format["ADDRESS_LINE_1"] = ad1
    #                     centene_format["ADDRESS_LINE_2"] = ad2
    #                     centene_format["CITY"] = get_city(ad3)
    #                     centene_format["STATE"] = get_state(ad3)
    #                     centene_format["ZIP_CODE_4"] = get_zip(ad3)
    #             else:  # Query name is wrong or doesn't exist, pass to query defaults
    #                 street_address = ""
    #                 city = ""
    #                 state = ""
    #                 zip_code = ""

    #                 # If queries found fields use them
    #                 if query_data["STREET_ADDRESS"]:
    #                     street_address = query_data["STREET_ADDRESS"]["value"]
    #                 if query_data["CITY"]:
    #                     city = query_data["CITY"]["value"]
    #                 if query_data["STATE"]:
    #                     state = query_data["STATE"]["value"]
    #                 if query_data["ZIP_CODE_4"]:
    #                     zip_code = query_data["ZIP_CODE_4"]["value"]

    #                 # Now input the values we have
    #                 centene_format["ADDRESSEE"] = name
    #                 centene_format["ADDRESS_LINE_1"] = street_address
    #                 centene_format["ADDRESS_LINE_2"] = ""
    #                 centene_format["CITY"] = city
    #                 centene_format["STATE"] = state
    #                 centene_format["ZIP_CODE_4"] = zip_code

    #             break  # break out of loop

    # # Finally, update the query_data with new confidence values
    # # and values based on what we found
    # # query_data
    # # TEMPORARY: make fields derived from name have same confidence
    # # We may want to pass in the confidence on the line (e.g. on City line, state line
    # # etc. instead
    # # OPTIONAL: Make confidence minimum of all fields, that way all fields
    # # show in the A2I when one field is below cutoff level

    # name_rule_data = {
    #     "value": centene_format["ADDRESSEE"],
    #     "confidence": name_conf,
    #     "block": None,
    # }
    # ad1_rule_data = {
    #     "value": centene_format["ADDRESS_LINE_1"],
    #     "confidence": name_conf,
    #     "block": None,
    # }
    # ad2_rule_data = {
    #     "value": centene_format["ADDRESS_LINE_2"],
    #     "confidence": name_conf,
    #     "block": None,
    # }
    # city_rule_data = {
    #     "value": centene_format["CITY"],
    #     "confidence": name_conf,
    #     "block": None,
    # }
    # state_rule_data = {
    #     "value": centene_format["STATE"],
    #     "confidence": name_conf,
    #     "block": None,
    # }
    # zip_rule_data = {
    #     "value": centene_format["ZIP_CODE_4"],
    #     "confidence": name_conf,
    #     "block": None,
    # }
    # # reg_id moved to bottom to match where it is found on page
    # # labelers wanted the webpage they see to display regid as the last field
    # reg_id_rule_data = {
    #     "value": centene_format["REGULATORY_APPROVAL_ID"],
    #     "confidence": reg_id_conf,
    #     "block": None,
    # }
    # rules_data = {
    #     "ADDRESSEE": name_rule_data,
    #     "ADDRESS_LINE_1": ad1_rule_data,
    #     "ADDRESS_LINE_2": ad2_rule_data,
    #     "CITY": city_rule_data,
    #     "STATE": state_rule_data,
    #     "ZIP_CODE_4": zip_rule_data,
    #     "REGULATORY_APPROVAL_ID": reg_id_rule_data,
    # }

    # # Validate business rules
    # con = Condition(rules_data, rules)
    # rule_missed, rule_checked = con.check_all()
    # rule_missed_string = ", ".join(element["message"] for element in rule_missed)
    # centene_format["RULE_MISSED_COUNT"] = len(rule_missed)
    # centene_format["RULE_MISSED_STRING"] = rule_missed_string

    # # Save filenames
    # output_file_name = f"{key[:-4]}.json"
    # rm_file_name = f"{key[:-4]}-rule-missed.json"
    # rs_file_name = f"{key[:-4]}-rule-checked.json"

    # # Save data to s3 and return if save was successful or not
    # status = save_dict_to_s3(centene_format, output_bucket, output_file_name)
    # # Save rule missed list
    # s3.Object(output_bucket, rm_file_name).put(Body=json.dumps(rule_missed))
    # # Save rule satisfied list
    # s3.Object(output_bucket, rs_file_name).put(Body=json.dumps(rule_checked))

    # # -------Image for A2I--------------------------------------
    # s3_filename_a2i = f"{key[:-4]}-image-for-A2I.png"

    # # Combine the partial pages where the fields were found
    # if name_file and reg_id_file:
    #     name_file_parts = name_file.split("/")
    #     reg_id_file_parts = reg_id_file.split("/")
    #     tmp_name_file = "tmp/" + name_file_parts[-1]
    #     tmp_regid_file = "tmp/" + reg_id_file_parts[-1]
    #     # Recombine creates an image in tmp called image-for-A2I.png
    #     # using the top and bottom images where names and resp. codes
    #     # were found.
    #     recombine(tmp_name_file, tmp_regid_file)

    #     # We upload that to the correct bucket and prefix
    #     print(f"s3_filename is {s3_filename_a2i}")
    #     local_a2i_image_path = "tmp/image-for-A2I.png"
    #     client.upload_file(local_a2i_image_path, output_bucket, s3_filename_a2i)

    # else:  # CILT
    #     # upload cilt image to correct bucket with prefix

    #     local_a2i_image_path = "tmp/image-for-A2I.png"
    #     # upload cilt image
    #     client.upload_file("tmp/cilt.png", output_bucket, s3_filename_a2i)

    # # -----------------Create A2I labeling job--------------------------

    # # -----------------Clean up------------------------------------------
    # # Get disk usage of tmp directory
    # usage = os.statvfs("tmp")
    # total_space = usage.f_frsize * usage.f_blocks
    # used_space = usage.f_frsize * (usage.f_blocks - usage.f_bfree)
    # available_space = usage.f_frsize * usage.f_bavail

    # # Convert to human-readable format
    # total_space_gb = total_space / (1024**3)
    # used_space_gb = used_space / (1024**3)
    # available_space_gb = available_space / (1024**3)

    # # Print the disk usage
    # print(f"Total Space: {total_space_gb:.2f} GB")
    # print(f"Used Space: {used_space_gb:.2f} GB")
    # print(f"Available Space: {available_space_gb:.2f} GB")
    ################################################################################################

    # Iterate over all files in the directory
    for filename in os.listdir("tmp"):
        # if filename.endswith('.tif'):  # Check if the file ends with ".tif"
        file_path = os.path.join("tmp", filename)  # Get the full file path
        os.remove(file_path)  # Remove the file
        print(f"Removed file: {filename}")

    # return status, centene_format, rule_missed, rule_checked


def recover_from_aligned_candidates(
    textract_line_confidence,
    aligned_s1,
    aligned_s2,
    aligned_s3,
    dummychar="-",
):
    def get_char_if_not_dummy(ch):
        return ch if ch != dummychar else ""

    recovered_string = ""
    for c1, c2, c3 in zip(aligned_s1, aligned_s2, aligned_s3):
        if c1 == c2:
            recovered_string += get_char_if_not_dummy(c1)
        elif c1 == c3:
            recovered_string += get_char_if_not_dummy(c1)
        elif c2 == c3:
            recovered_string += get_char_if_not_dummy(c2)
        else:
            # No agreement, first go with Textract (c1) as we know our accuracy level with Textract
            if c1 != dummychar:
                recovered_string += c1
            elif c2 != dummychar:
                recovered_string += c2
            else:
                recovered_string += c3

    new_line_confidence = 0
    match_proportion = 1
    for c1, c2, c3 in zip(aligned_s1, aligned_s2, aligned_s3):
        # If all different, then use textract line confidence
        if c1 != c2 and c2 != c3 and c1 != c3:
            # cut confidence boost in half
            match_proportion *= 1 / 2

        # if all agree, then for that character maintain match proportion
        elif (c1 == c2) and (c2 == c3) and (c1 == c3):
            # maintain full confidence boost
            match_proportion *= 1

        else:  # two of three agree, reduce match proportion to 2/3 previously level
            # cut confidence boost by 10%
            match_proportion *= (9) / 10
    confidence_boost = (100 - textract_line_confidence) * (match_proportion)
    new_line_confidence = textract_line_confidence + confidence_boost

    return recovered_string, new_line_confidence


def needleman_wunsch(x, y):
    """Run the Needleman-Wunsch algorithm on two sequences.

    x, y -- sequences.

    Code based on pseudocode in Section 3 of:

    Naveed, Tahir; Siddiqui, Imitaz Saeed; Ahmed, Shaftab.
    "Parallel Needleman-Wunsch Algorithm for Grid." n.d.
    https://upload.wikimedia.org/wikipedia/en/c/c4/ParallelNeedlemanAlgorithm.pdf
    """
    N, M = len(x), len(y)
    s = lambda a, b: int(a == b)

    DIAG = -1, -1
    LEFT = -1, 0
    UP = 0, -1

    # Create tables F and Ptr
    F = {}
    Ptr = {}

    F[-1, -1] = 0
    for i in range(N):
        F[i, -1] = -i
    for j in range(M):
        F[-1, j] = -j

    option_Ptr = DIAG, LEFT, UP
    for i, j in product(range(N), range(M)):
        option_F = (
            F[i - 1, j - 1] + s(x[i], y[j]),
            F[i - 1, j] - 1,
            F[i, j - 1] - 1,
        )
        F[i, j], Ptr[i, j] = max(zip(option_F, option_Ptr))

    # Work backwards from (N - 1, M - 1) to (0, 0)
    # to find the best alignment.
    alignment = deque()
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
        direction = Ptr[i, j]
        if direction == DIAG:
            element = i, j
        elif direction == LEFT:
            element = i, None
        elif direction == UP:
            element = None, j
        alignment.appendleft(element)
        di, dj = direction
        i, j = i + di, j + dj
    while i >= 0:
        alignment.appendleft((i, None))
        i -= 1
    while j >= 0:
        alignment.appendleft((None, j))
        j -= 1

    return list(alignment)


def get_two_alignment(
    seq1, seq2, gap_penalty=-1, match_score=1, mismatch_penalty=-5, dummychar="-"
):
    # Initialize the scoring matrix
    def create_matrix(dimensions):
        rows, cols = dimensions
        return [[0] * cols for _ in range(rows)]

    rows = len(seq1) + 1
    cols = len(seq2) + 1
    scores = create_matrix((rows, cols))
    pointers = create_matrix((rows, cols))
    alignment = []

    # Fill the first row and column with gap penalties
    for i in range(rows):
        scores[i][0] = i * gap_penalty
        pointers[i][0] = (i - 1, 0)
    for j in range(cols):
        scores[0][j] = j * gap_penalty
        pointers[0][j] = (0, j - 1)

    # Calculate the scores and pointers for each cell
    for i in range(1, rows):
        for j in range(1, cols):
            match = scores[i - 1][j - 1] + (
                match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
            )
            delete = scores[i - 1][j] + gap_penalty
            insert = scores[i][j - 1] + gap_penalty
            scores[i][j] = max(match, delete, insert)
            if scores[i][j] == match:
                pointers[i][j] = (i - 1, j - 1)
            elif scores[i][j] == delete:
                pointers[i][j] = (i - 1, j)
            else:
                pointers[i][j] = (i, j - 1)

    # Traceback to construct the alignment
    i, j = rows - 1, cols - 1
    while i > 0 or j > 0:
        di, dj = pointers[i][j]
        if di == i - 1 and dj == j - 1:
            aligned_chars = (seq1[i - 1], seq2[j - 1])
        elif di == i - 1 and dj == j:
            aligned_chars = (seq1[i - 1], dummychar)
        else:
            aligned_chars = (dummychar, seq2[j - 1])
        alignment.append(aligned_chars)
        i, j = di, dj

    return alignment[::-1], scores


def get_three_alignment(
    seq1,
    seq2,
    seq3,
    gap_penalty=-1,
    match_score=1,
    half_match_score=0.5,
    mismatch_score=-100,
    half_mismatch_penalty=-50,
    dummychar="-",
):
    def create_matrix(dimensions):
        rows, cols, depth = dimensions
        return [[[0] * depth for _ in range(cols)] for _ in range(rows)]

    # Initialize the scoring matrix
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    depth = len(seq3) + 1
    scores = create_matrix((rows, cols, depth))
    pointers = create_matrix((rows, cols, depth))
    alignments = []

    def get_aligned_word(i):
        return "".join([chars[i] for chars in alignment])

    # Fill the first row, column, and depth with gap penalties
    for i in range(rows):
        scores[i][0][0] = i * gap_penalty
        pointers[i][0][0] = (i - 1, 0, 0)
    scores_jk = get_two_alignment(
        seq2,
        seq3,
        gap_penalty=gap_penalty,
        match_score=half_match_score,
        mismatch_penalty=half_mismatch_penalty,
        dummychar="-",
    )[1]
    for j in range(cols):
        for k in range(depth):
            scores[0][j][k] = scores_jk[j][k]

    for j in range(cols):
        scores[0][j][0] = j * gap_penalty
        pointers[0][j][0] = (0, j - 1, 0)
    scores_ik = get_two_alignment(
        seq1,
        seq3,
        gap_penalty=gap_penalty,
        match_score=half_match_score,
        mismatch_penalty=half_mismatch_penalty,
        dummychar="-",
    )[1]
    for i in range(rows):
        for k in range(depth):
            scores[i][0][k] = scores_ik[i][k]

    for k in range(depth):
        scores[0][0][k] = k * gap_penalty
        pointers[0][0][k] = (0, 0, k - 1)
    scores_ij = get_two_alignment(
        seq1,
        seq2,
        gap_penalty=gap_penalty,
        match_score=half_match_score,
        mismatch_penalty=half_mismatch_penalty,
        dummychar="-",
    )[1]
    for i in range(rows):
        for j in range(cols):
            scores[i][j][0] = scores_ij[i][j]

    # Calculate the scores and pointers for each cell
    for i in range(1, rows):
        for j in range(1, cols):
            for k in range(1, depth):
                match = scores[i - 1][j - 1][k - 1] + (
                    match_score
                    if seq1[i - 1] == seq2[j - 1] == seq3[k - 1]
                    else mismatch_score
                )
                match_ij = scores[i - 1][j - 1][k] + (
                    half_match_score
                    if seq1[i - 1] == seq2[j - 1]
                    else half_mismatch_penalty
                )
                match_ik = scores[i - 1][j][k - 1] + (
                    half_match_score
                    if seq1[i - 1] == seq3[k - 1]
                    else half_mismatch_penalty
                )
                match_jk = scores[i][j - 1][k - 1] + (
                    half_match_score
                    if seq2[j - 1] == seq3[k - 1]
                    else half_mismatch_penalty
                )
                insert_i = scores[i - 1][j][k] + gap_penalty
                insert_j = scores[i][j - 1][k] + gap_penalty
                insert_k = scores[i][j][k - 1] + gap_penalty
                scores[i][j][k] = max(
                    match, insert_i, insert_j, insert_k, match_ij, match_ik, match_jk
                )
                if scores[i][j][k] == match:
                    pointers[i][j][k] = (i - 1, j - 1, k - 1)
                elif scores[i][j][k] == match_ij:
                    pointers[i][j][k] = (i - 1, j - 1, k)
                elif scores[i][j][k] == match_ik:
                    pointers[i][j][k] = (i - 1, j, k - 1)
                elif scores[i][j][k] == match_jk:
                    pointers[i][j][k] = (i, j - 1, k - 1)
                elif scores[i][j][k] == insert_i:
                    pointers[i][j][k] = (i - 1, j, k)
                elif scores[i][j][k] == insert_j:
                    pointers[i][j][k] = (i, j - 1, k)
                else:
                    pointers[i][j][k] = (i, j, k - 1)

    # Traceback to construct the alignments
    i, j, k = rows - 1, cols - 1, depth - 1
    while i > 0 or j > 0 or k > 0:
        if sum([i == 0, j == 0, k == 0]) == 1:
            if i == 0:
                align_2 = get_two_alignment(
                    seq2[:j], seq3[:k], gap_penalty, match_score, mismatch_score
                )[0]
                align_2 = [
                    (dummychar, seq2_ch, seq3_ch) for seq2_ch, seq3_ch in align_2
                ]
            elif j == 0:
                align_2 = get_two_alignment(
                    seq1[:i], seq3[:k], gap_penalty, match_score, mismatch_score
                )[0]
                align_2 = [
                    (seq1_ch, dummychar, seq3_ch) for seq1_ch, seq3_ch in align_2
                ]
            else:
                align_2 = get_two_alignment(
                    seq1[:i], seq2[:j], gap_penalty, match_score, mismatch_score
                )[0]
                align_2 = [
                    (seq1_ch, seq2_ch, dummychar) for seq1_ch, seq2_ch in align_2
                ]
            align_2.extend(alignments[::-1])
            alignment = align_2
            return get_aligned_word(0), get_aligned_word(1), get_aligned_word(2)
        di, dj, dk = pointers[i][j][k]
        if di == i - 1 and dj == j - 1 and dk == k - 1:
            alignments.append((seq1[i - 1], seq2[j - 1], seq3[k - 1]))
        elif di == i - 1 and dj == j - 1 and dk == k:
            alignments.append((seq1[i - 1], seq2[j - 1], dummychar))
        elif di == i - 1 and dj == j and dk == k - 1:
            alignments.append((seq1[i - 1], dummychar, seq3[k - 1]))
        elif di == i and dj == j - 1 and dk == k - 1:
            alignments.append((dummychar, seq2[j - 1], seq3[k - 1]))
        elif di == i - 1 and dj == j and dk == k:
            alignments.append((seq1[i - 1], dummychar, dummychar))
        elif di == i and dj == j - 1 and dk == k:
            alignments.append((dummychar, seq2[j - 1], dummychar))
        else:
            alignments.append((dummychar, dummychar, seq3[k - 1]))
        i, j, k = di, dj, dk

    alignment = alignments[::-1]
    return get_aligned_word(0), get_aligned_word(1), get_aligned_word(2)




# Example usage
# sequence1 = "NA2WCME.OB79520E_.2022"
# sequence2 = "NA2WCMEOB79520E_2022"
# sequence3 = "NA.2WCMEOB79520E_2022"

# s1, s2, s3 = get_three_alignment(sequence1, sequence2, sequence3)

# print(s1)
# print(s2)
# print(s3)


def get_event(tif_key):
    event = {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": "2019-09-03T19:37:27.192Z",
                "eventName": "ObjectCreated:Put",
                "userIdentity": {"principalId": "AWS:AIDAINPONIXQXHT3IKHL2"},
                "requestParameters": {"sourceIPAddress": "205.255.255.255"},
                "responseElements": {
                    "x-amz-request-id": "D82B88E5F771F645",
                    "x-amz-id-2": "vlR7PnpV2Ce81l0PRw6jlUpck7Jo5ZsQjryTjKlc5aLWGVHPZLj5NeC6qMa0emYBDXOo6QBU0Wo=",
                },
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "configurationId": "828aa6fc-f7b5-4305-8584-487c791949c1",
                    "bucket": {
                        "name": test_bucket_name,
                        "ownerIdentity": {"principalId": "A3I5XTEXAMAI3E"},
                        "arn": "arn:aws:s3:::lambda-artifacts-deafc19498e3f2df",
                    },
                    "object": {
                        "key": tif_key,
                        "size": 1305107,
                        "eTag": "b21b84d653bb07b05b1e6b33684dc11b",
                        "sequencer": "0C0F6F405D6ED209E1",
                    },
                },
            }
        ]
    }
    return event


# INPUT SETUP
########################################################################################################################
# tif_keys_list = [f"real-doc/C00194251{i}.tiff" for i in range(10) if i != 3]


tif_keys_list = sorted(
    [f"real-doc/{file_name}" for file_name in os.listdir("./070723sample")]
)


events = [get_event(tif_key) for tif_key in tif_keys_list if tif_key[-1] in {"f", "F"}]

# output_suffix = (
#     "idp-textract-prod10525-target 06292023 CUR6470-01_4010_20230628_0001157189"
# )

output_suffix = "centenetransfer Centenetesting"
########################################################################################################################
extraction_log_df = pd.DataFrame()

dummy_extration_dict = {}
dummy_extration_dict["app_code_type"] = None
dummy_extration_dict["twenty_code_type"] = None
dummy_extration_dict["textract_conf_app_code"] = None
dummy_extration_dict["textract_conf_twenty_code"] = None
dummy_extration_dict["tesseract_conf_app_code"] = None
dummy_extration_dict["tesseract_conf_twenty_code"] = None
dummy_extration_dict["doc"] = None
dummy_extration_dict["RegId"] = None
dummy_extration_dict["RegIdConf"] = None
dummy_extration_dict["app_code_textract"] = None
dummy_extration_dict["app_code_tesseract"] = None
dummy_extration_dict["app_code_table_lookup"] = None
dummy_extration_dict["twenty_code_textract"] = None
dummy_extration_dict["twenty_code_tesseract"] = None
dummy_extration_dict["twenty_code_table_lookup"] = None
dummy_extration_dict["app_code_fallback"] = None
dummy_extration_dict["twenty_code_fallback"] = None
dummy_extration_dict["approval_back_up"] = None
dummy_extration_dict["twenty_back_up"] = None
dummy_extration_dict["history"] = None

extraction_log_df = pd.concat(
    [extraction_log_df, pd.DataFrame([dummy_extration_dict])],
    ignore_index=True,
)

# print(
#     "\n".join(
#         get_three_alignment(
#             "NA3WCMEOB00186E 0000", "NA3WCMEOBO00186E_0000",  "NA3WCMEOB00186E_0000"
#         )
#     )
# )


# string, score = recover_from_aligned_candidates(
#     70,
#     *get_three_alignment(
#             "NA3WCMEOB00186E 0000", "NA3WCMEOBO00186E_0000",  "NA3WCMEOB00186E_0000"
#         )
# )

# print(string)

if __name__ == "__main__":
    # m = first_match_info = extractOne(
    #             '___NA2PDGLTR11297E_0000_   ',
    #             twenty_digit_codes,
    #             scorer=ratio,
    #             score_cutoff=80,
    #             processor=default_process,
    #         )
    # print(m[0])
    # print(m[1])
    for event in events[-19:-18]:
        lambda_handler(event, None)
    extraction_log_df.to_csv(f"outputs/extraction_log_{output_suffix}.csv")
#     for i in range(0, 1):
#         for j in range(0, 1):
#             for k in range(0, 1):
#                 string, score = recover_from_aligned_candidates(
#                     70,
#                     *get_three_alignment(
#                         "Y0020_WCM_100186E_C INTERNAL APPROVED 07252022",
#                         "Y0020_WCM_100186E_ C INTERNAL SMAL APPROVED 07252022",
#                         "Y0020 WCM 100186E_C INTERNAL APPROVED 07252022",
#                         gap_penalty=-10,
#                         half_match_score=-5,
#                     ),
#                 )
#                 print(f"{string} with {score}")


# print(
#     "\n".join(
#         get_three_alignment(
#             "Y0020_WCM_100186E_C INTERNAL APPROVED 07252022",
#             "Y0020_WCM_100186E_ C INTERNAL SMAL APPROVED 07252022",
#             "Y0020_WCM_100186E_C INTERNAL APPROVED 07252022",
#             gap_penalty=-10,
#             half_match_score=-5,
#         )
#     )
# )
# import random


# def sample_across_batches(bucket_name, folder_path, num_files=2, local_folder='sample', s3_client=client):
#     # Retrieve the list of subfolders in the specified folder
#     response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path, Delimiter='/')
#     subfolders = [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

#     # Sample uniformly at random from the subfolders to download files from
#     selected_subfolders = random.choices(subfolders, k=num_files)

#     # Create the local target folder if it does not exist
#     if not os.path.isdir(local_folder):
#         os.mkdir(local_folder)

#     # Download the files
#     for subfolder in selected_subfolders:
#         # Retrieve the list of objects (files) in the subfolder
#         response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=subfolder)
#         files = [file['Key'] for file in response.get('Contents', [])]

#         # Sample a file uniformly at random
#         file_to_download = random.choice(files[1:])

#         # Download the selected file
#         local_file_name = '-'.join(file_to_download.split('/')[-2:])
#         s3_client.download_file(bucket_name, file_to_download, os.path.join(local_folder, local_file_name))

# if __name__ == "__main__":
#     sample_across_batches('centene-test', 'real-doc/')
