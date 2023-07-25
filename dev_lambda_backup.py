from utils.alignment import (
    get_two_alignment,
    get_three_alignment,
    recover_from_aligned_candidates,
)

from utils.reg_id import get_reg_id
from utils.rules import get_missed_and_checked_rules, rules, Condition
from utils.image_processing import *

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
import os
from utils.misc import tmp_folder, get_last_dot_index


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




# os.environ["BUCKET"] = data_bucket
# os.environ["REGION"] = region
# role = sm.get_execution_role()

# ----------------Load regID table values-----------------------
response = client.get_object(Bucket=tables_bucket, Key="twenty_codes.json")
content = response["Body"].read().decode("utf-8")
twenty_digit_codes = json.loads(content)
twenty_digit_codes.sort(key=lambda x: x[-4:])  # order by year descending

response = client.get_object(Bucket=tables_bucket, Key="approval_codes.json")
content = response["Body"].read().decode("utf-8")
reg_id_approved = json.loads(content)
reg_id_approved
# ----------------- Queries---------------



queries = [
    {"Text": "Who is the person?", "Alias": "ADDRESSEE"},
    {"Text": "What is the street address of the person?", "Alias": "STREET_ADDRESS"},
    {"Text": "What is the city of the person?", "Alias": "CITY"},
    {"Text": "What is the state of the person?", "Alias": "STATE"},
    {"Text": "What is the zip code of the person?", "Alias": "ZIP_CODE_4"},
]

# --------------- Helper Functions to call textract APIs ------------------


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


def parse_response_to_json(response_json, query_output):
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


# --------------- Main handler ------------------
def lambda_handler(event, context):
    """Demonstrates S3 trigger that uses
    textract APIs to detect text, query text in S3 Object.
    """

    for filename in os.listdir(tmp_folder):
        # if filename.endswith('.tif'):  # Check if the file ends with ".tif"
        file_path = os.path.join("tmp", filename)  # Get the full file path
        os.remove(file_path)  # Remove the file
        print(f"Removed file: {filename}")

    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event
    bucket = event["Records"][0]["s3"]["bucket"]["name"]
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    print(key)
    # --------------------Clean up tmp----------------------------------
    for filename in os.listdir(tmp_folder):
        file_path = os.path.join(tmp_folder, filename)  # Get the full file path
        os.remove(file_path)  # Remove the file

    # ------------Image processing on TIF ----------------
    ## save TIF
    local_file_name = os.path.join(tmp_folder, os.path.basename(key))
    # s3.download_file(Bucket=bucket, Key=key, Filename=local_file_name)
    # inFile = open(local_file_name, "r")
    print(f"local_file_name is: {local_file_name}")

    # Download file from S3 to local /tmp directory
    try:
        s3.Object(bucket, key).download_file(local_file_name)
        print(f"File downloaded successfully from S3.")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        
    # given the local_file_name of the original multipage .TIF file
    # we split it into pages and for each page, further split
    # into top and bottom.
    split_tif(local_file_name)

    full_response = False
    number_of_pages_to_check = 1
    # first check the first number_of_pages_to_check pages only 
    # if this didn't resulted in a full response to the whole document
    # usually the required answer is contained in the first 2 pages
    # number_of_pages_to_check = -1 means check all pages
    while not full_response and number_of_pages_to_check >= -1:
        
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
            
        
        _, status, rule_missed, rule_checked, full_response = \
            get_response(bucket, key, local_file_name, 0, number_of_pages_to_check, centene_format, query_output)
        number_of_pages_to_check = -1 if number_of_pages_to_check > -1 else -2
        

    # -----------------Clean up------------------------------------------
    # Get disk usage of /tmp directory
    usage = os.statvfs(tmp_folder)
    total_space = usage.f_frsize * usage.f_blocks
    used_space = usage.f_frsize * (usage.f_blocks - usage.f_bfree)
    available_space = usage.f_frsize * usage.f_bavail

    # Convert to human-readable format
    total_space_gb = total_space / (1024**3)
    used_space_gb = used_space / (1024**3)
    available_space_gb = available_space / (1024**3)

    # Print the disk usage
    print(f"Total Space: {total_space_gb:.2f} GB")
    print(f"Used Space: {used_space_gb:.2f} GB")
    print(f"Available Space: {available_space_gb:.2f} GB")


    # Iterate over all files in the directory
    for filename in os.listdir(tmp_folder):
        # if filename.endswith('.tif'):  # Check if the file ends with ".tif"
        file_path = os.path.join(tmp_folder, filename)  # Get the full file path
        os.remove(file_path)  # Remove the file
        print(f"Removed file: {filename}")
        

    print(json.dumps(centene_format))
    return status, centene_format, rule_missed, rule_checked 


def get_response(bucket, key, local_file_name, lo_page, hi_page, centene_format, query_output):
    reg_id_code_found = False
    non_reg_id_data_found = False
    split_into_pages_top_and_bottom(local_file_name, lo_page, hi_page)

    # directory = tmp_folder  # Replace with your directory path
    # print('currently in tmp')
    # for filename in os.listdir(directory):
    #    print(filename)

    # Insert 'file' key pair
    # Split the string by '/'
    parts = key.split("/")

    # Get the text after the final '/'
    # ricoh_dcn = parts[-1]

    ##########################################################
    centene_format['RICOH_DCN'] = parts[-1][:-4] #strip .T
    # centene_format['BATCH'] = parts[-2]
    # centene_format['BATCH_PREFIX'] = parts[-3]
    ##########################################################

    # Save to S3 in correct place in output bucket

    # Get Regid

    bottom_list = []
    top_list = []
    reg_id_conf = 0
    files = os.listdir(tmp_folder)
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
            client.upload_file(
                os.path.join(tmp_folder, file), output_bucket, s3_filename
            )
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
            reg_id, reg_id_conf = get_reg_id(bottom_response, file.split("/")[-1])
            if reg_id_conf > 0:
                print(f"RegID code found in {file}.")
                print(reg_id, reg_id_conf)
                # save location
                reg_id_file = file

                centene_format["REGULATORY_APPROVAL_ID"] = reg_id
                reg_id_code_found = True
                break

    # In case no name or no regid found, make sure regid blank
    if not reg_id_file:
        centene_format["REGULATORY_APPROVAL_ID"] = ""

    query_response = []
    # Define to be cilt if none
    name_file = None
    name_conf = 0
    top_list = sorted(top_list, key=lambda x: x[-13:])
    print(f"Top list is: {top_list}")

    for file in top_list:
        print(f"Top file is : {file}")
        try:
            # Calls textract detect_document_text API to detect text in the document S3 object
            # text_response = detect_text(bucket, key)
            print(f"Calling Textract on {file}")
            # Calls textract analyze_document API to query S3 object
            query_response = textract.analyze_document(
                Document={"S3Object": {"Bucket": output_bucket, "Name": file}},
                FeatureTypes=["QUERIES"],
                QueriesConfig={"Queries": queries},
            )
        except Exception as e:
            print(e)
            print(
                "Error processing object {} from bucket {}. ".format(key, bucket)
                + "Make sure your object and bucket exist and your bucket is in the same region as this function."
            )
            # Save error to file
            # status = save_dict_to_s3(centene_format, output_bucket,output_file_name)
            # print(status)
            raise e
        if query_response:
            # Store json of parsed response of first page
            query_data = parse_response_to_json(query_response, query_output)

            # Check if query name available
            if query_data["ADDRESSEE"]:
                non_reg_id_data_found = True
                print(f"Name found on {file}")
                name_file = file
                name = query_data["ADDRESSEE"]["value"]
                name_conf = query_data["ADDRESSEE"]["confidence"]

                # Function get_next_line searches for name in lines, if not found, returns null
                # If ad1 returns null, then query name wrong or does not exist in document lines
                ad1 = get_next_line(name, query_response)

                if (
                    ad1
                ):  # If ad1 not null, query name is correct, so use it to get fields
                    ad2 = get_next_line(ad1, query_response)
                    ad3 = get_next_line(ad2, query_response)

                    if not ad3:  # If no 3rd address line
                        centene_format["ADDRESSEE"] = name
                        centene_format["ADDRESS_LINE_1"] = ad1
                        centene_format["ADDRESS_LINE_2"] = ""
                        centene_format["CITY"] = get_city(ad2)
                        centene_format["STATE"] = get_state(ad2)
                        centene_format["ZIP_CODE_4"] = get_zip(ad2)
                    else:  # If there is a 3rd address line
                        centene_format["ADDRESSEE"] = name
                        centene_format["ADDRESS_LINE_1"] = ad1
                        centene_format["ADDRESS_LINE_2"] = ad2
                        centene_format["CITY"] = get_city(ad3)
                        centene_format["STATE"] = get_state(ad3)
                        centene_format["ZIP_CODE_4"] = get_zip(ad3)
                else:  # Query name is wrong or doesn't exist, pass to query defaults
                    street_address = ""
                    city = ""
                    state = ""
                    zip_code = ""

                    # If queries found fields use them
                    if query_data["STREET_ADDRESS"]:
                        street_address = query_data["STREET_ADDRESS"]["value"]
                    if query_data["CITY"]:
                        city = query_data["CITY"]["value"]
                    if query_data["STATE"]:
                        state = query_data["STATE"]["value"]
                    if query_data["ZIP_CODE_4"]:
                        zip_code = query_data["ZIP_CODE_4"]["value"]

                    # Now input the values we have
                    centene_format["ADDRESSEE"] = name
                    centene_format["ADDRESS_LINE_1"] = street_address
                    centene_format["ADDRESS_LINE_2"] = ""
                    centene_format["CITY"] = city
                    centene_format["STATE"] = state
                    centene_format["ZIP_CODE_4"] = zip_code

                break  # break out of loop

    # Finally, update the query_data with new confidence values
    # and values based on what we found
    # query_data
    # TEMPORARY: make fields derived from name have same confidence
    # We may want to pass in the confidence on the line (e.g. on City line, state line
    # etc. instead
    # OPTIONAL: Make confidence minimum of all fields, that way all fields
    # show in the A2I when one field is below cutoff level

    rule_missed, rule_checked = get_missed_and_checked_rules(
        reg_id_conf, name_conf, centene_format, rules
    )

    rule_missed_string = ", ".join(element["message"] for element in rule_missed)
    centene_format["RULE_MISSED_COUNT"] = len(rule_missed)
    centene_format["RULE_MISSED_STRING"] = rule_missed_string

    # Save filenames
    output_file_name = f"{key[:get_last_dot_index(key)]}.json"
    rm_file_name = f"{key[:get_last_dot_index(key)]}-rule-missed.json"
    rs_file_name = f"{key[:get_last_dot_index(key)]}-rule-checked.json"

    # Save data to s3 and return if save was successful or not
    status = save_dict_to_s3(centene_format, output_bucket, output_file_name)
    # Save rule missed list
    s3.Object(output_bucket, rm_file_name).put(Body=json.dumps(rule_missed))
    # Save rule satisfied list
    s3.Object(output_bucket, rs_file_name).put(Body=json.dumps(rule_checked))

    # -------Image for A2I--------------------------------------
    s3_filename_a2i = f"{key[:get_last_dot_index(key)]}-image-for-A2I.png"

    # Combine the partial pages where the fields were found
    if name_file and reg_id_file:
        name_file_parts = name_file.split("/")
        reg_id_file_parts = reg_id_file.split("/")
        tmp_name_file = os.path.join(tmp_folder, name_file_parts[-1])
        tmp_regid_file = os.path.join(tmp_folder, reg_id_file_parts[-1])
        # Recombine creates an image in /tmp called image-for-A2I.png
        # using the top and bottom images where names and resp. codes
        # were found.
        recombine(tmp_name_file, tmp_regid_file)

        # We upload that to the correct bucket and prefix
        print(f"s3_filename is {s3_filename_a2i}")
        local_a2i_image_path = os.path.join(tmp_folder, "image-for-A2I.png")
        client.upload_file(local_a2i_image_path, output_bucket, s3_filename_a2i)

    else:  # CILT
        # upload cilt image to correct bucket with prefix
        local_a2i_image_path = os.path.join(tmp_folder, "image-for-A2I.png")
        # upload cilt image
        client.upload_file(
            os.path.join(tmp_folder, "cilt.png"), output_bucket, s3_filename_a2i
        )

    return centene_format, status, rule_missed, rule_checked, non_reg_id_data_found and reg_id_code_found



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
    [f"real-doc/{file_name}" for file_name in os.listdir("./problematic_samples/072423")]
)


events = [get_event(tif_key) for tif_key in tif_keys_list if tif_key[-1] in {"f", "F"}]

if __name__ == "__main__":
    for event in events[:1]:
        lambda_handler(event, None)
