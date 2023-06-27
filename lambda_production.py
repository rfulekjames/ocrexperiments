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


print('Loading function')

textract = boto3.client('textract')
s3 = boto3.resource('s3')
client = s3.meta.client #used for interacting with client for convenience

region = boto3.session.Session().region_name

a2i=boto3.client('sagemaker-a2i-runtime', region_name=region)

#Must be different from trigger bucket
#Lambda IAM role only has write permission to this bucket
tables_bucket = 'ibp-textract-prod1-output'
output_bucket = 'ibp-textract-prod1-output'
data_bucket = 'sagemaker-us-west-2-638834599306'


#os.environ["BUCKET"] = data_bucket
#os.environ["REGION"] = region
#role = sm.get_execution_role()

#----------------Load regID table values-----------------------
response = client.get_object(Bucket=tables_bucket, Key='twenty_codes.json')
content = response['Body'].read().decode('utf-8')
twenty_digit_codes = json.loads(content)
twenty_digit_codes.sort(key=lambda x : x[-4:])#order by year descending

response = client.get_object(Bucket=tables_bucket, Key='approval_codes.json')
content = response['Body'].read().decode('utf-8')
reg_id_approved = json.loads(content)
reg_id_approved
#-----------------Ouput, Queries and Rules ---------------
# JSON structure to hold the extraction result


queries = [ 
    {
        'Text':'Who is the person?',
        'Alias': 'ADDRESSEE'
    },
    {
        'Text': 'What is the street address of the person?',
        'Alias': 'STREET_ADDRESS'
    },
    {
        'Text': 'What is the city of the person?',
        'Alias': 'CITY'
    },
    {
        'Text': 'What is the state of the person?',
        'Alias': 'STATE'
    },
    {
        'Text': 'What is the zip code of the person?',
        'Alias': 'ZIP_CODE_4'
    }
    ]
#confidence_threshold = 101 #For manual verification of all docs
confidence_threshold = 95 #cutoff for automatic verification
rules = [
    {
        'description': f'ADDRESSEE confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'ADDRESSEE',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    },
    {
        'description': f'ADDRESS_LINE_1 confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'ADDRESS_LINE_1',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    },
    {
        'description': f'ADDRESS_LINE_2 confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'ADDRESS_LINE_2',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    },
    {
        'description': f'CITY confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'CITY',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    },
    {
        'description': f'STATE confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'STATE',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    },
    {
        'description': f'ZIP_CODE_4 confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'ZIP_CODE_4',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    },
    {
        'description': f'REGULATORY_APPROVAL_ID confidence score should be greater than or equal to {confidence_threshold}',
        'field_name': 'REGULATORY_APPROVAL_ID',
        'field_name_regex': None, # support Regex: '_confidence$',
        'condition_category': 'Confidence',
        'condition_type': 'ConfidenceThreshold',
        'condition_setting': confidence_threshold,
    }
    ]
#----------------Image processing -----------------------------------------
def extract_bounding_image(filepath,bounding_box):
    '''Given Textract bounding box, use Image Magick to
    extract as tif image the area in the bounding box plus a
    5% margin. Save to tmp'''
    save_path = f'/tmp/{filepath[:-4]}-bounding-box.tif'
    command = ['convert', filepath, '-crop', '110%x110%+',x_offset,'+',y_offset, save_path]
    try:
        subprocess.run(command, check=True)
        print("Image splitting completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")
        
    
    
def split_tif(filepath):
    print(filepath)
    
    command = ['convert', filepath, filepath[:-4]+ '-%02d.tif']
    command2 = ['convert', filepath, '-crop', '100%x100%', '+repage', '-write', filepath[:-4]+'-%02d.tif', 'null:']

    try:
        subprocess.run(command, check=True)
        print("Image splitting completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")
        
    # list the TIFF individual pages
    page_files = []

    # Get a list of the generated page files
    for file in os.listdir('/tmp'):
        if file.endswith('.tif'):
            page_files.append('/tmp/' +file)
            
    # Combine the individual pages into a single PNG file
    cilt_name = '/tmp/cilt.png'
    combine_command = ['convert'] + page_files + ['-append',cilt_name]
    try:       
        # Execute the command to combine the pages into a PNG file
        subprocess.run(combine_command, check=True)
        print("Image CILT combining completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image CILT combining failed: {e}")

def extract_bottom(input_path, output_path):
    command = ['convert', input_path, '-gravity', 'south', '-crop', '100%x15%', output_path]
    
    try:
        subprocess.run(command, check=True)
        print("Bottom extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")

def extract_top(input_path, output_path):
    command = ['convert', input_path, '-gravity', 'north', '-crop', '100%x85%', output_path]
    try:
        subprocess.run(command, check=True)
        print("Top extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")
        
def split_into_pages_top_and_bottom(filepath):
    #given the filepath of the original multipage .TIF file
    # we split it into pages and for each page, further split
    #into top and bottom.
    split_tif(filepath)
    #iterate over pages
    files = os.listdir('/tmp')
    for file in files:
        if '-' and '.tif' in file:
            print("Splitting the following page into the two files below:")
            bottom_filename = '/tmp/'+ file[:-4] + '-bottom.tif'
            print(bottom_filename)
            top_filename = '/tmp/' + file[:-4] + '-top.tif'
            print(top_filename)
            extract_bottom('/tmp/'+file, bottom_filename)
            extract_top('/tmp/'+file, top_filename)
            
def add_border(filepath):
    
    input_file = filepath
    output_file = filepath[:-4]+'-border.png'

    command = ['convert',input_file,'-bordercolor', 'lime','-border', '5x5', output_file]
    try:
        subprocess.run(command, check=True)
        print("Border addition completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")

def recombine(top_path, bottom_path):
    #add border to both top and bottom image
    add_border(top_path)
    add_border(bottom_path)
    
    #append bottom to top and compress as png
    command = ['convert',
               top_path[:-4]+'-border.png',
               bottom_path[:-4]+'-border.png',
               '-append',
               '-define',
               'png:compression-filter=5',
               '-define',
               'png:compression-level=9',
               '-define',
               'png:compression-strategy=1',
               '/tmp/image-for-A2I.png'
              ]
    try:
        subprocess.run(command, check=True)
        print("Image append completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Image extraction failed: {e}")
    
    os.remove(top_path[:-4]+'-border.png')  # Remove the intermediate files
    os.remove(bottom_path[:-4]+'-border.png')            
# --------------- Helper Functions to call textract APIs ------------------
#copy

def get_reg_id(response_json):
    '''To obtain the reg id using AWS Textract Detect Document Text.
    Outputs a tuple: string consisting of regulatory approval id and confidence score.
    '''
    #Default values
    twenty_code, twenty_max_conf = '',0
    approval_code, approval_max_conf = '',0
    
    approval_re = re.compile(r'[(A-Z0-9_]+[_ ].*?(?:MATERIALS?|MODIFIED|MODIFIED_2023|ACCEPTED|SPN|)\d{7,8}')
    #approval_re = re.compile(r'[A-Z][A-Z0-9]+[_ ].*\d\d\d\d\d\d\d\d')
    #twenty_re_old = re.compile(r'[A-Z][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9][A-Z0-9]+[ _.A-Z]\d{3,4}')
    twenty_re = re.compile(r'[A-Z][A-Z][O0-9][A-Z][A-Z][A-Z][A-Z0-9][A-Z0-9][A-Z0-9]+[ _.][A-Z\d]{3,4}$') 

    if 'Blocks' in response_json:
        #Get all lines
        lines = [(item['Text'], item['Confidence']) for item in response_json['Blocks'] if item['BlockType'] == 'LINE']
        #Examine last lines
        for ln in lines:
            #Get line text and confidence
            ln_text = ln[0].upper()
            ln_conf = ln[1]
            #print(ln_text)
            #print(ln_text)
            #print(ln_conf)
            #Check if approval code in line
            ln_approval = approval_re.findall(ln_text)
            #Check if 20 code in line
            ln_twenty = twenty_re.findall(ln_text)
            #ln_twenty2 = twenty_re2.findall(ln_text)
            #If approval code found and we have not found one yet
            #How to choose cutoff? We have done limited experimentation. This is a guess that seemed to perform well in the handful of tests we did
            if ln_approval and approval_code == '':
                extracted_app = ln_approval[0]    
                #The processor removes all non alphanumeric characters, trims whitespace, converting all characters to lower case, then does the comparison
                match_info = extractOne(extracted_app, reg_id_approved, scorer=ratio, score_cutoff=80, processor=default_process)
                #Check if close match to result in table
                #search through table for best match, cutoff improves speed...if no
                #score for entry in table lookup lower than cutoff then further processing
                #on that entry stopped. If all items in table below cutoff then highest
                #score among them returned
                #If match above cutoff found then format close to table format
                if match_info:
                    ln_match = match_info[0]
                    ln_match_conf = match_info[1]
                    #If line confidence exceeds threshold, and format close to table format, then get raw Textract extraction
                    if (ln_conf > confidence_threshold) and ln_match_conf > confidence_threshold:
                        approval_code = extracted_app
                        approval_max_conf = ln_conf
                        print('Raw output used for app. code')
                        print(approval_code)
                        print(approval_max_conf)
                        print(ln_text)
                        print(ln_conf)

                    else: # ln_match_conf > confidence_threshold: #If  confidence lower, use table 
                        approval_code = ln_match
                        #we don't want confidence to be 100 just because a perfect match was found in the table: it could be we misread a character
                        #and matched the wrong one due to bad Textract read. So, we multiply line confidence by match confidence (match_info[1]).
                        #In this way confidence we return is at most equal to cutoff when table is used.
                        approval_max_conf = match_info[1]*(ln_conf/100)
                        print('Table used for app. code')
                        print(approval_code)
                        print(approval_max_conf)
                        print(match_info[1])
                        print(ln_text)
                        print(ln_conf)


            #It is possible for both approval code and 20 digit code to be on same line, that 
            #is why we don't use if else. Check if regex matches and that we have not found one yet.
            if ln_twenty and twenty_code == '':
                #Get first match
                
                extracted_20 = ln_twenty[0]
                #if ln_twenty2:
                #    extracted_20 = ln_twenty2[0]
                match_info = extractOne(extracted_20, twenty_digit_codes, scorer=ratio, score_cutoff=80,processor=default_process)    
                if match_info:
                    ln_match = match_info[0]
                    ln_match_conf = match_info[1]
                    #If line confidence exceeds threshold, and format close to table format, then get raw Textract extraction
                    if (ln_conf > confidence_threshold) and ln_match_conf > confidence_threshold:
                        twenty_code = extracted_20
                        twenty_max_conf = ln_conf
                        print('Raw output used for inv. code')
                        print(twenty_code)
                        print(twenty_max_conf)
                        print(ln_text)
                        print(ln_conf)
                    else:# ln_match_conf > confidence_threshold: #If  confidence lower, use table 
                        twenty_code = ln_match
                        twenty_max_conf = match_info[1]*(ln_conf/100)
                        print('Table used for inv. code')
                        print(twenty_code)
                        print(twenty_max_conf)
                        print(match_info[1])
                        print(ln_text)
                        print(ln_conf)

    #Take min of the two conf. levels as the confidence overall that way
    #code only above cutoff if both parts are above cutoff
    if (twenty_max_conf > 0) and (approval_max_conf > 0):
        confidence_reg_id = min(twenty_max_conf, approval_max_conf)
    else: #take the average so that if both are zero then it is 0, we use 0 to determine whenn no codes found (e.g. CILT)
        confidence_reg_id = (twenty_max_conf + approval_max_conf)/2
        
    reg_id_match = ' '.join((approval_code,twenty_code))

    return reg_id_match, confidence_reg_id

def detect_text(bucket, key):
    response = textract.detect_document_text(
        Document={
            'S3Object': {
                'Bucket': bucket,
                'Name': key
            }
        }
    )
    return response

def euclidean_distance(point1, point2):
    # Calculate the squared differences of coordinates
    squared_diffs = [(x - y) ** 2 for x, y in zip(point1, point2)]
    
    # Sum the squared differences and take the square root
    distance = math.sqrt(sum(squared_diffs))
    
    return distance



def get_next_line(query_string, response_json):
    '''To obtain the next line in the text after the query string, we search
    through lines to find lower left x value of the query box. If distance
    between lower left of `query_string` and upper left of following line is
    close, we return it.
    Outputs a string consisting of the next line.
    '''
    query_x = 0
    upper_left = 0
    next_line = None
    
    #Check that query_string not null
    if query_string and ('Blocks' in response_json):
        for item in response_json['Blocks']:
            if item['BlockType'] == 'LINE':
                #Search through lines to find line corresponding
                # to query string, record lower_left, then continue until
                # line found with upper_left approximately equal to it.
                #query_x == 0 indicates it was not found yet
                if query_x != 0:
                    #check if upper left of line equals lower_left of query
                    line_x = item['Geometry']['Polygon'][0]['X']
                    line_y = item['Geometry']['Polygon'][0]['Y']
                    distance = euclidean_distance((query_x,query_y),(line_x,line_y))
                    if distance <= 0.008:
                        next_line = item['Text']
                        break #exit loop and return
                        
                #Check if we found line corresponding to input string
                #Sometimes query result can miss characters such as umlaut
                #May assume query_result is non-empty but may have dropped
                #a character from the actual line_string
                line_string = item['Text']
                
                #check if query chars are substring of line string
                diff = len(line_string) - len(query_string)
                if 0 <= diff <= 1 and all(x in line_string for x in query_string):
                    #get lower left
                    query_x = item['Geometry']['Polygon'][3]['X']
                    query_y = item['Geometry']['Polygon'][3]['Y']
                
    return next_line

def get_city(address_line2_string):
    city = None
    if address_line2_string:
        pattern = r"\b[A-Z]{2}\b"
        no_state = re.sub(pattern, '', address_line2_string)
        pattern = "[0-9]{5}(?:-[0-9]{4})?"
        no_zip_no_state = re.sub(pattern, '', no_state) 
        city = no_zip_no_state.rstrip()
        city = city.rstrip(',')
        city = city.rstrip('.')
    return city

def get_state(address_line2_string):
    state_string = None
    if address_line2_string:
        pattern = r"\b[A-Z]{2}\b"
        result = re.findall(pattern, address_line2_string)
        if result:
            state_string = result[-1] #get last in case two letter city
    return state_string

def get_zip(address_line2_string):
    zip_string = None
    if address_line2_string:
        # Extract ZIP code from string
        #in case of bad characters, etc.
        pattern = "[0-9]{5}(?:-[0-9]{4})?"
        formatted_zip = re.findall(pattern, address_line2_string) # returns only zip
        if formatted_zip:
            zip_string = formatted_zip[0]
    return zip_string

def get_query_results(ref_id, response_json):
    '''Given id and response, search for 
    query results for that id and return results'''
    
    for b in response_json['Blocks']:
        if b['BlockType'] == 'QUERY_RESULT' and b['Id'] == ref_id:
            return {
                        'value': b.get('Text'), 
                        'confidence': b.get('Confidence'), 
                        'block': b
                    }
    return None

def parse_response_to_json(response_json):
    '''Update query_output dictionary above with response information
    in format required for Condition to check it
    Input: response JSON from textract API
    '''
    #if response not null
    if response_json:
        #for each query/alias parse results from response 
        for b in response_json['Blocks']:
            if b['BlockType'] == 'QUERY' and 'Relationships' in b:
                ref_id = b['Relationships'][0]['Ids'][0] #record id
                results = get_query_results(ref_id, response_json)
                q_alias = b['Query']['Alias'] #record alias
                query_output[q_alias] = results
                    
    return query_output
    
    
def save_dict_to_s3(dict_obj, bucket_name, file_name):
    '''Saves dict_obj json key value pairs obtained from the queries,
    to the target S3 bucket bucket_name under file_name.
    input1: python dictionary object
    input2: bucket name in form of string
    input3: filename in form of string
    '''
    try:
        s3.Object(bucket_name, file_name).put(Body=json.dumps(dict_obj))
        status = f'{file_name} saved to s3://{bucket_name}'
    except Exception as e:
        status = f'Failed to save: {e}'
            
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
        r,s = [],[]
        for c in self._conditions:
            # Matching field_name or field_name_regex
            condition_setting = c.get("condition_setting")
            if c["field_name"] == field_name \
                    or (c.get("field_name") is None and c.get("field_name_regex") is not None and re.search(c.get("field_name_regex"), field_name)):
                field_value, block = None, None
                if obj is not None:
                    field_value = obj.get("value")
                    block = obj.get("block")
                    confidence = obj.get("confidence")
                
                if c["condition_type"] == "Required" \
                        and (obj is None or field_value is None or len(str(field_value)) == 0):
                    r.append({
                                "message": f"The required field [{field_name}] is missing.",
                                "field_name": field_name,
                                "field_value": field_value,
                                "condition_type": str(c["condition_type"]),
                                "condition_setting": condition_setting,
                                "condition_category":c["condition_category"],
                                "block": block
                            })
                elif c["condition_type"] == "ConfidenceThreshold" \
                    and obj is not None \
                    and c["condition_setting"] is not None and float(confidence) < float(c["condition_setting"]):
                    r.append({
                                "message": f"The field [{field_name}] confidence score {confidence} is LOWER than the threshold {c['condition_setting']}",
                                "field_name": field_name,
                                "field_value": field_value,
                                "condition_type": str(c["condition_type"]),
                                "condition_setting": condition_setting,
                                "condition_category":c["condition_category"],
                                "block": block
                            })
                
                elif field_value is not None and c["condition_type"] == "ValueRegex" and condition_setting is not None \
                        and re.search(condition_setting, str(field_value)) is None:
                    r.append({
                                "message": f"{c['description']}",
                                "field_name": field_name,
                                "field_value": field_value,
                                "condition_type": str(c["condition_type"]),
                                "condition_setting": condition_setting,
                                "condition_category":c["condition_category"],
                                "block": block
                            })
                
                # field has condition defined and sastified
                s.append(
                    {
                        "message": f"The field [{field_name}] confidence score is {confidence}.",
                        "field_name": field_name,
                        "field_value": field_value,
                        "condition_type": str(c["condition_type"]),
                        "condition_setting": condition_setting,
                        "condition_category":c["condition_category"],
                        "block": block
                    })
                
        return r, s
        
    def check_all(self):
        if self._data is None or self._conditions is None:
            return None
        #if rule missed, this list holds list
        broken_conditions = []
        rules_checked = []
        broken_conditions_with_all_fields_displayed = []
        #iterate through rules_data dictionary
        #key is a field, value/obj is a
        #dictionary, for instance name_rule_data
        #is the dictionary for ADDRESSEE
        for key, obj in self._data.items():
            value = None
            if obj is not None:
                value = obj.get("value")

            if value is not None and type(value)==str:
                value = value.replace(' ','')
            #Check if this field passed rules:
            #if so, r and s are both []
            #if not, r is list of one or more of the dictionaries
            #seen in the check function
            r, s = self.check(key, obj)
            #If field missed rule
            if r and len(r) > 0:
                #append to bc list
                broken_conditions += r
            #If rule checked for field
            if s and len(s) > 0:
                rules_checked += s
            #If field missed or not
            #append to b_c_w_a_f_d
            if s and len(s) > 0:
                if r and len(r) > 0:
                    #If rule missed on this field, record it
                    broken_conditions_with_all_fields_displayed += r
                else:
                    #If no rule missed, record field 
                    broken_conditions_with_all_fields_displayed += s

        # apply index
        idx = 0
        #iterate through dictionaries
        for r in broken_conditions_with_all_fields_displayed:
            idx += 1
            r["index"] = idx
        #If at least one rule missed, display with it all fields
        #otherwise broken c
        if broken_conditions:
            broken_conditions = broken_conditions_with_all_fields_displayed
        return broken_conditions, rules_checked
# --------------- Dictionaries for data storage ------------------
query_output = {
          'ADDRESSEE': None,
          'STREET_ADDRESS': None,
          'CITY': None,
          'STATE': None,
          'ZIP_CODE_4': None
        }
#Centene fields format for saving to S3 
centene_format ={
    'RICOH_DCN': None,
    'REGULATORY_APPROVAL_ID': None,
    'ADDRESSEE': None,
    'ADDRESS_LINE_1': None,
    'ADDRESS_LINE_2': None,
    'CITY':None,
    'STATE': None,
    'ZIP_CODE_4': None,
    'BATCH': None,
    'BATCH_PREFIX': None
}
# --------------- Main handler ------------------
def lambda_handler(event, context):
    '''Demonstrates S3 trigger that uses
    textract APIs to detect text, query text in S3 Object.
    '''
    print('Received event: ' + json.dumps(event, indent=2))

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    print(key)
    #------------Image processing on TIF ----------------
    ## save TIF
    local_file_name = '/tmp/{}'.format(os.path.basename(key))
    #s3.download_file(Bucket=bucket, Key=key, Filename=local_file_name)
    #inFile = open(local_file_name, "r")
    print(f'local_file_name is: {local_file_name}')
    
    #Download file from S3 to local /tmp directory
    try:
        s3.Object(bucket, key).download_file(local_file_name)
        print(f"File downloaded successfully from S3.")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")

    split_into_pages_top_and_bottom(local_file_name)
    
    #Split into pages
    #split_tif(local_file_name)
    
    #directory = '/tmp'  # Replace with your directory path
    #print('currently in tmp')
    #for filename in os.listdir(directory):
    #    print(filename)
    
    # Insert 'file' key pair 
    # Split the string by '/'
    parts = key.split('/')

    # Get the text after the final '/'
    ricoh_dcn = parts[-1]

    centene_format['RICOH_DCN'] = parts[-1][:-4] #strip .T
    centene_format['BATCH'] = parts[-2]
    centene_format['BATCH_PREFIX'] = parts[-3]
    
     #Save to S3 in correct place in output bucket
    
    
    #Get Regid
    
    bottom_list = []
    top_list = []
    page_list = []
    reg_id_conf = 0
    files = os.listdir('/tmp')
    for file in files:
        #print('Here is one file:')
        #print(file)
        if '.tif' in file:
            print('Save to output bucket this:')
            s3_filename = f'{key.rstrip(parts[-1])}{file}'
            
            #s3_filename = f'{parts[-4]}/{parts[-3]}/{parts[-2]}/{file}'
            print(f's3_filename is {s3_filename}')
            # Read the local file as bytes
            #with open(local_file_name, 'rb') as f:
            #    file_bytes = f.read()
            ##s3.upload_fileobj(
            ##    fileobj=file_bytes,
            #    bucket=output_bucket,
            #    key=s3_filename
            #    )
            #s3.Object(output_bucket, s3_filename).put(Body=file_bytes)
            #s3.put_object(
            #    Body=file_bytes,
            #    Bucket=output_bucket,
            #    Key=s3_filename
            #    )
            client.upload_file('/tmp/' + file, output_bucket, s3_filename)
            if 'bottom' in file:
                bottom_list.append(s3_filename)
            if 'top' in file:
                top_list.append(s3_filename)
    #iterate over pages bottom  
    bottom_response = []
    #set regid file to be cilt if None, if found update to be that
    reg_id_file = None
    bottom_list = sorted(bottom_list, key=lambda x: x[-13:])
    print(f'Bottom list is: {bottom_list}')
    
    for file in bottom_list:
    # Call the analyze_document API
    
        print(file)
        
        try:
            # Call the analyze_document API
            print(f'calling Textract on {file}')
            bottom_response = textract.detect_document_text(
                Document={'S3Object': {'Bucket': output_bucket, 'Name':file}})
                    
        except Exception as e:
            print(e)
                
        if bottom_response:
            print(f'Textract called on {file} sucessfully.')
            #Use document knowledge that RegID at bottom and certain format to grab it 
            reg_id, reg_id_conf = get_reg_id(bottom_response)
            if reg_id_conf > 0:
                print(f'RegID code found in {file}.')
                print(reg_id, reg_id_conf)
                #save location
                reg_id_file = file
                
                centene_format['REGULATORY_APPROVAL_ID'] = reg_id
                break
    
    #In case no name or no regid found, make sure regid blank         
    if not reg_id_file:
        centene_format['REGULATORY_APPROVAL_ID'] = ''
    
    query_response = [] 
    #Define to be cilt if none
    name_file = None
    name_conf = 0
    top_list = sorted(top_list, key=lambda x: x[-13:])
    print(f'Top list is: {top_list}')
    
    for file in top_list:
        print(f'Top file is : {file}')
        try:
            # Calls textract detect_document_text API to detect text in the document S3 object
            #text_response = detect_text(bucket, key)
            print(f'Calling Textract on {file}')
            # Calls textract analyze_document API to query S3 object
            query_response = textract.analyze_document(Document={'S3Object': {'Bucket': output_bucket, 'Name': file}},
            FeatureTypes=['QUERIES'],
            QueriesConfig={'Queries': queries}
            )
        except Exception as e:
            print(e)
            print('Error processing object {} from bucket {}. '.format(key, bucket) +
              'Make sure your object and bucket exist and your bucket is in the same region as this function.')
            #Save error to file
            #status = save_dict_to_s3(centene_format, output_bucket,output_file_name)
            #print(status)
            raise e
        if query_response:
            
            # Store json of parsed response of first page
            query_data = parse_response_to_json(query_response)
        
            #Check if query name available
            if query_data['ADDRESSEE']:
                print(f'Name found on {file}')
                name_file = file
                name = query_data['ADDRESSEE']['value']
                name_conf = query_data['ADDRESSEE']['confidence']
                
                #Function get_next_line searches for name in lines, if not found, returns null
                #If ad1 returns null, then query name wrong or does not exist in document lines
                ad1 = get_next_line(name,query_response)
                
                if ad1: #If ad1 not null, query name is correct, so use it to get fields
                    ad2 = get_next_line(ad1, query_response)
                    ad3 = get_next_line(ad2, query_response)
                    
                
                    if not ad3: #If no 3rd address line 
                        centene_format['ADDRESSEE'] = name
                        centene_format['ADDRESS_LINE_1'] = ad1
                        centene_format['ADDRESS_LINE_2'] = ''
                        centene_format['CITY'] = get_city(ad2)
                        centene_format['STATE'] = get_state(ad2)
                        centene_format['ZIP_CODE_4'] = get_zip(ad2)
                    else: #If there is a 3rd address line
                        centene_format['ADDRESSEE'] = name
                        centene_format['ADDRESS_LINE_1'] = ad1
                        centene_format['ADDRESS_LINE_2'] = ad2
                        centene_format['CITY'] = get_city(ad3)
                        centene_format['STATE'] = get_state(ad3)
                        centene_format['ZIP_CODE_4'] = get_zip(ad3)
                        #Barcode document fix
                        #the code of 1's and i's below address gets read
                        #when this happens, state is None as it tries
                        #to pull two capital letters from a numeric code
                        #since there is no 3rd address line
                        if not centene_format['STATE']:
                            #then use no 3rd address line code
                            centene_format['ADDRESSEE'] = name
                            centene_format['ADDRESS_LINE_1'] = ad1
                            centene_format['ADDRESS_LINE_2'] = ''
                            centene_format['CITY'] = get_city(ad2)
                            centene_format['STATE'] = get_state(ad2)
                            centene_format['ZIP_CODE_4'] = get_zip(ad2)
                            
                            
                        
                else: #Query name is wrong or doesn't exist, pass to query defaults
                    street_address = ''
                    city = ''
                    state = ''
                    zip_code = ''
                    
                    #If queries found fields use them
                    if query_data['STREET_ADDRESS']:
                        street_address = query_data['STREET_ADDRESS']['value']
                    if query_data['CITY']:
                        city = query_data['CITY']['value']
                    if query_data['STATE']:
                        state = query_data['STATE']['value']
                    if query_data['ZIP_CODE_4']:
                        zip_code = query_data['ZIP_CODE_4']['value']
                        
                    #Now input the values we have    
                    centene_format['ADDRESSEE'] = name    
                    centene_format['ADDRESS_LINE_1'] = street_address
                    centene_format['ADDRESS_LINE_2'] = ''
                    centene_format['CITY'] = city
                    centene_format['STATE'] = state
                    centene_format['ZIP_CODE_4'] = zip_code
                    
                break #break out of loop
    
    
    #Finally, update the query_data with new confidence values
    # and values based on what we found
    #query_data
    #TEMPORARY: make fields derived from name have same confidence
    #We may want to pass in the confidence on the line (e.g. on City line, state line
    # etc. instead
    #OPTIONAL: Make confidence minimum of all fields, that way all fields
    #show in the A2I when one field is below cutoff level
    
    
    name_rule_data = {'value': centene_format['ADDRESSEE'],
            'confidence': name_conf,
            'block': None
        }
    ad1_rule_data = {'value': centene_format['ADDRESS_LINE_1'],
        'confidence': name_conf,
        'block': None
        }
    ad2_rule_data = {'value': centene_format['ADDRESS_LINE_2'],
        'confidence': name_conf,
        'block': None
        }
    city_rule_data = {'value': centene_format['CITY'],
        'confidence': name_conf,
        'block': None
        }
    state_rule_data = {'value': centene_format['STATE'],
        'confidence': name_conf,
        'block': None
        }
    zip_rule_data = {'value': centene_format['ZIP_CODE_4'],
        'confidence': name_conf,
        'block': None
        }
    #reg_id moved to bottom to match where it is found on page
    #labelers wanted the webpage they see to display regid as the last field
    reg_id_rule_data = {'value': centene_format['REGULATORY_APPROVAL_ID'],
        'confidence': reg_id_conf,
        'block': None
        }
    rules_data = {'ADDRESSEE': name_rule_data,
        'ADDRESS_LINE_1': ad1_rule_data,
        'ADDRESS_LINE_2': ad2_rule_data,
        'CITY': city_rule_data,
        'STATE' : state_rule_data,
        'ZIP_CODE_4': zip_rule_data,
        'REGULATORY_APPROVAL_ID': reg_id_rule_data,
        }

    #Validate business rules
    con = Condition(rules_data,rules)
    rule_missed, rule_checked = con.check_all()
    rule_missed_string = ', '.join(element['message'] for element in rule_missed)
    centene_format['RULE_MISSED_COUNT'] = len(rule_missed)
    centene_format['RULE_MISSED_STRING'] = rule_missed_string
    
    #Save filenames 
    output_file_name = f'{key[:-4]}.json'
    rm_file_name = f'{key[:-4]}-rule-missed.json'
    rs_file_name = f'{key[:-4]}-rule-checked.json'
    
    #Save data to s3 and return if save was successful or not
    status = save_dict_to_s3(centene_format, output_bucket,output_file_name)
    #Save rule missed list
    s3.Object(output_bucket, rm_file_name).put(Body=json.dumps(rule_missed))
    #Save rule satisfied list
    s3.Object(output_bucket, rs_file_name).put(Body=json.dumps(rule_checked))
    
    #-------Image for A2I--------------------------------------
    s3_filename_a2i = f'{key[:-4]}-image-for-A2I.png'
    
    #Combine the partial pages where the fields were found
    if name_file and reg_id_file:
        name_file_parts = name_file.split('/')
        reg_id_file_parts = reg_id_file.split('/')
        tmp_name_file = '/tmp/'+ name_file_parts[-1]
        tmp_regid_file = '/tmp/' + reg_id_file_parts[-1]
        #Recombine creates an image in /tmp called image-for-A2I.png
        #using the top and bottom images where names and resp. codes
        #were found.
        recombine(tmp_name_file,tmp_regid_file)
        
        #We upload that to the correct bucket and prefix
        print(f's3_filename is {s3_filename_a2i}')
        local_a2i_image_path = '/tmp/image-for-A2I.png'
        client.upload_file(local_a2i_image_path, output_bucket, s3_filename_a2i)

    
    else: #CILT
        #upload cilt image to correct bucket with prefix
        
        local_a2i_image_path = '/tmp/image-for-A2I.png'
        #upload cilt image
        client.upload_file('/tmp/cilt.png', output_bucket, s3_filename_a2i)
        
        
    
    #-----------------Create A2I labeling job--------------------------

    
    #-----------------Clean up------------------------------------------
    # Get disk usage of /tmp directory
    usage = os.statvfs('/tmp')
    total_space = usage.f_frsize * usage.f_blocks
    used_space = usage.f_frsize * (usage.f_blocks - usage.f_bfree)
    available_space = usage.f_frsize * usage.f_bavail

    # Convert to human-readable format
    total_space_gb = total_space / (1024 ** 3)
    used_space_gb = used_space / (1024 ** 3)
    available_space_gb = available_space / (1024 ** 3)

    # Print the disk usage
    print(f"Total Space: {total_space_gb:.2f} GB")
    print(f"Used Space: {used_space_gb:.2f} GB")
    print(f"Available Space: {available_space_gb:.2f} GB")
    
    
    
    # Iterate over all files in the directory
    for filename in os.listdir('/tmp'): 
        #if filename.endswith('.tif'):  # Check if the file ends with ".tif"
        file_path = os.path.join('/tmp', filename)  # Get the full file path
        os.remove(file_path)  # Remove the file
        print(f"Removed file: {filename}")

    
    return status, centene_format, rule_missed, rule_checked