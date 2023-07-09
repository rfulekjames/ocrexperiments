
import boto3
import os

def sample_across_batches(bucket_name, folder_path, num_files=2, local_folder='sample', s3_client=boto3.client('s3')):
    # Retrieve the list of subfolders in the specified folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_path, Delimiter='/')
    subfolders = [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

    # Sample uniformly at random from the subfolders to download files from
    selected_subfolders = random.choices(subfolders, k=num_files)

    # Create the local target folder if it does not exist
    if not os.path.isdir(local_folder):
        os.mkdir(local_folder)
        
    sampled_files = set()

    # Download the files
    for subfolder in selected_subfolders:
        # Retrieve the list of objects (files) in the subfolder
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=subfolder)
        files = [file['Key'] for file in response.get('Contents', [])]
        files = [file for file in files if file not in sampled_files]
        if len(files) < 2:
            break
        # Sample a file uniformly at random without repetition
        file_to_download = random.choice(files[1:])
        sampled_files.add(file_to_download)

        # Download the selected file
        local_file_name = '-'.join(file_to_download.split('/')[-2:])
        s3_client.download_file(bucket_name, file_to_download, os.path.join(local_folder, local_file_name))
        print(f'sampled {file_to_download}')
         
import random

sample_across_batches('centenetransfer', 'Centenetesting/', num_files=200)