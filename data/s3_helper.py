"""
Notes:
    This module facilitates loading files from and saving files to s3 buckets 

example:

s3_to_local(s3_uri="s3://njtpa.auraison.aegean.ai/njtpa-year-2/labels_ground_truth/year-2/output/tmp/",
                desired_formats=['txt'],
                target_dir='cwd',
                verbose=True)

local_to_s3(local_folder='cwd',
                 file_formats=['txt'],
                 s3_uri="s3://njtpa.auraison.aegean.ai/njtpa-year-2/labels_ground_truth/year-2/output/tmp/",
                 verbose=True)

"""

import os
import boto3
import configparser
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv(verbose=True)

def __getCredentials(aws_sso_flag: bool):

    configParser = configparser.RawConfigParser()
    # The default location of credentials in container
    configFilePath = r'/workspaces/sidewalk-detection/.aws/credentials'                
    
    if os.path.isfile(configFilePath):
        print('Aws credentials found')
    else:
        print(f'Aws credential file not found. Searched in {configFilePath}\n')
        return None,None,None 
        
    configParser.read(configFilePath)                      
    configParser.sections()              

    key_id = configParser.get('default','aws_access_key_id',raw=False) 
    access_key = configParser.get('default','aws_secret_access_key',raw=False)
    if aws_sso_flag:
        token = configParser.get('default', 'aws_session_token',raw = False)
    else:
        token = None
    return key_id, access_key, token


def __getS3bucket(bucket_name:str, aws_sso_flag: bool):
    
    # key_id,access_key, token = __getCredentials(aws_sso_flag=aws_sso_flag)

    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    token=os.getenv('TOKEN')                            
    #Create Session: 
    
    if aws_sso_flag:
        s3 = boto3.resource('s3',
            endpoint_url='http://s3-2.nj.aegean.ai:9000',
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1',
            aws_session_token = token
        )
    else: 
        s3 = boto3.resource('s3',
            endpoint_url='http://s3-2.nj.aegean.ai:9000',
            aws_access_key_id = aws_access_key_id,
            aws_secret_access_key = aws_secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
    
    bucket = s3.Bucket(bucket_name)
   
    return bucket

def __parse_s3uri(s3_uri:str)->str:
    
    bucket_name, prefix = s3_uri.split('s3://',1)[1].split('/',1)         #URI format: 's3://{bucket_name}/{prefix}
    
    return bucket_name,prefix


def __listdir(directory:str,extensions:list,verbose:bool)->list:                #list files with specified extensions (filter for tif/png/jpeg etc
    
    
    if directory == 'cwd':
        directory = os.getcwd()
        
    Files = os.listdir(directory)                                               #list all files
    files = []
    
    print(Files,'\n') if verbose else None
    
    for file in Files:
        if '.' not in file:
            continue
        elif (file.lower().rsplit('.',1)[1] in extensions) or ('all' in extensions):      #find extension and check membership in requested 
            files.append(file)
    
    print(f'{len(files)} found with {extensions} in {directory}\n') if verbose else None
    
    return files                                                                #return file names that match requested extensions


def s3_to_local(s3_uri:str, desired_formats:list, target_dir:str, aws_sso_flag: bool, verbose:bool)->None:      #this function is called in loadFiles()
    """
    Notes:
        Use this method to download files from an s3 bucket to a local folder using an s3 URI
        S3 > Local
    
    Inputs:
        s3_uri (str): s3URI taken from aws bucket
        desired_formats (list): list of file formats to filter by
        target_dir (str): folder to download s3 contents to
        verbose (bool): bool for summary or debugging information     
    
    Outputs:
        None: this is a method that results in files of desired format being uploaded to s3
    
    """
    bucket_name, s3_directory = __parse_s3uri(s3_uri)
    
    s3_bucket = __getS3bucket(bucket_name=bucket_name, aws_sso_flag=aws_sso_flag)
    
    if target_dir == 'cwd':
        target_dir= os.getcwd()
    
    # Iterates through files  in provided S3 bucket prefix 
    for s3_object in s3_bucket.objects.filter(Prefix=s3_directory):                             
        
        print('{0}:{1}'.format(s3_bucket.name, s3_object.key))
        
        #Separate out filename from object name (string)
        path, filename = os.path.split(s3_object.key)                                           

        # Finds file extension 
        if '.' in filename:                                                                     
            image_format = filename.rsplit('.',1)[1].lower()
        else:
            image_format = 'no format'
            
        # Download the files with desired file formats
        if (image_format in desired_formats) or ('all' in desired_formats):                            
            print(f'path: {path}  |  file: {filename}  | type: {type(filename)}') if verbose else None 
             #create local file path
            Filepath = fr'{target_dir}/{filename}'                                                      
            try:
                 #downloads S3object into (local) folder w/ Filepath
                s3_bucket.download_file(s3_object.key, Filepath)                                     
                print(f'file: {filename} created successfully') if verbose else None                 
            except:
                print(f'file: {filename} could not load successfully') if verbose else None
        
    print('download to local complete\n') if verbose else None    


def local_to_s3(local_folder:str,file_formats:list,s3_uri:str,verbose:bool)->None:
    """
    Notes:
        Use this method to upload files from a local folder to an s3 bucket using an s3 URI 
        local > S3
    
    Inputs:
        s3_uri (str): s3URI taken from aws bucket
        desired_formats (list): list of file formats to filter by
        verbose (bool): bool for summary or debugging information     
    
    Outputs:
        None: this is a method that results in files of desired format being uploaded to s3
    
    """    
    bucket_name, s3_directory = __parse_s3uri(s3_uri)
    s3_bucket = __getS3bucket(bucket_name=bucket_name)

    file_list = __listdir(directory=local_folder,extensions=file_formats,verbose=verbose)
    
    for n, file_name in enumerate(file_list):
        s3_filename = fr'{s3_directory}{file_name}'
        print(f'uploading {file_name} to {s3_filename}') if verbose else None
        
        x = s3_bucket.upload_file(Filename=file_name,Key=s3_filename)
        print('returned: ',x)
        
        
    print(f'{n+1} file(s) uploaded to {s3_uri}\n') if verbose else None
    


