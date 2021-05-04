"""
Notes:
    This module facilitates loading files from and saving files to s3 buckets 


ex URI:
    s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/output/tmp/
    s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/output/tmp/test.txt

Bucket methods:
s3.Bucket(name='cv.datasets.aegean.ai')
['Acl', 'Cors', 'Lifecycle', 'LifecycleConfiguration', 'Logging', 'Notification', 'Object', 'Policy', 'RequestPayment', 'Tagging', 'Versioning', 'Website', '__class__', 
'__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', 
'__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_name', 'copy', 'create', 
'creation_date', 'delete', 'delete_objects', 'download_file', 'download_fileobj', 'get_available_subresources', 'load', 'meta', 'multipart_uploads', 'name', 'object_versions', 'objects', 
'put_object', 'upload_file', 'upload_fileobj', 'wait_until_exists', 'wait_until_not_exists']


"""


import os
import boto3
import configparser


def __getCredentials():
    configParser = configparser.RawConfigParser()
    configFilePath = r'/home/ubuntu/.aws/credentials'                           #default location on ec2 instance
    
    if os.path.isfile(configFilePath):
        print('Aws credentials found')
    else:
        print(f'Aws credential file not found. Searched in {configFilePath}\n')
        return None,None,None 
        
    configParser.read(configFilePath)                      
    configParser.sections()              

    key_id = configParser.get('default','aws_access_key_id',raw=False) 
    access_key = configParser.get('default','aws_secret_access_key',raw=False) 
    token = configParser.get('default', 'aws_session_token',raw = False)
    return key_id, access_key, token


def __getS3bucket(bucket_name:str):
    
    key_id,access_key,token = __getCredentials()

    #Create Session: 
    session = boto3.Session(aws_access_key_id = key_id,
                            aws_secret_access_key = access_key,
                            aws_session_token = token)

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    
    return bucket

def __parse_s3uri(s3_uri:str)->str:
    
    bucket_name, s3_directory = s3_uri.split('s3://',1)[1].split('/',1)         #URI format: 's3://{bucket_name}/{s3_directory}
    
    return bucket_name,s3_directory


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


def load_s3_to_local(s3_uri:str,desired_formats:list,load_dir:str,verbose:bool)->None:      #this function is called in loadFiles()
    """
    Notes:
        Use this method to download files from an s3 bucket to a local folder using an s3 URI
        S3 > Local
    
    Inputs:
                  s3_uri (str): s3URI taken from aws bucket
        desired_formats (list): list of file formats to filter by
                load_dir (str): folder to load s3 contents to
                verbose (bool): bool for summary or debugging information     
    
    Outputs:
        None: this is a method that results in files of desired format being uploaded to s3
    
    """
    bucket_name, s3_directory = __parse_s3uri(s3_uri)
    
    s3_bucket = __getS3bucket(bucket_name=bucket_name)
    
    if load_dir == 'cwd':
        load_dir= os.getcwd()
    
    for s3_object in s3_bucket.objects.filter(Prefix=s3_directory):                             #iterates through files in desired directory of S3 bucket
        
        path, filename = os.path.split(s3_object.key)                                           #Seperate out filename from object name (string)

        if '.' in filename:                                                                     #Finds file extension 
            image_format = filename.rsplit('.',1)[1].lower()
        else:
            image_format = 'no format'
            
                                                                                    ##Download the files
        if (image_format in desired_formats) or ('all' in desired_formats):                            #Retrieve files with desired file formats
            print(f'path: {path}  |  file: {filename}  | type: {type(filename)}') if verbose else None #print info about file
            Filepath = fr'{load_dir}/{filename}'                                                       #create local file path
            try:
                s3_bucket.download_file(s3_object.key, Filepath)                                      #downloads S3object into (local) folder w/ File
                print(f'file: {filename} created successfully') if verbose else None                  #print success message
            except:
                print(f'file: {filename} could not load successfully') if verbose else None
        
    print('download to local complete\n') if verbose else None    


def load_local_to_s3(local_folder:str,file_formats:list,s3_uri:str,verbose:bool)->None:
    """
    Notes:
        Use this method to upload files from a local folder to an s3 bucket using an s3 URI 
        local > S3
    
    Inputs:
                  s3_uri (str): s3URI taken from aws bucket
        desired_formats (list): list of file formats to filter by
                load_dir (str): folder to load s3 contents to
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
    


"""
example:

load_s3_to_local(s3_uri="s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/output/tmp/",
                desired_formats=['txt'],
                load_dir='cwd',
                verbose=True)

load_local_to_s3(local_folder='cwd',
                 file_formats=['txt'],
                 s3_uri="s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/output/tmp/",
                 verbose=True)

"""