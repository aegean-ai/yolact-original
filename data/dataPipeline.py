import os
import traceback
import sys
import json
import boto3
import configparser
from split_merge_tiff import Open_Tiff,Split_Tiff,Save_Tiff
from pycocotools import mask
import numpy as np
import cv2

config = {
    'pipeline':[],
    'verbose': 1,
    'ds':None,
    'pef': 0.1,
    'score_threshold':0.0,
    'det_score_threshold':0.0,
    'fmts':'jpg,png',
    'sfmt':'jpeg',
    'rowsSplitPerTile':20,
    'colsSplitPerTile':20,
    'patchDimX':256,
    'patchDimY':256,
    'tileDimX':5000,
    'tileDimY':5000
}

def vbPrint(s):
    '''
    prints only if verbose is configured to 1
    '''
    if(config['verbose']==1):
        print(s)

def setConfig(argv):
    '''
    Sets the configuration for the pipeline if format is --<KEY>=<VALUE>
    Adds steps to the pipelien if format is -<PipelineStep>
    '''
    argType = { # Default is str
        'verbose': int,
        'pef' : float,
        'score_threshold': float,
        'det_score_threshold':float,
        'rowsSplitPerTile':int,
        'colsSplitPerTile':int,
        'patchDimX':int,
        'patchDimY':int
    }
    
    global config
    argv = argv[1:]
    if('--verbose=0' in argv):
        argv = ['--verbose=0']+argv

    for arg in argv:
        try:
            if(arg[:2] == '--'):
                x = arg[2:].split('=')
                if(x[0] in argType):
                    config[x[0]] = argType[x[0]](x[1])
                else:
                    config[x[0]] = str(x[1])
            else:
                assert arg[0] == '-'
                config['pipeline'].append(arg[1:])
        except:
            traceback.print_exc()
            vbPrint('Bad Argument: %s'%(arg))

def __getCredentials():
    configParser = configparser.RawConfigParser()           
    configFilePath = r'/home/ubuntu/.aws/credentials'      
          
    configParser.read(configFilePath)                      
    configParser.sections()              

    key_id = configParser.get('default','aws_access_key_id',raw=False) 
    access_key = configParser.get('default','aws_secret_access_key',raw=False) 
    token = configParser.get('default', 'aws_session_token',raw = False)
    return key_id, access_key, token

def __getS3bucket(bucketname):
    
    key_id,access_key,token = __getCredentials()

    #Create Session: 
    session = boto3.Session(aws_access_key_id = key_id,
                            aws_secret_access_key = access_key,
                            aws_session_token = token)

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucketname)
    
    return bucket


############## PRE TRAINING DATA PIPELINE FUNCTIONS #############

def loadTiles():
    '''
    loads tiles data from s3 into the directory:
        - ./data/<datset>/tiles/
        
    bn (str): bucket name
    s3Dir (str): bucket path to files
    file_formats (str): list of file formats to search for in s3Dir, gets converted to list. Ex: "jpg, png" -> ['jpg','png']
    
    '''

    ds = config['ds']
    ts = config['ts']
    bn = config['bn']                                                           #bucket name
    s3Dir = config['s3td']                                                      #image path in bucket
    file_formats = config['fmts'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    loadDir = '%s/tiles_%s'%(ds,ts)
    
    ## Making the dirs
    if(os.path.isdir(ds)):
        vbPrint('Found dir: %s'%(ds))
    else:
        vbPrint('Making dir: %s'%(ds))
        os.mkdir(ds)

    if(os.path.isdir(loadDir)):
        vbPrint('Found dir: %s'%(loadDir))
    else:
        vbPrint('Making dir: %s'%(loadDir))
        os.mkdir(loadDir)

    ## Loading the data
    bucket = __getS3bucket(bn)
    
    for s3_object in bucket.objects.filter(Prefix=s3Dir):
        
        path, filename = os.path.split(s3_object.key)

        if '.' in filename:
            image_format = filename.rsplit('.',1)[1].lower()
        else:
            image_format = 'no format'
            
        #Download the files
        if image_format in file_formats:
            vbPrint(f'path: {path}  |  file: {filename}  | type: {type(filename)}')
            File = fr'{loadDir}/{filename}'
            bucket.download_file(s3_object.key, File)
            vbPrint(f'file: {filename} written successfuly')
    
    vbPrint('Tiles loaded succesfully')
    '''
    #Example:
    python3 dataPipeline.py --ds=dataset2 --ts=plTest --fmts=jpg,png --bn=cv.datasets.aegean.ai --s3td=njtpa/njtpa-year-2/DOM2015/Selected_500/pipelineTest/ -loadTiles
    '''



def genImgPatches():
    '''
    Generates image patches by splitting the tiles
    Prerequisite: loadTiles() has been run or tiles are present in the data/<dataset>/tiles directory
    
    save_fmt (str): save format. Exs: jpeg, png
    
    '''
    ds = config['ds']
    ts = config['ts']
    save_fmt = config['sfmt']
    tilesDir = '%s/tiles_%s'%(ds,ts)
    tiles = os.listdir(tilesDir)
    vbPrint('%i Tile files found in %s'%(len(tiles),tilesDir))
    patchesDir = '%s/imagePatches_%s'%(ds,ts)
    
    ## Making the dirs
    if(os.path.isdir(patchesDir)):
        vbPrint('Found dir: %s'%(patchesDir))
    else:
        vbPrint('Making dir: %s'%(patchesDir))
        os.mkdir(patchesDir)
    
    ## Creation of Image patches
    
    for imageName in tiles:
        imgage_Path = fr'{tilesDir}/{imageName}'                                    #Generate Image Path 
        vbPrint(f'Current Image Path: {imgage_Path}')
        
        #Open File
        array,profile = Open_Tiff(path_to_tiff = imgage_Path,
                                  verbose = config['verbose'])
                
        #SplitTiff:
        split_array = Split_Tiff(image=array,
                                blockX = 256,
                                blockY = 256,
                                minScrapPercent = 0,
                                verbose = config['verbose'])
        
        
        #Save Image Patches:
        Save_Tiff(array=split_array,
                  save_directory=patchesDir,
                  image_index=imageName,
                  profile = profile,
                  save_fmt = save_fmt,
                  verbose = config['verbose'])
                  
       
    vbPrint('Image Patches made successfuly')    
    '''
    #Example:
    python3 dataPipeline.py --ds=dataset2 --ts=plTest  --sfmt=jpg -genImgPatches
    '''


'''
INVESTIGATION NEEDED FOR genLabelPatches AND genAnnotations:
    - How to get the labels from the source (Find the source)
    - How to make annotations from this
'''
def genLabelPatches():
    '''
    - Generates the label patches
    - For training only
    '''
    ds = config['ds']
    ts = config['ts']
    labelPatchesDir = '%s/labelPatches_%s'%(ds,ts)
    
    ## Making the dirs
    if(os.path.isdir(labelPatchesDir)):
        vbPrint('Found dir: %s'%(labelPatchesDir))
    else:
        vbPrint('Making dir: %s'%(labelPatchesDir))
        os.mkdir(labelPatchesDir)

    ## Creation of Label patches
    '''
    TODO [Similar to generation of Image patches]
    - Each tile has to be split into 400 patches
    - The patches have to be saved in the [labelPatchesDir] directory
    - The naming of the paches should be such that if we have the name of tile, we can generate the name of its patches
      We could follow the format that has been used for year-1, it has been documented in the 'Formation of image patches'
      section at docs.upabove.app/sidewalk/data-pipeline/_index.md. The names should map to their respective image patches
    - Investigate and implement how to handle the world files
    '''

    vbPrint('Label Patches made successfuly')

def genAnnotations():
    '''
    - Generates the train and test annotations files
    - For training only
    '''
    ds = config['ds']
    ts = config['ts']
    labelPatchesDir = '%s/annotations_%s'%(ds,ts)
    
    ## Making the dirs
    if(os.path.isdir(labelPatchesDir)):
        vbPrint('Found dir: %s'%(labelPatchesDir))
    else:
        vbPrint('Making dir: %s'%(labelPatchesDir))
        os.mkdir(labelPatchesDir)

    ## Creation of Train and Test annotation files
    '''
    TODO
    - Generate the train and test annotation files using the images and labels.
    - Check out the ML_Clip repo's czhUtils.py file. It seems they have made functions to generate the annotations.
    '''

    vbPrint('Annotations made successfuly')
    pass

############## POST TRAINING DATA PIPELINE FUNCTIONS #############
def genInferenceJSON():
    '''
    - Generates inferences and saves the sidewalk detections (Dets) in a JSON file
    - This works on the data mentioned in the selected configuration from config.py
    '''
    ds = config['ds']
    ts = config['ts']

    detPath = '%s/inferencesJSON_%s/'%(ds,ts)
    if(os.path.isdir(detPath)):
        vbPrint('Found dir: %s'%(detPath))
    else:
        vbPrint('Making dir: %s'%(detPath))
        os.mkdir(detPath)

    shCmd = 'cd .. && python eval.py --trained_model="%s" \
        --config=%s  --web_det_path="%s" \
        --score_threshold=%f --top_k=15  --output_web_json'%(
            config['trained_model'],
            config['config'],
            'data/'+detPath,
            config['score_threshold']
        )

    if(config['verbose']==0):
        shCmd += ' --no_bar'

    vbPrint('Initializing Inferences')
    os.system(shCmd)
    vbPrint('Completed')

def genInferenceTiles():
    '''
    Prerequisite: genInferenceJSON() has been run and the Dets JSON is available in ./<datset>/inferenceTiles_<tileset>/
    Make all inferences
    For each tile, pick and merge an inference for each of its patches
    '''
    ds = config['ds']
    ts = config['ts']
    inferenceTilesDir = '%s/inferenceTiles_%s'%(ds,ts)

    # Making the dirs
    if(os.path.isdir(inferenceTilesDir)):
        vbPrint('Found dir: %s'%(inferenceTilesDir))
    else:
        vbPrint('Making dir: %s'%(inferenceTilesDir))
        os.mkdir(inferenceTilesDir)
        
    detFilePath = '%s/inferencesJSON_%s/%s'%(ds,ts,config['infJSON'])
    #annFilePath = '%s/annotations_%s/%s'%(ds,ts,config['annJSON'])
    annFilePath = '%s/annotations/%s'%(ds,config['annJSON']) # For testing. Can be reverted once the data-pipeline part is completed.

    # Generates a HashMap that stores the image tiles information
    # The program iterates through this to generate each tile
    tileMap = dict()
    vbPrint('Loading the annotations JSON')
    with open(annFilePath) as f:
        inpImageData = json.load(f)
        for img in inpImageData['images']:
            tile,i,j =  img['file_name'].split('.')[0].split('_')
            if(tile not in tileMap):
                tileMap[tile] = []
            tileMap[tile].append((img['id'],int(i),int(j)))
        del inpImageData
    
    vbPrint('Generated information for %i tiles'%(len(tileMap.keys())))

    vbPrint('Loading the inference JSON')
    with open(detFilePath) as f:
        infData = json.load(f)['images']
    
    
    patchShape = (config['patchDimX'],config['patchDimY'])
    infTileShape = (patchShape[0]*config['colsSplitPerTile'],
                    patchShape[1]*config['rowsSplitPerTile'])

    vbPrint('Expected Patch dimensions: %ix%i'%(patchShape[0],patchShape[1]))
    vbPrint('Expected Tiles dimensions: %ix%i'%(infTileShape[0],infTileShape[1]))

    # Creating a hashmap for the image dets with the image ids as keys
    # Each prediction is a merged image of many detections
    # Only detections above the threshold are selected to be merged
    vbPrint('Generating det masks with a threshold of %0.2f'%(config['det_score_threshold']))
    infPatchMap = dict()
    i=0
    n=len(infData)
    peFactor = config['pef']
    
    for img in infData:
        i += 1
        pred = np.zeros(patchShape)
        for det in img['dets']:
            if(det['score']>config['det_score_threshold']):
                detMask = mask.decode(det['mask'])
                pred[detMask==1]=255
        
        infPatchMap[img['image_id']] = pred

        # Prints progress at every 'pef' factor of completion
        # PEF stands for 'Print-Every factor'
        facComp = (i+1)/(n)
        if(facComp >= peFactor):
            vbPrint('Generating detection masks: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
            peFactor += config['pef']

    del infData
    
    vbPrint('Detection Masks made')
    vbPrint('Generating inference tiles from detection masks')

    # Rows and columns a tile is split into.
    # Default is 20,20
    rows = config['rowsSplitPerTile']
    cols = config['colsSplitPerTile']

    i = 0
    n = len(tileMap.keys())
    peFactor = config['pef']

    '''
    For each tile, gets the detection masks and merges them into one image.
    It seems that many patches are missing in inferences. It is assumed these don't have any detections and are skipped.
    '''
    for i,tile in enumerate(tileMap.keys()):
        # Skipping tile if unexpected number of patches exist for the tile in the annotations JSON
        expectedPatches = rows*cols
        if(len(tileMap[tile])!=expectedPatches):
            vbPrint('Found %i patches for tile %s. Skipping'%(len(tileMap[tile]), tile))
        else:
            # The inference tile. Initialized with 0. Masks are overlayed at their respective locations.
            infTile = np.zeros((rows*patchShape[0],cols*patchShape[1]),dtype=np.uint8)
            for patchID,row,col in tileMap[tile]:
                # Overlays the patch mask at the correct location in infTile
                # Skip any patches with no inference. It is assumed that no inference data is generated for patches with 0 detections.
                if(patchID in infPatchMap):
                    # Cropshift is the extra pixels to the left and top in the final row and column of the image inference
                    # This is only done due to the way the year-1 project was 'padded'.
                    # This will be need to be removed when correct padding is applied.
                    cropShiftX = 0
                    cropShiftY = 0
                    if(row==rows):
                        cropShiftX = 120
                    if(col==cols):
                        cropShiftY = 120
                    infTile[(row-1)*patchShape[0]:((row)*patchShape[0])-cropShiftX, (col-1)*patchShape[1]:((col)*patchShape[1])-cropShiftY] = infPatchMap[patchID][cropShiftX:,cropShiftY:]
            
            # Cropping image into the final tile dimension
            infTile = infTile[:config['tileDimX'],:config['tileDimY']]
            
            # Writing the image tile
            # File format kept as png because it has lossless compression and to work well with rasterio if needed.
            cv2.imwrite('%s/%s.png'%(inferenceTilesDir,tile),infTile)

            # Prints progress at every 'pef' factor of completion
            # PEF stands for 'Print-Every factor'
            facComp = (i+1)/(n)
            if(facComp >= peFactor):
                vbPrint('Generating inferences tile: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
                peFactor += config['pef']

    vbPrint('Inference Tiles generated Successfully')

if __name__ == '__main__':
    '''
    Argument Formats:
        --<KEY>=<VALUE>     -> Used to set config
        -<PipelineStep>     -> Used to set pipeline Steps

    Base Arguments:
        --ds                -> [dataset name] The datset name. This will be used to create the file directory and refer to the dataset.
        --ts                -> [tileset name] Name of the tileset. This will be used to create the unique tiles and images folder.
                                    Separate tile and image folders are needed to separate training data from inference data.
                                    This also allows speration of inference set. Ex of values for ts: train,test, XYZ_County_inference, PQR_Inference, etc.
        -<PipelineStep>     -> [Function Name] name of the function to call in the pipeline step. Multiple can be passed to create the pipeline.

    LoadTiles Arguments:
        --s3td              -> [s3 URL] The s3 URI of the input image tiles. Ex: s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/DOM2015/Selected_500/

    genInferenceJSON Arguments:
        --trained_model     -> [path] Path to the trained model. Ex: weights/DVRPCResNet50_8_88179_interrupt.pth
        --config            -> [config name] name of the config to be used from config.py. Example dvrpc_config
        --score_threshold   -> [float] The score threshold to use for inferences. Example: 0.0

    genInferenceTiles Arguments:
        --annJSON           -> [path] name of the annotations .JSON file. Example: DVRPC_test.json
        --infJSON           -> [path] name of the created inference .JSON file. Example: DVRPCResNet50.json

    Misc Arguments:
        --verbose           -> [0 or 1] Setting it to 1 allows print statements to run. Default: 1
        --pef               -> [float] (Used for long processes) Determines after how much increase in the factor of completion of a process, a vbPrint is made. 
                                    For example: 0.1 means at every 10% completion, a print is made. (Only works if --verbose is set to 1). Default: 0.1

    Examples:
        - Training Pipeline
            ```bash
            python dataPipeline.py --ds=ds1 --ts=train -loadTiles --s3td=test -genImgPatches -genLabelPatches -genAnnotations
            ```

        - MEVP Pipeline
            ```bash
            python dataPipeline.py --ds=ds1 --ts=inf1 -loadTiles --s3td=test -genImgPatches -genAnnotations
            python -genInferenceJSON --trained_model=weights/DVRPCResNet50_8_88179_interrupt.pth --config=dvrpc_config --score_threshold=0.0
            python dataPipeline.py --ds=ds1 --ts=inf1 -genInferenceTiles --annJSON=DVRPC_test.json --infJSON=DVRPCResNet50.json
            ```

    The complete Pipeline:
        ### Data Pipeline
        - loadTiles
        - genImgPatches
        - genLabelPatches
        - genAnnotations
        ### MEVP Pipeline
        - genInferenceJSON
        - genInferenceTiles
    '''
    setConfig(sys.argv)
    vbPrint('Configuration set:')
    for key in config:
        vbPrint('%s -> %s'%(key,config[key]))
    vbPrint('Running the Pipeline')
    for step in config['pipeline']:
        try:
            vbPrint('------------------')
            vbPrint('Performing step: %s'%(step))
            eval(step+'()')
        except Exception as e:
            traceback.print_exc()
            vbPrint('There was an error evaluating the step: %s'%(step))
            vbPrint('Terminating')
            exit()
    vbPrint('------------------')
    vbPrint('Pipeline Completed')