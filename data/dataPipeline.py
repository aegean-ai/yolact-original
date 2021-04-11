"""
Important Definitions:
    > World-File/World-Image: This is any input image of arbitrary size that must be broken down into patches.
    > Tile-Image: This is any subdivision of a World-Image that is not a patch. The default is 5000x5000 but can be changed
    > Patch(s)/ Image-Patch: This is the smallest subdivision of a World-Image. This is hardcoded as 256x256
    
    > Input world-image: this is(are) the world files at the start of the pipeline
    > Label world-image: this is(are) the world files representing the sidewalks(labels) for training
    
    
Workflow:
    loadFiles (world inputs/labels) -> genImgPatches (inputs/labels) ->... 
                                            |__ Open_Raster > Split_Raster > Save_Patches

"""
############## Imports/Setup #############

import os
import traceback
import sys
import json
from s3_helper import load_s3_to_local
from split_merge_raster import Open_Raster,Split_Raster,Save_Patches            #These are all called w/in genImgPatches
import time                                                                     #used in decorator __time_this()

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
    'rowsSplitPerTile':20,
    'colsSplitPerTile':20,
    'patchDimX':256,
    'patchDimY':256,
    'tileDimX':5000,
    'tileDimY':5000,
    'mWH': '5000,5000',
    'fmts':'jpg,png,tif',
    'sfmt':'jpeg'
}

############## Helper Functions #############


def __listdir(directory:str,extensions:list)->list:                             #list files with specified extensions (filter for tif/png/jpeg etc)
    
    Files = os.listdir(directory)                                               #list all files
    files = []
    for file in Files:
        if (file.lower().rsplit('.',1)[1] in extensions) or ('all' in extensions):      #find extension and check membership in requested 
            files.append(file)
    
    return files                                                                #return file names that match requested extensions
    
def __make_folders(rootfolder:str,subfolder:str)->None:
    ## Making the dirs
    if(os.path.isdir(rootfolder)):                                              #Make/check for Root directory 
        vbPrint('Found dir: %s'%(rootfolder))
    else:
        vbPrint('Making dir: %s'%(rootfolder))
        os.mkdir(rootfolder)

    if(os.path.isdir(subfolder)):                                               #Make/check for world subfolder
        vbPrint('Found dir: %s'%(subfolder))
    else:
        vbPrint('Making dir: %s'%(subfolder))
        os.mkdir(subfolder)
    

          
def __getWindow(window_config:str):                                             #Called in genImgPatches()
    """
    Note:
        parses window_config to get height and width as integers
    
    Inputs:
        window_config (str): string of window height and width. Example: '5000,5000'
    
    Outputs:
        window (dict): dictionary containing ax:dim pairs
        
    """
    dims = window_config.split(',',1)
    axis = ('width','height')
    window = {ax:int(dim) for (ax,dim) in zip(axis,dims)} 
    return window
  
def __time_this(func):                                                          #Currently used on loadFiles & genImgPatches
    
    def __wrapper(*args,**kwargs):
        start = time.perf_counter()
        result = func(*args,**kwargs)
        end = time.perf_counter()
        
        delta = end-start
        ms = round((delta%1)*1000,2)
        
        ty_res = time.gmtime(delta)
        res = f"{time.strftime('h:%H m:%M s:%S',ty_res)} ms:{ms}"
        vbPrint(f'{func.__name__} took: {res}')
        
        if result is not None:
            return result
    
    return __wrapper

############## PRE TRAINING DATA PIPELINE FUNCTIONS #############



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
            
            

@__time_this
def loadFiles():
    '''
    loads world data from s3 into the directory:
        - ./data/<dataset>/world_<fileset>/
        
    
    s3Dir (str): bucket path to files
    file_formats (str): list of file formats to search for in s3Dir, gets converted to list. Ex: "jpg, png" -> ['jpg','png']
    
    '''
    ##----------------------------Configuration/Setup---------------------------##
    ds = config['ds']                                                           #dataset string. root folder of files
    fs = config['fs']                                                           #fileset string. folder to hold world files
    s3Dir = config['s3td']                                                      #AWS URI for folder located in S3 bucket 
    
    file_formats = config['fmts'].lower().replace(' ','').split(',')            #convert string of file formats to list of formats
    loadDir = '%s/world_%s'%(ds,fs)                                             #root folder + subfolder
    
    vbPrint('Load File(s) Starting...')
                                                                                
    __make_folders(rootfolder=ds,                                               ## Making the dirs
                   subfolder=loadDir)                                           #method for making project rootfolder/subfolders

    ##----------------------------Load Files------------------------------------##
                                                                                ##Retrieve aws bucket object                     
    #s3bucket = __getS3bucket(bn)                                                #S3 Bucket access. Uses boto3 + local aws credentials to access bucket, 
    
    
    load_s3_to_local(s3_uri=s3Dir,
                     desired_formats=file_formats,
                     load_dir=loadDir,
                     verbose=config['verbose'])
    
    
    vbPrint('Loading World file(s) Completed')
    
    '''
    #Examples:
    python3 dataPipeline.py --ds=dataset6 --fs=inputs --fmts=all --s3td=s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/DOM2015/Selected_500/pipelineTest/ -loadFiles
    
    python3 dataPipeline.py --ds=dataset6 --fs=inputs --fmts=all --s3td=s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/output/ -loadFiles
    
    '''




@__time_this  
def genImgPatches():
    '''
    Notes: 
        Generates image patches by splitting the tiles
        Prerequisite: loadFiles() has been run or tiles are present in the data/<dataset>/tiles directory
    
    Inputs: 
        ds (str): dataset folder. aka root folder for project
        fs (str): fileset folder. aka subfolder to hold patches
        mWH (int): maximum Width/Height of tiles
        fmts (str): what raster file types should be converted to patches 
                        (this is important since image files are associated w/ other file type like an .xml,.cpg,.dbf this makes sure those are ignored)
        save_fmt (str): file format for patches. Exs: jpeg, png
    
    outputs:
        folder containing patch image-files and any support files. 
    
    '''
    ##----------------------------Configuration/Setup---------------------------##
    ds = config['ds']                                                           #dataset (root folder)
    fs = config['fs']                                                           #fileset (subfolder)
    window = __getWindow(window_config=config['mWH'])                           #dictonary w/ max window dimensions (default: width:5000,height:5000) 
    
    file_formats = config['fmts'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    save_fmt = config['sfmt']                                                   #file format for patches when they are saved. (png,jpeg,GTiff) https://gdal.org/drivers/raster/index.html

    worldDir = '%s/world_%s'%(ds,fs)                                            #directory with files to convert to patches 
    worldFiles = __listdir(directory=worldDir,
                           extensions=file_formats)                             #list of files to convert to patches
                           
    patchesDir = '%s/imagePatches_%s'%(ds,fs)                                   #subfolder for saving patches
    vbPrint('%i World files found in %s'%(len(worldFiles),worldDir))            #display the number of files to be processed
    
    
    __make_folders(rootfolder=ds,                                               ##Make folder to hold patches
                   subfolder=patchesDir)                                        #method for making project rootfolder/subfolders
    
    
    ##----------------------------Create Patches------------------------------------##
    for count,imageName in enumerate(worldFiles):
        imgage_Path = fr'{worldDir}/{imageName}'                                #recreate image path 
        vbPrint(f'Current Image Path: {imgage_Path}')
        
        
        #Open world file & apply window to generate Tiles 
        for tile_array,tile_meta,tile_name in Open_Raster(path_to_world = imgage_Path,      #current world file
                                                          maxWidth = window['width'],       #this is maxWidth/maxHeight
                                                          maxHeight = window['height'],        #actual height/width may be less since no padding happens when opening
                                                          verbose = config['verbose']):
                                                             
            if 0 in tile_array.shape:
                vbPrint(f'empty array generated: {tile_array.shape}. Skipping')
                continue
        
                                                                                ## Creation of Image patches
            split_array = Split_Raster(image=tile_array,                        #current tile to split into patches
                                       patchX = 256,                            #patch width
                                       patchY = 256,                            #patch height
                                       minScrapPercent = 0,                     #0 means every tile_array will be padded before splitting (otherwise patches along bot/right edges are discarded)
                                       verbose = config['verbose'])
        
        
                                                                                ##Save Image Patches:
            Save_Patches(array=split_array,                                     #split tile containing patches
                         save_directory=patchesDir,                             #folder location to save patches in 
                         tile_name=tile_name,                                 #patches will be named as: tileName_tileXpos_tileYpos_patchXpos_patchYpos
                         profile = tile_meta,                                   #rasterio requires a 'profile'(meta data) to create/write files
                         save_fmt = save_fmt,                                   #format to save patches as (ex: png,jpeg,geoTiff) 
                         verbose = config['verbose'])

    
    ##----------------------------Quick Summary------------------------------------##
    patchFiles = __listdir(directory=patchesDir,
                           extensions=['all']) 
    
    vbPrint(f"Number of files created in {patchesDir}: {len(patchFiles)}\n{'-'*4}Image Patches made successfuly{'-'*4}")
    
    '''
    #Example:
    python3 dataPipeline.py --ds=dataset5 --fs=labels --fmts=tif --sfmt=png --mWH=5000,5000 -genImgPatches
    python3 dataPipeline.py --ds=dataset5 --fs=inputs --fmts=jpg --sfmt=png --mWH=5000,5000 -genImgPatches
    
    '''



'''
INVESTIGATION NEEDED FOR genAnnotations:
    - How to make annotations from this
'''


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
        - loadTiles - John
        - genImgPatches - John
        - genLabelPatches - TODO -> ARCGIS
        - genAnnotations - TODO  -> ARCGIS 
        ### MEVP Pipeline
        - genInferenceJSON - Done
        - genInferenceTiles - Done
        - Vectorize - TODO -> Rasterio or PyQGIS or ArcPy (Licensing)
        - Cleaning and Metrics
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