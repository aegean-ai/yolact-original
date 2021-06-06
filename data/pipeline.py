"""
Definitions:
    > Region-File/Region-Image: This is any input image of arbitrary size that must be broken down into patches.
    > Tile-Image: This is any subdivision of a World-Image. The default is 5000x5000.
    > Patch(s)/ Image-Patch: This is the smallest subdivision of a World-Image. This is hardcoded 256x256
    > The patch size is set according to the input tensor size for the Yolact Model
    > Dataset
      > Each project (like upabove) can be considered one dataset. For the current scope of the project, only 
      one dataset would be made i.e. for The NJ County. 
      > It is possible that we want to work in a project similar to
      upabove, say to do object detection on a completely new type of images i.e. we want to use the same pipeline for an entirely
      different project. 
      > To ensure separation of this data, a new dataset can be made for the new project.
      > The dataset is abbreviated as 'ds' throughout the documentation and code.
    > Tileset
      > Each batch of tiles can be considered as a tileset. A dataset may have multiple tilesets.
      > Any new batch of data that arrives should be treated like a new tileset. 
      > All Pipeline steps take the dataset and tileset information and process only the mentioned dataset and tileset. 
      This ensures that we don't have to unnecessarily process any old data in the project.
      > The tileset is abbreviated as 'ts' throughout the documentation and code.

General Info:
    > This file has the code to perform all steps of the pipeline.
    > The individual steps can be found in our github documentation: docs.upabove.app/sidewalk/_index.md
    > The configuration of the pipeline as well as the steps to perform are passed as terminal arguments 

Structure:
This code in this file is present in N sections:
    1. Imports/Setup
        > Has all imports required throughout the code.
    2. Helper Functions
        > Common functions used throughout the code.
    3. Pipeline Funtions
        > Each step of the pipeline is modularized as one function.
        > The functions are meant to be completely independent i.e. as long as prequisite the data is present,
          They should run regardless of the fact that other pipeline steps have run before it.
        > The pipeline Functions are further divided into 2 parts:
            1. Data Pipeline Functions (Prior to Model Training)
            2. Model Evaluation and Verification Functions (After Model Training)
    4. Main Function

Working logic of the Pipeline:
    > Each pipeline step function works with one part of the pipeline. It has well defined input data and output data formats and locations.
      This information is available as a comment over each pipeline function.
    > Each step takes a set of parameters. These parameters are passed as arguments. The applicable arguments are also provided 
      as comments along with the functions.

The complete Pipeline:
    ### Data Pipeline
    - loadTiles #TODO
    - genImgPatches
    - genAnnotations
    ### MEVP Pipeline
    - genInferenceJSON
    - genInferenceData #TODO -> GeoJSON projections
    - Cleaning and Metrics #TODO
    - exportData #TODO

    Attributes: 
    
    Todo: 
    
"""

#----------------------------------------------------------------#
#------------------------ Imports/Setup -------------------------#
#----------------------------------------------------------------#

import os
import traceback
import sys
import json
from s3_helper import load_s3_to_local,load_local_to_s3
from split_merge_raster import Open_Raster,Split_Raster,Save_Tile,Save_Patches            #These are all called w/in genImgPatches
import time                                                                               #used in decorator __time_this()
from pycocotools import mask
import numpy as np
import cv2
from raster2coco import Raster2Coco
import pyproj
# The default configs, may be overridden
import wget                                                                     #called within download_labels
from zipfile import ZipFile                                                     #called within download_labels

# This is the default configuration of parameters for the pipeline
config = {
    'pipeline':[], # Steps of the pipeline
    'verbose': 1, # If set to 0, does not print detailed information to screen
    'ds':None, # The Dataset
    'ts':None, # The Tileset
    'pef': 0.1, # Print-every-factor: Prints percentage completion after ever 'pef' factor of completion.
    'score_threshold':0.0, # During evaluation of inferences from raw data using Model: Minimum score threshold for detecting sidewalks.
    'det_score_threshold':0.0, # During generation of inference data from already generated inferenceJSON: Minimum sidewalk detection score threshold to include detection. 
    'rowsSplitPerTile':20, # Expected rows per tile
    'colsSplitPerTile':20, # Expected columns per tile
    'patchDimX':256, # Patch dimension X
    'patchDimY':256, # Patch dimension Y
    'tileDimX':5000, # Tile Dimension X
    'tileDimY':5000, # Tile Dimension Y
    'mWH': '5000,5000', # maximum Width/Height of tiles: used in genImagePatches
    'fmts':'jpg,png,tif', # image formats to consider when reading files
    'sfmt':'jpeg', # Save format for generated patches in genImagePatches
    'tvRatio':0.8, # The train-validation ratio used for generating training annotations
    'genImgs':1, # in genInferenceData Setting this to 0 skips saving images 
    'genGeoJSON':1 # in genInferenceData Setting this to 0 skips generation of geoJSON 
}

#----------------------------------------------------------------#
#----------------------- Helper Functions -----------------------#
#----------------------------------------------------------------#

def vbPrint(s):
    """print only if verbose is enabled

    Args:
        s ([type]): [description]
    """
    
    if(config['verbose']==1):
        print(s)

def setConfig(argv):
    '''
    > Takes the terminal arguments as input
    > Sets configuration variable for the pipeline or sets pipeline step depending on argument signature

    Argument Signatures:
        > Configuration Variable    : --<KEY>=<VALUE>
        > Pipeline Step             : -<PIPELINE_STEP>
    '''
    
    # This sets the types for each argument. Any new arguments that are not strings need to have their types defined.
    # The arguments will be parsed as these, if not parsable in the below format, the program will terminate.
    # The default is argument is str
    argType = {
        'verbose': int,
        'pef' : float,
        'score_threshold': float,
        'det_score_threshold':float,
        'rowsSplitPerTile':int,
        'colsSplitPerTile':int,
        'patchDimX':int,
        'patchDimY':int,
        'tvRatio':float
    }
    
    global config
    argv = argv[1:]
    if('--verbose=0' in argv):
        argv = ['--verbose=0']+argv

    # Processes each argument based on their argType and overwrites their value in the config variable
    # If it is a pipeline step, it gets added into the pipeline.
    # If an argument cannot be parsed, an error is thrown and the program quits.
    # The pipeline only runs after all arguments have been parsed.
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

def __listdir(directory:str,extensions:list)->list:                             #list files with specified extensions (filter for tif/png/jpeg etc)
    """Returns a list of files that match the given extensions .

    Args:
        directory (str): [description]
        extensions (list): [description]

    Returns:
        list: [description]
    """
    
    Files = os.listdir(directory)                                               #list all files
    files = []
    for file in Files:
        if (file.lower().rsplit('.',1)[1] in extensions) or ('all' in extensions):      #find extension and check membership in requested 
            files.append(file)
    
    return files                                                                #return file names that match requested extensions
    
def __make_folders(rootfolder:str,subfolder:str)->None:
    """Create the directories in the specified rootfolder and subfolder .

    Args:
        rootfolder (str): [description]
        subfolder (str): [description]
    """
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
    """Parses window_config to get height and width as integers
    
    Args:
        window_config (str): string of window height and width. Example: '5000,5000'
    
    Outputs:
        window (dict): dictionary containing ax:dim pairs
        
    """
    dims = window_config.split(',',1)
    axis = ('width','height')
    window = {ax:int(dim) for (ax,dim) in zip(axis,dims)} 
    return window
  
def __time_this(func):                                                          #Currently used on loadFiles & genImgPatches
    """CPU Time of a function execution.

    Args:
        func ([type]): [description]
    """
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

#----------------------------------------------------------------#
#-------------------- Data Pipeline Functions -------------------#
#----------------------------------------------------------------#
@__time_this
def loadFiles():
    '''
    > This function loads all required files from S3.
    > The files required at this stage of the pipeline would be the Tiles, Labels, and their World Files
    > This code may have to change depending on how the Input data is available. 

    [Code by John]
    Note:
        loads region data from s3 into the directory:
            - ./data/<dataset>/region_<fileset>/
    
    Inputs:    
            ds (str): local dataset folder (root folder)
            fs (str): local fileset folder (sub folder)
            s3td (str): bucket path to files
            fmts (str): list of file formats to search for in s3Dir, gets converted to list. Ex: "jpg, png" -> ['jpg','png'], "all" -> ["all"] (all returns all files)
    '''
    ##----------------------------Configuration/Setup---------------------------##
    ds = config['ds']                                                           #dataset string. root folder of files
    fs = config['fs']                                                           #fileset string. folder to hold world files
    s3Dir = config['s3d']                                                       #AWS URI for folder located in S3 bucket 
    
    file_formats = config['fmts'].lower().replace(' ','').split(',')            #convert string of file formats to list of formats
    loadDir = '%s/region_%s'%(ds,fs)                                             #root folder + subfolder
    
    vbPrint('Load File(s) Starting...')
                                                                                
    __make_folders(rootfolder=ds,                                               ## Making the dirs
                   subfolder=loadDir)                                           #method for making project rootfolder/subfolders

    ##----------------------------Load Files------------------------------------##
    
                                                                                ##Retrieve aws bucket object    
    load_s3_to_local(s3_uri=s3Dir,                                              #S3 Bucket access. Uses boto3 + local aws credentials to access bucket
                     desired_formats=file_formats,
                     load_dir=loadDir,
                     verbose=config['verbose'])
    
    
    vbPrint('Loading Region File(s) Completed')
    
    '''
    #Examples:
    python3 dataPipeline.py --ds=dataset7 --fs=inputs --fmts=all --s3d=s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/DOM2015/Selected_500/pipelineTest/ -loadFiles
    
    python3 dataPipeline.py --ds=dataset7 --fs=labels --fmts=all --s3d=s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/output/ -loadFiles
    
    python3 dataPipeline.py --ds=dataset7 --fs=vectors --fmts=geojson --s3d=s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/labels_ground_truth/year-2/vector_files/ -loadFiles 
    '''


    
        
    
@__time_this  
def genImgPatches():
    '''
    [Code by John]
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
    fs = config['fs']                                                           #existing fileset (subfolder)
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    window = __getWindow(window_config=config['mWH'])                           #dictonary w/ max window dimensions (default: width:5000,height:5000) 
    
    file_formats = config['fmts'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    save_fmt = config['sfmt']                                                   #file format for patches when they are saved. (png,jpeg,GTiff) https://gdal.org/drivers/raster/index.html

    worldDir = '%s/region_%s'%(ds,fs)                                           #directory with files to convert to patches 
    worldFiles = __listdir(directory=worldDir,
                           extensions=file_formats)                             #list of files to convert to patches
    
    if ts is not None:                                                          #subfolder for saving tiles
        tilesDir = '%s/imageTiles_%s'%(ds,ts)
        __make_folders(rootfolder=ds,
                       subfolder=tilesDir)    
    
                           
    patchesDir = '%s/imagePatches_%s'%(ds,fs)                                   #subfolder for saving patches
    vbPrint('%i Region files found in %s'%(len(worldFiles),worldDir))           #display the number of files to be processed
    
    
    __make_folders(rootfolder=ds,                                               ##Make folder to hold patches
                   subfolder=patchesDir)                                        #method for making project rootfolder/subfolders
    
    

    
    
    ##----------------------------Create Patches------------------------------------##
    for count,imageName in enumerate(worldFiles):
        imgage_Path = fr'{worldDir}/{imageName}'                                #recreate image path 
        vbPrint(f'Current Image Path: {imgage_Path}')
        
        
        #Open world file & apply window to generate Tiles 
        for tile_array,tile_meta,tile_name in Open_Raster(path_to_region = imgage_Path,      #current world file
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
        
            if ts is not None:
                Save_Tile(array=tile_array,                                     #tile array
                          save_directory=tilesDir,                              #folder location to save tile in 
                          tile_name=tile_name,                                  #patches will be named as: region_tile_{tileXpos}_{tileYpos}
                          profile = tile_meta,                                  #rasterio requires a 'profile'(meta data) to create/write files
                          save_fmt = save_fmt,                                  #format to save tiles as (ex: png,jpeg,geoTiff) 
                          verbose = config['verbose'])
            
                                                                                ##Save Image Patches:
            Save_Patches(array=split_array,                                     #split tile containing patches
                         save_directory=patchesDir,                             #folder location to save patches in 
                         tile_name=tile_name,                                   #patches will be named as: tileName_tileXpos_tileYpos_patchXpos_patchYpos
                         profile = tile_meta,                                   #rasterio requires a 'profile'(meta data) to create/write files
                         save_fmt = save_fmt,                                   #format to save patches as (ex: png,jpeg,GTiff) 
                         verbose = config['verbose'])

    
    ##----------------------------Quick Summary------------------------------------##
    if ts is not None:
        tileFiles = __listdir(directory=tilesDir,
                              extensions=['all']) 
        vbPrint(f"Number of files created in {tilesDir}: {len(tileFiles)}\n{'-'*4}Image Tiles made successfuly{'-'*4}")
        
        
    patchFiles = __listdir(directory=patchesDir,
                           extensions=['all']) 
                           
    vbPrint(f"Number of files created in {patchesDir}: {len(patchFiles)}\n{'-'*4}Image Patches made successfuly{'-'*4}")
    
    '''
    #Example:
    python3 dataPipeline.py --ds=dataset5 --fs=labels --fmts=tif --sfmt=png --mWH=5000,5000 -genImgPatches
    python3 dataPipeline.py --ds=dataset5 --fs=inputs --fmts=jpg --sfmt=png --mWH=5000,5000 -genImgPatches
    python3 dataPipeline.py --ds=dataset7 --fs=labels --ts=labels --fmts=tif --sfmt=gtiff --mWH=5000,5000 -genImgPatches
    python3 dataPipeline.py --ds=dataset7 --fs=labels --ts=labels --fmts=tif --sfmt=tiff --mWH=5000,5000 -genImgPatches
    python3 dataPipeline.py --ds=dataset8 --fs=labels --ts=labels --fmts=gtiff --sfmt=gtiff --mWH=5000,5000 -genImgPatches
    python3 dataPipeline.py --ds=dataset9 --fs=labels --ts=labels9 --fmts=gtiff --sfmt=gtiff --mWH=5000,5000 -genImgPatches
    '''

def genAnnotations():
    '''
    > Generates the train and test annotations files using the image and label patches
    
    Requirements:
        > label patches should be available in <ds>/labelPatches_<ts>/..
        > While not required by this function, corresponding image patches should be available in <ds>/imagePatches_<ts>/..

    Generates:
        > Two annotation files to be used by the model to train:
            1. <ds>/annotations_<ts>/annotations_train.json
            2. <ds>/annotations_<ts>/annotations_test.json

    '''
    ds = config['ds']
    ts = config['ts']
    tvRatio = config['tvRatio']

    trainAnnFilename = 'annotations_train.json'
    testAnnFilename = 'annotations_test.json'

    labelPatchesDir = '%s/labelPatches_%s'%(ds,ts)
    annDir = '%s/annotations_%s'%(ds,ts)

    ## Making the dirs
    if(os.path.isdir(annDir)):
        vbPrint('Found dir: %s'%(annDir))
    else:
        vbPrint('Making dir: %s'%(annDir))
        os.mkdir(annDir)

    ## Getting the labels and splitting into training and validation images
    vbPrint('Reading `%s` for label patches'%(labelPatchesDir))
    labels = __listdir(labelPatchesDir, ['tif'])
    vbPrint('Dataset Size   : %i'%(len(labels)))
    labels = np.asarray(labels)
    np.random.shuffle(labels)
    splitIdx = int(labels.shape[0]*tvRatio)
    trainData,valData = np.split(labels,[splitIdx])
    
    vbPrint('Training Data  : %i'%(len(trainData)))
    vbPrint('Val Data       : %i'%(len(valData)))
    
    # Generate Annotations
    vbPrint('Generating annotations for the training data')
    trainR2C = Raster2Coco(trainData, labelPatchesDir)
    trainJSON = trainR2C.createJSON()

    with open('%s/annotations_train.json'%(annDir), 'w+') as outfile:
        json.dump(trainJSON, outfile)
    
    del trainJSON
    del trainR2C

    vbPrint('Generating annotations for the validation data')
    valR2C = Raster2Coco(valData, labelPatchesDir)
    valJSON = valR2C.createJSON()

    with open('%s/annotations_val.json'%(annDir), 'w+') as outfile:
        json.dump(valJSON, outfile)
    
    del valJSON
    del valR2C

    vbPrint('Annotations made and saved successfuly')


#----------------------------------------------------------------#
#--------- Model Evaluation and Verification Functions ----------#
#----------------------------------------------------------------#
def genInferenceJSON():
    '''
    > Uses shell commands to run the eval.py for the configured parameters of the pipeline
    > This generates inferences in a JSON file saved in <ds>/inferencesJSON_<ts>/
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

def genInferenceData():
    '''
    Generates Upto 2 types of inference data from the inferences JSON output by the model which is generated by genInferenceJSON():
        1. Inference GeoJSON: a single geoJSON that has all the inferences as vector polygons. The file is a .geojson in <ds>/inferenceGeoJSON_<ts>
        2. Inference Tiles: Merged inferences for each of the 5000x5000 tiles, each tile is available as a .png images in <ds>/inferenceTiles_<ts>
    Requirements:
        > The required annotation file must be present in <ds>/annotations_<ts>
        > The inferences JSON generated by the model must be present in <ds>/inferencesJSON_<ts>/..
    
    Algorithm:
        > The detections are available based on image_id. This id does not correspond to any tile or patch identification directly.
        > The annotations file has, for each image_id, the associated tile number and patch co-ordinates encoded into the image name.
        > We first read the entire annotations file and create a HashMap that stores, for each tile: a list of all patches' information.
        > The information for a patch includes its image_id, its tile number, and its patch co-ordinates
        > Once this hashMap is prepared, we are ready to iterate on a tile by tile basis:
            > For each tile, we have the patch info for all patches. For each patch:
                > We have the tile number which is the 'tile' itself
                > We have raw image present with the name 'tile_<patch Co-ordinate X>_<patch Co-ordinate Y>'
                > We have the associated patch detections with the 'image_id' in the inferences JSON
            > Essentilly: We have a mapping between the tile and image_ids for all patches in the tile, for all tiles.
            > For each patch:
                > We have the detection vectors. This is in a COCO vector format and needs to be rasterized.
                > For the 256x256 patch, we have many detections each with a score.
                > We generate an empty image 256x256 image filled with 0s.
                > For each detection, we rasterize the detection and add it to the raster image if it is above the configurable threshold.
                > Sidewalk pixels are marked 255. If multiple detections mark the same pixel as a sidewalk, it is still 255.
                > 255 was chosen instead of 1 for the sake of simplicity: If viewed as an image, this raster shows the sidewalk in white
                  with a black background very clearly.
            > Essentilly: For each 'image_id' (i.e. for each patch), we now have a rasterized inference.
            > For each tile:
                > We now go through the all of patches, pull their rasters from the HashMap.
                > Since the patch information includes the co-ordinates of a patch in a tile, we know the indexes of the 256x256 block in the 5120x5120 tile.
                > Using this info, we generate a single 5120x5120 raster for each tile. This is then cropped to 5000x500px similar to year-1.
                > Generation of the GeoJSON:
                    > Since we know the tile i.e. the path and name of the tile image, we can infer the path and name of the respective world file
                    > The Raster can be converted into polygon vectors
                    > For each detected polygon of >3 points, it is added into the geoJSON.
                    > The geoJSON file is initialized at the start of the code, opened for processing each tile and closed. 
        > Once all the tiles are processed, the appropriate line endings are appended to the geoJSON.

    Notes:
        > The saving images can be skipped the argument '--genImgs=0'
        > The generation of geoJSON can be skipped with the argument '--genGeoJSON=0'
        > If both arguments are provided, nothing is saved. (Possible use-case is for debugging)
    '''
    ds = config['ds']
    ts = config['ts']
    tilesDir = '%s/tiles_%s'%(ds,ts)
    inferenceTilesDir = '%s/inferenceTiles_%s'%(ds,ts)
    inferenceGeoJSONDir = '%s/inferenceGeoJSON_%s'%(ds,ts)
    geoJSONFilename = '%s_%s.geojson'%(ds,ts)
    genImgs = config['genImgs']
    genGeoJSON = config['genGeoJSON']

    # Making the dirs
    if(os.path.isdir(inferenceTilesDir)):
        vbPrint('Found dir: %s'%(inferenceTilesDir))
    else:
        vbPrint('Making dir: %s'%(inferenceTilesDir))
        os.mkdir(inferenceTilesDir)

    if(os.path.isdir(inferenceGeoJSONDir)):
        vbPrint('Found dir: %s'%(inferenceGeoJSONDir))
    else:
        vbPrint('Making dir: %s'%(inferenceGeoJSONDir))
        os.mkdir(inferenceGeoJSONDir)

    detFilePath = '%s/inferencesJSON_%s/%s'%(ds,ts,config['infJSON'])
    annFilePath = '%s/annotations_%s/%s'%(ds,ts,config['annJSON'])
    #annFilePath = '%s/inference_annotations/%s'%(ds,config['annJSON']) # For testing. Can be reverted once the data-pipeline part is completed.

    if(genGeoJSON):
        # Initialize GeoJSONFile
        with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'w+') as geoJSONf:
            geoJSONf.write('{"type": "FeatureCollection","name": "sidewalk_detections_%s_%s","features": ['%(ds,ts))

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

    #### epsg: 3702 is close but thats Wyoming east
    #  
    # Initialize transformer
    #   From:     epsg projection 32111 - nad83 / new jersey
    #   To:       lat longs
    projectionTransformer = pyproj.Transformer.from_crs("epsg:32111","epsg:4326",always_xy=False)

    i = 0
    sidewalkCount = 0
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

            if(genGeoJSON):
                # Getting affine transform parameters from the world file
                with open('%s/%s.JGw'%(tilesDir,tile)) as worldFile:
                    rows = worldFile.read().split('\n')

                A = float(rows[0])
                D = float(rows[1])
                B = float(rows[2])
                E = float(rows[3])
                C = float(rows[4])
                F = float(rows[5])

                converter = Raster2Coco(None,None)
                binMask = np.zeros_like(infTile)
                binMask[infTile==255]=1
                vectors = converter.binaryMask2Polygon(binMask)
                JSONRows = ''
                for sidewalk in vectors:
                    # Skipping any triangles
                    if(len(sidewalk) >= 4):
                        sidewalkCount += 1
                        # Applying affine transform
                        '''
                        print(('-'*10)+'\nSidewalk pixels vector')
                        print(sidewalk)
                        
                        print(('-'*10)+'\nUTM Vector')
                        print(['[%f,%f]'%(
                                ((A*x) + (B*y) + C,
                                (D*x) + (E*y) + F)
                        ) for y,x in sidewalk])
                        '''
                        vecLi = ['[%f,%f]'%(
                            projectionTransformer.transform(
                                ((A*x) + (B*y) + C),
                                ((D*x) + (E*y) + F)
                            )[::-1]
                        ) for x,y in sidewalk]
                        #print(('-'*10)+'\nlat long vector')
                        #print(vecLi)

                        vecStr = '[[%s]]'%(','.join(vecLi))
                        #print(('-'*10)+'\nFinal vector string for ')
                        #print('%s'%(vecStr))
                        rowStr = ',\n{"type":"Feature","properties":{"objectid":%i}, "geometry":{ "type": "Polygon", "coordinates":%s}}'%(sidewalkCount,vecStr)
                        
                        # Skip comma for first sidewalk
                        if(sidewalkCount == 1):
                            rowStr = rowStr[1:]

                        JSONRows += '%s'%(rowStr)
                    
                if(JSONRows != ''):
                    with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'a+') as geoJSONf:
                        geoJSONf.write(JSONRows)

            if(genImgs):
                # Writing the image tile
                # File format kept as png because it has lossless compression and to work well with rasterio if needed.
                cv2.imwrite('%s/%s.png'%(inferenceTilesDir,tile),infTile)

            # Prints progress at every 'pef' factor of completion
            # PEF stands for 'Print-Every factor'
            facComp = (i+1)/(n)
            if(facComp >= peFactor):
                vbPrint('Generating inferences tile: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
                peFactor += config['pef']

    if(genGeoJSON):
        #print('TILE: ',tile)
        with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'a+') as geoJSONf:
            geoJSONf.write('\n]}')

    if(genGeoJSON and genImgs):
        vbPrint('Inference GeoJSON and Tiles generated Successfully')
    elif(genGeoJSON):
        vbPrint('Inference GeoJSON generated Successfully')
    elif(genImgs):
        vbPrint('Inference Tiles generated Successfully')
    else:
        vbPrint('Inference Data ran successfully. Nothing saved due to arguments passed.')

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

    genAnnotations Arguments:
        --tvRatio           -> The train/validation split ratio. For ex: 0.8 means out of 100 images, 80 will be train and 20 will be validation. 

    genInferenceJSON Arguments:
        --trained_model     -> [path] Path to the trained model. Ex: weights/DVRPCResNet50_8_88179_interrupt.pth
        --config            -> [config name] name of the config to be used from config.py. Example dvrpc_config
        --score_threshold   -> [float] The score threshold to use for inferences. Example: 0.0

    genInferenceData Arguments:
        --annJSON           -> [path] name of the annotations .JSON file. Example: DVRPC_test.json
        --infJSON           -> [path] name of the created inference .JSON file. Example: DVRPCResNet50.json
        --genGeoJSON        -> [0 or 1] Skips generation of geoJSON if set to 0. Default is 1
        --genImgs           -> [0 or 1] Skips generation of images if set to 0. Default is 1 

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
            python dataPipeline.py --ds=ds1 --ts=inf1 -genInferenceData --annJSON=DVRPC_test.json --infJSON=DVRPCResNet50.json
            ```
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