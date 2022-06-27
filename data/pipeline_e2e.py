"""
Definitions:
    * Region-File/Region-Image: This is any input image of arbitrary size that must be broken down into chips.
    * Tile-Image: This is any subdivision of a World-Image. The default is 5000x5000.
    * Chip/ Image-Chip: This is the smallest subdivision of a World-Image. This is hardcoded 256x256
    * The chip size is set according to the input tensor size for the Yolact Model
    
    * Project Root
      * Each project has one project root abbreviated as 'project_root' throughout the documentation and code.
    
    * Tileset
      * Each batch of tiles can be considered as a tileset. A project root may have multiple tilesets.
      * Any new batch of data should be treated like a new tileset. 
      * All Pipeline steps take the project_root and tileset information and process only the tileset under the project_root ensuring  that we don't unnecessarily 
      process older data in the project.
      * The tileset is abbreviated as 'ts' throughout the documentation and code.

General Info:
    * This file performs all steps of the pipeline.
    * The individual steps can be found in our github documentation site: docs.sidewalk.upabove.app
    * The configuration of the pipeline as well as the steps to perform are passed as terminal arguments and are also available in VS Code's launch.json allowing 
    GUI based launch of the various stages through the  VSCode Python debug plugin.

Structure:
    This code in this file is present in N sections:
    Imports/Setup
        * Has all imports required throughout the code.
    Helper Functions
        * Common functions used throughout the code.
    Pipeline Funtions
        * Each step of the pipeline is modularized as one function.
        * The functions are meant to be completely independent i.e. as long as prequisite the data is present,
          They should run regardless of the fact that other pipeline steps have run before it.
        * The pipeline Functions are further divided into 2 parts:
            1. Data Pipeline Functions (Prior to Model Training)
            2. Model Evaluation and Verification Functions (After Model Training)
    Main Function

Working logic of the Pipeline:
    * Each pipeline step function works with one part of the pipeline. It has well defined input data and output data formats and locations.
      This information is available as a comment over each pipeline function.
    * Each step takes a set of parameters. These parameters are passed as arguments. The applicable arguments are also provided 
      as comments along with the functions.

The complete Pipeline:
    Data Pipeline
    * loadTiles #TODO
    * genImgChips
    * genAnnotations
    MEVP Pipeline
    * genEvalJSON
    * genTileGeoJSON #TODO -> GeoJSON projections
    * Cleaning and Metrics #TODO
    * exportData #TODO

    Attributes: 
    
    Todo: 
    
"""

#----------------------------------------------------------------#
#------------------------ Imports/Setup -------------------------#
#----------------------------------------------------------------#

import os
import fnmatch
from os.path import exists
import traceback
import sys
import json
from s3_helper import s3_to_local
from split_merge_raster import convert_tiles, open_raster,split_raster,save_tile,save_chips            #These are all called w/in genImgChips
from split_merge_vector import _partitionGeoJSON, find_files
import time                                                                               #used in decorator __time_this()
from pycocotools import mask
import numpy as np
import cv2
from raster2coco import Raster2Coco
import pyproj
from strawberry import scalar
# The default configs, may be overridden
import wget                                                                     #called within download_labels
from zipfile import ZipFile                                                     #called within download_labels
from pathlib import Path, PurePath
import rasterio as rio
import geopandas as gpd
import subprocess
from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler

# This is the default configuration of parameters for the pipeline
config = {
    'pipeline':[], # Steps of the pipeline
    'verbose': True, # If set to False, does not print detailed information to screen
    'project_root': None, # the project root
    'fs': None, # The Fileset
    'ts':None, # The Tileset
    'pef': 0.1, # Print-every-factor: Prints percentage completion after ever 'pef' factor of completion.
    'score_threshold':0.0, # During evaluation of inferences from raw data using Model: Minimum score threshold for detecting sidewalks.
    'det_score_threshold':0.0, # During generation of inference data from already generated inferenceJSON: Minimum sidewalk detection score threshold to include detection. 
    'rowsSplitPerTile':20, # Expected rows per tile
    'colsSplitPerTile':20, # Expected columns per tile
    'chipDimX':256, # Chip dimension X
    'chipDimY':256, # Chip dimension Y
    'chipWindowStride': 192, # stride of the sliding window that patchifies the tile into chips
    'inferenceTileDimX':5000, # Tile Dimension X
    'inferenceTileDimY':5000, # Tile Dimension Y
    'mWH': '5000,5000', # maximum Width/Height of tiles: used in genImageChips
    'trainingTileDimX':7000, # Tile Dimension X
    'trainingTileDimY':7000, # Tile Dimension Y
    'file_extensions':'jpeg, jpg, jgw, png, pgw, tiff, tif, tfw', # image formats to consider when reading files
    'input_file_format': 'tif',
    'input_image_near_infrared':True,
    'source_tile_file_format': 'jpeg',
    'target_tile_file_format': 'jpeg',
    'chip_file_format': 'jpeg', #'tif', ## Save format for generated chips in genImageChips
    'tvRatio':1.0, # The default train-validation ratio used for generating training annotations (1.0: all data are training, 0.0: all data are validation)
    'genImgs':True, # in genTileGeoJSON Setting this to 0 skips saving images 
    'genGeoJSON':True, # in genTileGeoJSON Setting this to 0 skips generation of geoJSON
    'aws_sso_flag':False,
    'trainingImageryRootPath':None, # the path to the tiles that will be used for model training
    'tileIndexPathFile':None,
    'featuresGeoJSONRootPathFile':None,
    'buffer_value':int,
    'tile_geojson_target_crs': str, # the crs of the generated tile geojsons. 
    'sidewalk_annotation_threshold':int
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
    """
    * Takes the terminal arguments as input
    * Sets configuration variable for the pipeline or sets pipeline step depending on argument signature

    Argument Signatures:
        * Configuration Variable    : --<KEY>=<VALUE>
        * Pipeline Step             : -<PIPELINE_STEP>
    """
    
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
        'chipDimX':int,
        'chipDimY':int,
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
    filtered_files = []
    for file in Files:
        if (file.lower().rsplit('.',1)[1] in extensions):      # if desired extension  
            filtered_files.append(file)
    
    #return file names that match requested extensions    
    return filtered_files                                                                
    
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
    

          
def __getWindow(window_config:str):                                             #Called in genImgChips() 
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
  
def __time_this(func):                                                          #Currently used on loadFiles & genImgChips
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
    """
    * This function loads all required files from S3.
    * The files required at this stage of the pipeline would be the Tiles, Labels, and their World Files
    * This code may have to change depending on how the Input data is available. 

    Note:
        loads region data from s3 into the directory:
            * ./data/<dataset>/region_<fileset>/
    
    Inputs:    
            project_root (str): local dataset folder (root folder)
            fs (str): local fileset folder (sub folder)
            s3td (str): bucket path to files
            file_extensions (str): list of file formats to search for in s3Dir, gets converted to list. Ex: "jpg, png" -> ['jpg','png'], "all" -> ["all"] (all returns all files)
    
    Example: 
    
    python3 pipeline_e2e.py --project_root=project_root_10 --fs=inputs --file_extensions=all --s3d=s3://bucket-name  -loadFiles
    """
    ##----------------------------Configuration/Setup---------------------------##
    project_root = config['project_root']                                                           #dataset string. root folder of files
    fs = config['fs']                                                           #fileset string. folder to hold world files
    s3Dir = config['s3d']                                                       #AWS URI for folder located in S3 bucket 
    aws_sso_flag = config['aws_sso_flag']
    
    file_formats = config['file_extensions'].lower().replace(' ','').split(',')            #convert string of file formats to list of formats
    target_dir = '%s/region_%s'%(project_root,fs)                                             #root folder + subfolder
    
    vbPrint('Load File(s) Starting...')
    
    # Making the dirs                                                                     
    __make_folders(rootfolder=project_root, subfolder=target_dir)                                           

    ##----------------------------Download Files------------------------------------##
    
    ##Retrieve aws bucket object    
    s3_to_local(
        s3_uri=s3Dir,                                             
        desired_formats=file_formats,
        target_dir=target_dir,
        aws_sso_flag=aws_sso_flag,
        verbose=config['verbose']
    )
    
    vbPrint('Loading Region File(s) Completed')


@__time_this  
def genImgChips():
    """
    Notes: 
        Generates image chips by splitting the tiles
        Prerequisite: loadFiles() has been run or tiles are present in the data/<dataset>/tiles directory
    
    Inputs: 
        project_root (str): dataset folder. aka root folder for project
        fs (str): fileset folder. aka subfolder to hold chips
        mWH (int): maximum Width/Height of tiles
        file_extensions (str): what raster file types should be converted to chips 
                        (this is important since image files are associated w/ other file type like an .xml,.cpg,.dbf this makes sure those are ignored)
        chip_file_format (str): file format for chips. Exs: jpeg, png
    
    outputs:
        folder containing chip image-files and any support files. 

    Example:
        python3 pipeline_e2e.py --project_root=project_root --fs=labels --file_extensions=tif --chip_file_format=png --mWH=5000,5000 -genImgChips
    
    """

    ##----------------------------Configuration/Setup---------------------------##
    project_root = config['project_root']                                                           #dataset (root folder)
    fs = config['fs']                                                           #existing fileset (subfolder)
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    window = __getWindow(window_config=config['mWH'])                           #dictonary w/ max window dimensions (default: width:5000,height:5000) 
    
    file_formats = config['file_extensions'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    source_tile_file_format = config['source_tile_file_format']                                                   #file format for tiles
    target_tile_file_format = config['target_tile_file_format']                                                   #file format for tiles
    chip_file_format = config['chip_file_format']                                                   #file format for chips when they are saved. (png,jpeg,tif) https://gdal.org/drivers/raster/index.html

    regionDir = '%s/region_%s'%(project_root,fs)                                           #directory with files to convert to chips 
    regionFiles = __listdir(directory=regionDir, extensions=file_formats)                             #list of files to convert to chips
    
    
    tilesDir = '%s/imageTiles_%s'%(project_root,ts)
    __make_folders(rootfolder=project_root,
                    subfolder=tilesDir)    
                            
    chipsDir = '%s/imageChips_%s'%(project_root,ts)                                   #subfolder for saving chips
    vbPrint('%i Region files found in %s'%(len(regionFiles),regionDir))           #display the number of files to be processed
    
    __make_folders(rootfolder=project_root,                                               ##Make folder to hold chips
                   subfolder=chipsDir)                                        #method for making project rootfolder/subfolders
    
        
    # if the tile has 4 bands with band 4 being the Near Infra Red (NIR) band the  say so
    if config['input_image_near_infrared']:
        input_image_near_infrared=True

    # If the tile is in GeoTIFF format, then convert to 8-bit JPEG 
    if not target_tile_file_format==source_tile_file_format:
        convert_tiles(sourceIsNearInfrared=input_image_near_infrared, sourceTileFormat=source_tile_file_format, targetTileFormat=target_tile_file_format, sourceDir=regionDir, targetDir=regionDir)

    # list of tiles converted to JPEG that need further split into chips
    tilesFiles = __listdir(directory=regionDir, extensions=target_tile_file_format)                             
    
    # Loop over tiles
    for count,imageName in enumerate(tilesFiles):
        image_path = fr'{regionDir}/{imageName}'                                #recreate image path 
        vbPrint(f'Current Image Path: {image_path}')
        
        #Open raster files & apply window to generate chips 
        for tile_array, tile_meta, tile_name in open_raster(
            path_to_files = image_path,      #current tile
            maxWidth = window['width'],       #this is maxWidth/maxHeight
            maxHeight = window['height'],        #actual height/width may be less since no padding happens when opening
            verbose = config['verbose']):
                                                             
            if 0 in tile_array.shape:
                vbPrint(f'empty array generated: {tile_array.shape}. Skipping')
                continue
            
            ## Creation of Image chips
            split_array = split_raster(
                image_array=tile_array,                        #current tile to split into chips
                chipX = 256,                            #chip width
                chipY = 256,                            #chip height
                minScrapPercent = 0,                     #0 means every tile_array will be padded before splitting (otherwise chips along bot/right edges are discarded)
                verbose = config['verbose']
            )
        
            # if we specify a tileset to save then save the tiles. 
            # otherwise if the source tiles are Tif, the tileset would already contain the required JPEG tiles 
            if ts is not None:
                save_tile(
                    array=tile_array,              #tile array
                    save_directory=tilesDir,        #folder location to save tile in 
                    tile_name=tile_name,            #chips will be named as: region_tile_{tileXpos}_{tileYpos}
                    profile = tile_meta,                 #rasterio requires a 'profile'(meta data) to create/write files
                    save_fmt = target_tile_file_format,   #format to save tiles as (ex: png,jpeg,geoTiff) 
                    verbose = config['verbose']
                )
            
            ## Save the Image Chips
            save_chips(
                array=split_array,                                     #split tile containing chips
                save_directory=chipsDir,                             #folder location to save chips in 
                tile_name=tile_name,                                   #chips will be named as: tileName_tileXpos_tileYpos_chipXpos_chipYpos
                profile = tile_meta,                                   #rasterio requires a 'profile'(meta data) to create/write files
                chip_file_format = chip_file_format,                                   #format to save chips as (ex: png,jpeg,tiff) 
                verbose = config['verbose']
            )

    
    # Summary
    if ts is not None:
        tileFiles = __listdir(directory=tilesDir,
                              extensions=['all']) 
        vbPrint(f"Number of files created in {tilesDir}: {len(tileFiles)}\n{'-'*4}Image Tiles made successfully{'-'*4}")
        
        
    chipFiles = __listdir(directory=chipsDir,
                           extensions=['jpeg', 'tif']) 
                           
    vbPrint(f"Number of files created in {chipsDir}: {len(chipFiles)}\n{'-'*4}Image Chips made successfully{'-'*4}")

def _cropTiles(inputRootPath:str, outputRootPath:str, config:dict):

    # Only JPEG tiles are cropped

    input_image_pattern = '*.' + config['target_tile_file_format'].upper()
    image_filepaths = find_files(inputRootPath, input_image_pattern)

    chip_size = config['chipDimX']
    for image_filepath  in image_filepaths:
        if image_filepath.suffix == '.jpeg' or image_filepath.suffix == '.JPEG':
            print('Cropping image ' + str(image_filepath))
            image = cv2.imread(str(image_filepath), 1)  # Note that cv2 reads the image as BGR
            size_x = (image.shape[1]//chip_size)*chip_size
            size_y = (image.shape[0]//chip_size)*chip_size
            image = Image.fromarray(image)
            image = image.crop((0,0, size_x, size_y))
            image = np.array(image)
            output_image_filepath = os.path.join(outputRootPath,image_filepath.name)
            cv2.imwrite(output_image_filepath, image)

@__time_this  
def genTrainingImgChips():
    """
    Notes: 
        Generates training dataset image tiles and chips. Only tiles that have annotations and they exist as label Tiles will be generated. 
        For those generated tiles, we generate all chips irrespective of the presence of annotations. 
        
    Inputs: 
        imagery root directory
        mWH (int): maximum Width/Height of tiles
        file_extensions (str): what raster file types should be converted to chips 
                        (this is important since image files are associated w/ other file type like an .xml,.cpg,.dbf this makes sure those are ignored)
        chip_file_format (str): file format for chips. Exs: jpeg, png
    
    outputs:
        folder containing chip image-files and any support files. 

    Example:
        python3 pipeline_e2e.py --project_root=project_root --fs=labels --file_extensions=tif --chip_file_format=png --mWH=5000,5000 -genImgChips
    
    """

    ##----------------------------Configuration/Setup---------------------------##
    project_root = config['project_root']                                                           #dataset (root folder)
    fs = config['fs']                                                           #existing fileset (subfolder)
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    window = __getWindow(window_config=config['mWH'])                           #dictonary w/ max window dimensions (default: width:5000,height:5000) 
    
    file_formats = config['file_extensions'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    source_tile_file_format = config['source_tile_file_format']                                                   #file format for tiles
    target_tile_file_format = config['target_tile_file_format']                                                   #file format for tiles
    chip_file_format = config['chip_file_format']                                                   #file format for chips when they are saved. (png,jpeg,tif) https://gdal.org/drivers/raster/index.html

    # The path to imagery 
    trainingImageryRootPath = config['trainingImageryRootPath']                               

    # We need to consult the labelTiles dir as not all imagery was converted to labelTiles - images that didnt have any features annotations were ignored.  
    labelTilesPath = '%s/labelTiles_%s'%(project_root,ts)
    labelTileFiles = __listdir(directory=labelTilesPath, extensions=file_formats)                             #list of files to convert to chips
    
    # The image tiles in the desired format and CRS 
    tilesDir = '%s/imageTiles_%s'%(project_root,ts)
    __make_folders(rootfolder=project_root,
                    subfolder=tilesDir)    
    
    # The image chips
    chipsDir = '%s/imageChips_%s'%(project_root,ts)                                   
    __make_folders(rootfolder=project_root,                                               ##Make folder to hold chips
                   subfolder=chipsDir)                                        #method for making project rootfolder/subfolders
    
    imagery_tiles_filepath=[]
    imagery_tiles = []
    pattern = '*.'+config['input_file_format']
    
    for root, dirs, files in os.walk(trainingImageryRootPath):

        for filename in fnmatch.filter(files, pattern):
        
            imagery_tiles_filepath.append(os.path.join(root, filename))
            imagery_tiles.append(filename)
           
    print('Number of imagery  tiles', len(imagery_tiles))

    # if the tile has 4 bands with band 4 being the Near Infra Red (NIR) band the  say so
    if config['input_image_near_infrared']:
        input_image_near_infrared=True

    # # If the tile is in GeoTIFF format, then convert to 8-bit JPEG 
    # if not target_tile_file_format==source_tile_file_format:
    #     convert_tiles(sourceIsNearInfrared=input_image_near_infrared, sourceTileFormat=source_tile_file_format, targetTileFormat=target_tile_file_format, sourceDir=trainingImageryRootPath, targetDir=tilesDir)

    # # crop image tiles so that they are divisible by the chip size - images are overwritten
    # _cropTiles(inputRootPath=tilesDir, outputRootPath=tilesDir, config=config)


    # list of tiles converted to JPEG (and potentially cropped) that need further split into chips. Only imagery that has a label tile is processed.
    tilesFiles = __listdir(directory=labelTilesPath, extensions=target_tile_file_format) 
    
    # Loop over tiles
    for count, imageName in enumerate(tilesFiles):

        basename_without_ext = os.path.splitext(os.path.basename(imageName))[0]
        filename = basename_without_ext+ '.' + config['input_file_format']
        print(filename)
        try: 
            index = imagery_tiles.index(filename)
        except ValueError:
            continue
        
        corresponding_imagery_tile_file_path = imagery_tiles_filepath[index]

        
        vbPrint(f'Current Image Path: {corresponding_imagery_tile_file_path}')
        
        #Open raster files & apply window to generate chips 
        for tile_array, tile_meta, tile_name in open_raster(
            path_to_files = corresponding_imagery_tile_file_path,      #current tile
            maxWidth = window['width'],       #this is maxWidth/maxHeight
            maxHeight = window['height'],        #actual height/width may be less since no padding happens when opening
            verbose = config['verbose']):
                                                             
            if 0 in tile_array.shape:
                vbPrint(f'empty array generated: {tile_array.shape}. Skipping')
                continue
            
            ## Creation of Image chips
            split_array = split_raster(
                image_array=tile_array,                        #current tile to split into chips
                chipX = 256,                            #chip width
                chipY = 256,                            #chip height
                minScrapPercent = 0,                     #0 means every tile_array will be padded before splitting (otherwise chips along bot/right edges are discarded)
                verbose = config['verbose']
            )

            
            # if we specify a tileset to save then save the tiles. 
            # otherwise if the source tiles are Tif, the tileset would already contain the required JPEG tiles 
            if ts is not None:
                save_tile(
                    array=tile_array,              #tile array
                    save_directory=tilesDir,        #folder location to save tile in 
                    tile_name=tile_name,            #chips will be named as: region_tile_{tileXpos}_{tileYpos}
                    profile = tile_meta,                 #rasterio requires a 'profile'(meta data) to create/write files
                    save_fmt = target_tile_file_format,   #format to save tiles as (ex: png,jpeg,geoTiff) 
                    verbose = config['verbose']
                )
            
            ## Save the Image Chips
            save_chips(
                array=split_array,                                     #split tile containing chips
                save_directory=chipsDir,                             #folder location to save chips in 
                tile_name=tile_name,                                   #chips will be named as: tileName_tileXpos_tileYpos_chipXpos_chipYpos
                profile = tile_meta,                                   #rasterio requires a 'profile'(meta data) to create/write files
                chip_file_format = chip_file_format,                                   #format to save chips as (ex: png,jpeg,tiff) 
                verbose = config['verbose']
            )

    
    # Summary
    if ts is not None:
        tileFiles = __listdir(directory=tilesDir,
                              extensions=['all']) 
        vbPrint(f"Number of files created in {tilesDir}: {len(tileFiles)}\n{'-'*4}Image Tiles made successfully{'-'*4}")
        
        
    chipFiles = __listdir(directory=chipsDir,
                           extensions=['jpeg', 'tif']) 
                           
    vbPrint(f"Number of files created in {chipsDir}: {len(chipFiles)}\n{'-'*4}Image Chips made successfully{'-'*4}")


def genAnnotations():
    """
    Generates the train and val annotations files using the image and label chips
    
    Requirements:
        * label chips should be available in <project_root>/labelChips_<ts>/..
        * While not required by this function, corresponding image chips should be available in <project_root>/imageChips_<ts>/..

    Generates:
        * Two annotation files to be used by the model to train:
            1. <project_root>/annotations_<ts>/annotations_train.json
            2. <project_root>/annotations_<ts>/annotations_val.json

    """
    project_root = config['project_root']
    ts = config['ts']

    # We support the case where both train and val label chips are in one directory and they need to be split. 
    # The train to validation ratio controls the split. If tvRatio is 1.0 then either train or validation annotations are produced.
    tvRatio = config['tvRatio']

    labelChipsDir = config['labelChipsDir']

    annDir = '%s/annotations_%s'%(project_root,ts)

    ## Making the dirs
    if(os.path.isdir(annDir)):
        vbPrint('Found dir: %s'%(annDir))
    else:
        vbPrint('Making dir: %s'%(annDir))
        os.mkdir(annDir)

    train_or_val = Path(labelChipsDir).name
    
    # all the chips under labelChipsDir are either train (1.0) or val (0.0) 
    if (tvRatio==1.0 or tvRatio==0.0):
        
        ## Getting the label chips
        vbPrint('Reading `%s` for label chips'%(labelChipsDir))
        labels = __listdir(labelChipsDir, ['tif', 'jpeg'])
        vbPrint('Data  : %i'%(len(labels)))
        data = np.asarray(labels)
        
        # Generate Annotations
        vbPrint('Generating annotations for ' + train_or_val )
        dataR2C = Raster2Coco(data, labelChipsDir,  has_gt=True)
        dataJSON = dataR2C.createJSON()
        dataJSON = dataR2C.genDirAnnotations(band_number=1)
        
        filepath = '%s/annotations_%s.json'%(annDir,train_or_val)

        with open(filepath, 'w') as outfile:
            json.dump(dataJSON, outfile)
    
    # the chips under labelChipsDir need to be split into train and val
    elif (tvRatio > 0.0 or tvRatio < 1.0):

        ## Getting the label chips
        vbPrint('Reading `%s` for label chips'%(labelChipsDir))
        labels = __listdir(labelChipsDir, ['tif', 'jpeg'])
        vbPrint('Dataset Size   : %i'%(len(labels)))
        labels = np.asarray(labels)

        # permute the chips 
        np.random.shuffle(labels)
        
        # split
        splitIdx = int(labels.shape[0]*tvRatio)
        trainData,valData = np.split(labels,[splitIdx])
        
        vbPrint('Training Data  : %i'%(len(trainData)))
        vbPrint('Val Data       : %i'%(len(valData)))
        
        # Generate Annotations
        vbPrint('Generating annotations for the training data')
        trainR2C = Raster2Coco(trainData, labelChipsDir,  has_gt=True)
        trainJSON = trainR2C.createJSON()
        trainJSON = trainR2C.genDirAnnotations()

        with open('%s/annotations_train.json'%(annDir), 'w') as outfile:
            json.dump(trainJSON, outfile)
        
        del trainJSON
        del trainR2C

        vbPrint('Generating annotations for the validation data')
        valR2C = Raster2Coco(valData, labelChipsDir, has_gt=True)
        valJSON = valR2C.createJSON()
        valJSON = valR2C.genDirAnnotations()

        with open('%s/annotations_val.json'%(annDir), 'w') as outfile:
            json.dump(valJSON, outfile)
        
        del valJSON
        del valR2C

    vbPrint('Annotations made and saved successfully')


def genImageInfo():
    """
    Generates the COCO-test dataset compatible image info files
    
    Requirements:
        * Corresponding image chips should be available in <project_root>/imageChips_<ts>/..

    Generates:
        * COCO compatible JSON image info file
            <project_root>/annotations_<ts>/image-info-test.json
            
    """
    project_root = config['project_root']
    ts = config['ts']
    has_gt=config['has_gt']

    chipsDir = config['imageChipsDir']
    annDir = '%s/annotations_%s'%(project_root,ts)

    ## Making the dirs
    if(os.path.isdir(annDir)):
        vbPrint('Found dir: %s'%(annDir))
    else:
        vbPrint('Making dir: %s'%(annDir))
        os.mkdir(annDir)

    ## Getting the labels and splitting into training and validation images
    vbPrint('Reading `%s` for chips'%(chipsDir))
    chips  = __listdir(chipsDir, ['jpeg'])
    vbPrint('Dataset Size   : %i'%(len(chips)))
    chips = np.asarray(chips)
    
    # Generate image info - test data do not have ground truth
    vbPrint('Generating image info for the test data')
    testR2C = Raster2Coco(chips, chipsDir, has_gt=False)
    testJSON = testR2C.createJSON()

    with open('%s/image_info_test.json'%(annDir), 'w') as outfile:
        json.dump(testJSON, outfile)
    
    del testJSON
    del testR2C

    
    vbPrint('Test Image Info JSON made and saved successfully')


#----------------------------------------------------------------#
#--------- Model Evaluation and Verification Functions ----------#
#----------------------------------------------------------------#
def genInferenceEvalJSON():
    """
    Uses shell commands to run the eval.py for the configured parameters of the pipeline
    This generates inferences in a JSON file saved in <project_root>/inferencesJSON_<ts>/
    """
    project_root = config['project_root']
    ts = config['ts']

    detPath = '%s/inferencesJSON_%s/'%(project_root,ts)
    if(os.path.isdir(detPath)):
        vbPrint('Found dir: %s'%(detPath))
    else:
        vbPrint('Making dir: %s'%(detPath))
        os.mkdir(detPath)

    shCmd = 'python segmentation/models/yolact-original/eval.py --trained_model="%s" \
        --config=%s  --web_det_path="%s" \
        --score_threshold=%f --top_k=15  --output_web_json'%(
            config['trained_model'],
            config['config'],
            detPath,
            config['score_threshold']
        )

    if(config['verbose']==False):
        shCmd += ' --no_bar'

    vbPrint('Initializing Inferences')
    os.system(shCmd)
    vbPrint('Completed')

def genInferencePredictJSON():
    """
    Uses shell commands to run the predict.py for the configured parameters of the pipeline
    This generates inferences in a JSON file saved in <project_root>/inferencesJSON_<ts>/
    """
    project_root = config['project_root']
    ts = config['ts']

    detPath = '%s/inferencesJSON_%s/'%(project_root,ts)
    if(os.path.isdir(detPath)):
        vbPrint('Found dir: %s'%(detPath))
    else:
        vbPrint('Making dir: %s'%(detPath))
        os.mkdir(detPath)

    shCmd = 'python segmentation/models/yolact-original/predict.py --trained_model="%s" \
        --config=%s  --web_det_path="%s" \
        --score_threshold=%f --top_k=15  --output_web_json'%(
            config['trained_model'],
            config['config'],
            detPath,
            config['score_threshold']
        )

    if(config['verbose']==False):
        shCmd += ' --no_bar'

    vbPrint('Initializing Predictions')
    os.system(shCmd)
    vbPrint('Predictions Completed')

def genInferenceTileGeoJSON():
    """
    genTileGeoJSON Summary:
        Generates up to 2 types of inference data from the inferences JSON output which is generated by genEvalJSON():
        * Inference GeoJSON: a single geoJSON that has all the inferences as vector polygons. The file is a .geojson in <project_root>/inferenceGeoJSON_<ts>
        * Inference Tiles: Merged inferences for each of the 5000x5000 tiles, each tile is available as a .png images in <project_root>/inferenceTiles_<ts>
        Requirements
        * The required annotation file must be present in <project_root>/annotations_<ts>
        * The inferences JSON generated by the model must be present in <project_root>/inferencesJSON_<ts>/..
    
    genTileGeoJSON Algorithm:
        * The detections are available based on image_id. This id does not correspond to any tile or chip identification directly.
        * The annotations file has, for each image_id, the associated tile number and chip coordinates encoded into the image name.
        * We first read the entire annotations file and create a HashMap that stores, for each tile: a list of all chips' information.
        * The information for a chip includes its image_id, its tile number, and its chip co-ordinates
        * Once this hashMap is prepared, we are ready to iterate on a tile by tile basis:
            * For each tile, we have the chip info for all chips. For each chip:
                * We have the tile number which is the 'tile' itself
                * We have raw image present with the name 'tile_<chip Co-ordinate X>_<chip Co-ordinate Y>'
                * We have the associated chip detections with the 'image_id' in the inferences JSON
            * Essentially: We have a mapping between the tile and image_iproject_root for all chips in the tile, for all tiles.
            * For each chip:
                * We have the detection vectors. This is in a COCO vector format and needs to be rasterized.
                * For the 256x256 chip, we have many detections each with a score.
                * We generate an empty image 256x256 image filled with 0s.
                * For each detection, we rasterize the detection and add it to the raster image if it is above the configurable threshold.
                * Sidewalk pixels are marked 255. If multiple detections mark the same pixel as a sidewalk, it is still 255.
                * 255 was chosen instead of 1 for the sake of simplicity: If viewed as an image, this raster shows the sidewalk in white
                  with a black background very clearly.
            * Essentially: For each 'image_id' (i.e. for each chip), we now have a rasterized inference.
            * For each tile:
                * We  go through  all of chips, pull their rasters from the HashMap.
                * Since the chip information includes the co-ordinates of a chip in a tile, we know the indexes of the 256x256 block in the 5120x5120 tile.
                * Using this info, we generate a single 5120x5120 raster for each tile. This is then cropped to 5000x500px.
                * Generation of the GeoJSON:
                    * Since we know the tile i.e. the path and name of the tile image, we can infer the path and name of the respective world file
                    * The raster is then converted into a number of polygon vectors
                    * Each detected polygon of >3 points, it is appended to the geoJSON file.
        * Once all the tiles are processed, the appropriate line endings are appended to the geoJSON.

    genTileGeoJSON Notes:
        * The saving images can be skipped using the argument '--genImgs=0'
        * The generation of geoJSON can be skipped with the argument '--genGeoJSON=0'
    """
    project_root = config['project_root']
    ts = config['ts']
    tilesDir = '%s/imageTiles_%s'%(project_root,ts)
    inferenceTilesDir = '%s/inferenceTiles_%s'%(project_root,ts)
    inferenceGeoJSONDir = '%s/inferenceGeoJSON_%s'%(project_root,ts)
    geoJSONFilename = os.path.basename('%s_%s.geojson'%(project_root,ts))
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

    detFilePath = '%s'%(config['infJSON'])
    annFilePath = '%s'%(config['annJSON'])

    #annFilePath = '%s/inference_annotations/%s'%(project_root,config['annJSON']) # For testing. Can be reverted once the data-pipeline part is completed.

    if(genGeoJSON):
        # Initialize GeoJSONFile
        with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'w+') as geoJSONf:
            geoJSONf.write('{"type": "FeatureCollection","name": "sidewalk_detections_%s_%s","features": ['%(project_root,ts))

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
    
    chipShape = (config['chipDimX'],config['chipDimY'])
    infTileShape = (chipShape[0]*config['colsSplitPerTile'],
                    chipShape[1]*config['rowsSplitPerTile'])

    vbPrint('Expected Chip dimensions: %ix%i'%(chipShape[0],chipShape[1]))
    vbPrint('Expected Tiles dimensions: %ix%i'%(infTileShape[0],infTileShape[1]))

    # Creating a hashmap for the image dets with the image IDs as keys
    # Each prediction is a merged image of many detections
    # Only detections above the threshold are selected to be merged
    vbPrint('Generating det masks with a threshold of %0.2f'%(config['det_score_threshold']))
    infChipMap = dict()
    i=0
    n=len(infData)
    peFactor = config['pef']
    
    for img in infData:
        i += 1
        pred = np.zeros(chipShape)
        for det in img['dets']:
            if(det['score']>config['det_score_threshold']):
                detMask = mask.decode(det['mask'])
                pred[detMask==1]=255
        
        infChipMap[img['image_id']] = pred

        # Prints progress at every 'pef' factor of completion
        # PEF stanproject_root for 'Print-Every factor'
        facComp = (i+1)/(n)
        if(facComp >= peFactor):
            vbPrint('Generating detection masks: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
            peFactor += config['pef']

    del infData
    
    vbPrint('Detection Masks made')
    vbPrint('Generating inference tiles from detection masks')

    # Rows and columns a tile is split into.
    # Default is 20,20
    num_chips_per_tile_rows = config['rowsSplitPerTile']
    num_chips_per_tile_cols = config['colsSplitPerTile']
    expectedChips = num_chips_per_tile_rows*num_chips_per_tile_cols

    projectionTransformer = pyproj.Transformer.from_crs("epsg:32111","epsg:4326",always_xy=False)
    #projectionTransformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:6565", always_xy=False)

    i = 0
    sidewalkCount = 0
    n = len(tileMap.keys())
    peFactor = config['pef']

    """
    For each tile, gets the detection masks and merges them into one image.
    Chips that don't have any detections are skipped.
    """
    for i,tile in enumerate(tileMap.keys()):
        # Skipping tile if unexpected number of chips exist for the tile in the annotations JSON
        if(len(tileMap[tile])!=expectedChips):
            vbPrint('Found %i chips for tile %s. Skipping'%(len(tileMap[tile]), tile))
        else:
            # The inference tile. Initialized with 0. Inference chips are overlayed at their respective locations.
            infTile = np.zeros((num_chips_per_tile_rows*chipShape[0],num_chips_per_tile_cols*chipShape[1]), dtype=np.uint8)
            print(tile)
            for chipID,row,col in tileMap[tile]:
                print(chipID, row, col)
                
                #print('(%i, %i)'%(row, col))
                # Overlays the chip mask at the correct location in infTile
                # Skip any chips with no inference. It is assumed that no inference data is generated for chips with 0 detections.
                if(chipID in infChipMap):
                    # Cropshift is the extra pixels to the left and top in the final row and column of the image inference
                    # This is only done due to the way the year-1 project was 'padded'.
                    # This will be need to be removed when correct padding is applied.
                    cropShiftX = 0
                    cropShiftY = 0
                    if(row==num_chips_per_tile_rows):
                        cropShiftX = 120
                    if(col==num_chips_per_tile_cols):
                        cropShiftY = 120
                    #infTile[(row-1)*chipShape[0]:((row)*chipShape[0])-cropShiftX, (col-1)*chipShape[1]:((col)*chipShape[1])-cropShiftY] = infChipMap[chipID][cropShiftX:,cropShiftY:]
                    infTile[(row)*chipShape[0]:((row+1)*chipShape[0])-cropShiftX, (col)*chipShape[1]:((col+1)*chipShape[1])-cropShiftY] = infChipMap[chipID][cropShiftX:,cropShiftY:]
            
            # Cropping image into the final tile dimension
            infTile = infTile[:config['inferenceTileDimX'],:config['inferenceTileDimY']]

            # Getting affine transform parameters from the world file
            tile_exists = exists(os.path.join(tilesDir,tile+'.JPEG'))
            tile_world_file_exists = exists(os.path.join(tilesDir,tile+'.JGw'))
            
            if tile_exists and tile_world_file_exists:
                # Getting affine transform parameters from the world file
                with open('%s/%s.JGw'%(tilesDir,tile)) as worldFile:
                    world_file_rows = worldFile.read().split('\n')
                worldFile.close()

                A = float(world_file_rows[0])
                D = float(world_file_rows[1])
                B = float(world_file_rows[2])
                E = float(world_file_rows[3])
                C = float(world_file_rows[4])
                F = float(world_file_rows[5])

                if(genImgs):
                    # Writing the inference image and the world tile
                    # File format kept as png because it has lossless compression and to work well with rasterio if needed.
                    cv2.imwrite('%s/%s.png'%(inferenceTilesDir,tile),infTile)
                    # write the pgw world file for the png inference images
                    with open('%s/%s.pgw'%(inferenceTilesDir,tile), 'w') as pgwFile:
                        for i in range(len(world_file_rows)):
                            pgwFile.writelines(world_file_rows[i]+'\n')
                    pgwFile.close()

                if(genGeoJSON):
                    vbPrint('Generating GeoJSON for tile %s'%tile)
                    with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'a+') as geoJSONf:

                        converter = Raster2Coco(None,None, config['has_gt'])
                        binMask = np.zeros_like(infTile)
                        binMask[infTile==255]=1
                        vectors = converter.binaryMask2Polygon(binMask)
                        JSONRows = ""
                        
                        for sidewalk in vectors:
                            # Skipping any triangles
                            if(len(sidewalk) >= 4):
                                sidewalkCount += 1
                                # Applying affine transform
                                
                                # print(('-'*10)+'\nSidewalk pixels vector')
                                # print(sidewalk)
                                
                                # print(('-'*10)+'\nUTM Vector')
                                # print(['[%f,%f]'%(
                                #         ((A*x) + (B*y) + C,
                                #         (D*x) + (E*y) + F)
                                # ) for y,x in sidewalk])
                            
                                vecLi = ['[%f,%f]'%(
                                        ((A*x) + (B*y) + C,
                                        (D*x) + (E*y) + F)
                                ) for x,y in sidewalk]

                                # vecLi = ['[%f,%f]'%(
                                #     projectionTransformer.transform(
                                #         ((A*x) + (B*y) + C),
                                #         ((D*x) + (E*y) + F)
                                #     )[::-1]
                                # ) for x,y in sidewalk]
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
                            
                        if(JSONRows != ""):
                            geoJSONf.write(JSONRows)
                        
                    # Prints progress at every 'pef' factor of completion
                    # PEF stands for 'Print-Every factor'
                    facComp = (i+1)/(n)
                    if(facComp >= peFactor):
                        vbPrint('Generating inferences tile: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
                        peFactor += config['pef']

    with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'a+') as geoJSONf:
        geoJSONf.write('\n]}')
    geoJSONf.close()

    if(genGeoJSON and genImgs):
        vbPrint('Inference GeoJSON and Tiles generated Successfully')
    elif(genGeoJSON):
        vbPrint('Inference GeoJSON generated Successfully')
    elif(genImgs):
        vbPrint('Inference Tiles generated Successfully')
    else:
        vbPrint('Inference Data ran successfully. Nothing saved due to arguments passed.')

def genTrainingTileGeoJSONs():
    """
    Partition the area features to per-tile features. Generates geoJSON per tile from a large geoJSON input features file that corresponds to a region (eg DVRPC) and a tile index geoJSON file that defines the extents of the tiles. 

      Inputs: 
        project_root (str): dataset folder. aka root folder for project
        fs (str): fileset folder for the input geoJSON
        
    outputs:
        folder containing per tile geoJSON

    Example:
        python3 pipeline_e2e.py --project_root=project_root --fs=input-tile-dir  -genLabelTiles
    
    """

    ##----------------------------Configuration/Setup---------------------------##
    project_root = config['project_root']                                                           #dataset (root folder)
    fs = config['fs']                                                           #existing fileset (subfolder)
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    labelTilesDir = '%s/labelTiles_%s'%(project_root,ts)                                           
    dirExists = os.path.isfile(labelTilesDir) 
    if not dirExists: 
        __make_folders(rootfolder=project_root,
                subfolder=labelTilesDir)    
        print('You need to have the area and tile index geojson in ' + labelTilesDir)

    tileFeoJSONsDir = '%s/tileGeoJSONs_%s'%(project_root,ts)
    __make_folders(rootfolder=project_root,
                    subfolder=tileFeoJSONsDir)    
    
    geoJSONRootPathFile = config['featuresGeoJSONRootPathFile']
    
    tileindexRootPathFile = config['tileIndexPathFile']
    
    tile_geojson_target_crs = config['tile_geojson_target_crs']
    sidewalk_annotation_threshold = config['sidewalk_annotation_threshold']

    # Partition the whole area feature geoJSON to per-tile feature geoJSONs
    geojson_tiles = _partitionGeoJSON(geoJSONRootPathFile, tileindexRootPathFile,
        tileFeoJSONsDir, target_crs=tile_geojson_target_crs, sidewalk_annotation_threshold=sidewalk_annotation_threshold)

    return geojson_tiles

def genLabelTiles():
    """
    Generates one label tile from each per-tile geoJSON input file

      Inputs: 
        project_root (str): dataset folder. aka root folder for project
        fs (str): fileset folder for the input tiles
        label_file_format (str): file format for label tiles. Exs: jpeg, png
        buffer_value of the output vector features. Useful in a variety of use cases such as sidewalks, road, where the line that traces the feature is buffered to cover the whole width of the sidewalk, road as shown in the imagery etc. It is given by 
        an integer number that 
        
    outputs:
        folder containing label tiles
    

    Example:
        python3 pipeline_e2e.py --project_root=project_root --fs=input-tile-dir  -genLabelTiles
    
    """

    ##----------------------------Configuration/Setup---------------------------##
    project_root = config['project_root']                                                           #dataset (root folder)
    fs = config['fs']                                                           #existing fileset (subfolder)
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    file_formats = config['file_extensions'].replace(' ','').split(',')        # convert string of file formats to list of formats
    source_tile_file_format = config['source_tile_file_format']              
    target_tile_file_format = config['target_tile_file_format']              
    #chip_file_format = config['chip_file_format']             # file format for label chips when they are saved. (png,jpeg,tif) https://gdal.org/drivers/raster/index.html

    buffer_value = config['buffer_value']

    tileGeoJSONsDir = '%s/tileGeoJSONs_%s'%(project_root,ts)
    geoJSONFiles = __listdir(directory=tileGeoJSONsDir, extensions=['geojson'])                             #list of files to convert to chips
    vbPrint('%i geoJSON Tile files found in %s'%(len(geoJSONFiles),tileGeoJSONsDir))           #display the number of files to be processed

    labelTilesDir = '%s/labelTiles_%s'%(project_root,ts)                                           #directory with files to convert to chips 
    __make_folders(rootfolder=project_root,                                               ##Make folder to hold chips
                   subfolder=labelTilesDir)                                        #method for making project rootfolder/subfolders

    trainingImageryRootPath=config['trainingImageryRootPath']
    
    target_label_raster_crs =  config['tile_geojson_target_crs']

    # Create label tiles 
    # First create a list of the imagery tiles that will act as template tiles https://rasterio.readthedocs.io/en/latest/cli.html?highlight=geojson#rasterize from 
    # which we will borrow the resolution, crs and other metadata.

    imagery_tiles_filepath=[]
    imagery_tiles = []
    pattern = '*.'+config['input_file_format']
    for root, dirs, files in os.walk(trainingImageryRootPath):
        for filename in fnmatch.filter(files, pattern):
            imagery_tiles_filepath.append(os.path.join(root, filename))
            imagery_tiles.append(filename)

    print('Number of imagery geoTIFF tiles', len(imagery_tiles))

    # print(features_gdf.head())
    # print(tile_index_gdf.head())

   
    for t in geoJSONFiles:

        geojson_filepath = os.path.join(tileGeoJSONsDir, t)

        # Incorporate properties to GeoJSON tile such as the corresponding imagery raster tile size that is needed for creating the label rasters
        basename_without_ext = os.path.splitext(os.path.basename(geojson_filepath))[0]
        filename = basename_without_ext+ '.' + config['input_file_format']
        print(filename)
        try: 
            index = imagery_tiles.index(filename)
        except ValueError:
            continue
        corresponding_imagery_tile_file_path = imagery_tiles_filepath[index]
        dataset = rio.open(corresponding_imagery_tile_file_path)

        # Read in vector
        vector = gpd.read_file(geojson_filepath)

         # Loop over GeoJSON features and add the new properties
        vector = vector.assign(height=dataset.height)
        vector = vector.assign(width=dataset.width)

        vector = vector.set_crs(target_label_raster_crs, allow_override=True)

        # Buffer the features
        buffered_vector = vector.copy()
        buffered_vector.geometry = vector['geometry'].buffer(int(buffer_value))
        
        # overwrite the per tile geoJSON with buffered geometries
        buffered_vector.to_file(geojson_filepath, driver = 'GeoJSON')

        # path to store the label tiles - they are stored in the region dir 
        label_tile_filepath = os.path.join(labelTilesDir, PurePath(geojson_filepath).stem + '.tif') 
        
        # subprocess.run(['gdal_rasterize', '-ts', str(tile_dim_x), str(tile_dim_y), '-a_nodata', '0', '-burn', '255', '-co', 'COMPRESS=LZW', geojson_filepath, label_tile_filepath],
        #     env={'PROJ_LIB':'/opt/conda/envs/sidewalk-env/share/proj'})

        ## Example command line gdal_rasterize -ts 7000 7000 -te 520799.99991914 4420500.00002059 522899.99991914 4422600.00002059 -burn 255 /data/pipeline-runs/project_root4/tileGeoJSONs_ts4/UTM_X59_Y56.geojson /data/pipeline-runs/project_root4/labelTiles_ts4/UTM_X59_Y56.tif

        subprocess.run(['gdal_rasterize', '-ts', str(vector['width'][0]), str(vector['height'][0]), '-te', 
            str(dataset.bounds.left), str(dataset.bounds.bottom), str(dataset.bounds.right), str(dataset.bounds.top), 
            '-burn', '255', '-co', 'COMPRESS=LZW', geojson_filepath, label_tile_filepath],
            env={'PROJ_LIB':'/opt/conda/envs/sidewalk-env/lib/python3.7/site-packages/pyproj/proj_dir/share/proj'})
       

@__time_this  
def genLabelChips():
    """
    Notes: 
        Generates image chips by splitting the tiles
        Prerequisite: loadFiles() has been run or tiles are present in the data/<dataset>/tiles directory
    
    Inputs: 
        project_root (str): dataset folder. aka root folder for project
        fs (str): fileset folder. aka subfolder to hold chips
        mWH (int): maximum Width/Height of tiles
        file_extensions (str): what raster file types should be converted to chips 
                        (this is important since image files are associated w/ other file type like an .xml,.cpg,.dbf this makes sure those are ignored)
        chip_file_format (str): file format for chips. Exs: jpeg, png
    
    outputs:
        folder containing chip image-files and any support files. 

    Example:
        python3 pipeline_e2e.py --project_root=project_root --fs=labels --file_extensions=tif --chip_file_format=png --mWH=5000,5000 -genImgChips
    
    """

    ##----------------------------Configuration/Setup---------------------------##
    project_root = config['project_root']                                                           #dataset (root folder)
    fs = config['fs']           
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    window = __getWindow(
        window_config=str(config['trainingTileDimX']) + ',' + str(config['trainingTileDimY'])
    )                           #dictonary w/ max window dimensions (for training with DVRPC data its width:7000,height:7000) 
    
    file_formats = config['file_extensions'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    source_tile_file_format = config['source_tile_file_format']                                                   #file format for tiles
    target_tile_file_format = config['target_tile_file_format']                                                   #file format for tiles
    chip_file_format = config['chip_file_format']                                                   #file format for chips when they are saved. (png,jpeg,tif) https://gdal.org/drivers/raster/index.html
    
    labelTilesDir = '%s/labelTiles_%s'%(project_root,ts)                                           #directory with files to convert to chips 
                            
    chipsDir = '%s/labelChips_%s'%(project_root,ts)                                   #subfolder for saving chips
    __make_folders(rootfolder=project_root,                                               ##Make folder to hold chips
                   subfolder=chipsDir)                                        #method for making project rootfolder/subfolders
    
    chipDimX = config['chipDimX']
    chipDimY = config['chipDimY']
    chipWindowStride = config['chipWindowStride']

    sidewalk_annotation_threshold = config['sidewalk_annotation_threshold']
    
    # If the tile is in GeoTIFF format, then convert to 8-bit JPEG 
    # if not target_tile_file_format==source_tile_file_format:
    #     convert_tiles(sourceIsNearInfrared=False, sourceTileFormat=source_tile_file_format, targetTileFormat=target_tile_file_format, sourceDir=labelTilesDir, targetDir=labelTilesDir)

    # crop label tiles such that they are divisible by the chip size
    # _cropTiles(inputRootPath=labelTilesDir, outputRootPath=labelTilesDir, config=config)


    # list of tiles converted to JPEG that need further split into chips
    label_tile_filenames = __listdir(directory=labelTilesDir, extensions=target_tile_file_format)  
    label_tile_filenames = np.array(label_tile_filenames)
    
    # to scale the label chip
    # min_max_scaler = MinMaxScaler()
    
    raster_coco = Raster2Coco(label_tile_filenames, chipsDir,  has_gt=True)
    labelCOCOJSON = raster_coco.createJSON()

    annotation_index = 1

    annDir = '%s/annotations_%s'%(project_root,ts)

    ## Making the dirs
    if(os.path.isdir(annDir)):
        vbPrint('Found dir: %s'%(annDir))
    else:
        vbPrint('Making dir: %s'%(annDir))
        os.mkdir(annDir)

    #train_or_val = Path(labelChipsDir).name
    

    # Loop over tiles
    chip_index = 0
    for tile_filename in label_tile_filenames:
        print('Annotating tile ', tile_filename)

        tile_filepath = Path(os.path.join(labelTilesDir), tile_filename )
        
        #Open raster files & apply window to generate chips 
        # for tile_array, tile_meta, tile_name in open_raster(
        #     path_to_files = image_path,      #current tile
        #     maxWidth = window['width'],       #this is maxWidth/maxHeight
        #     maxHeight = window['height'],        #actual height/width may be less since no padding happens when opening
        #     verbose = config['verbose']):
                                                             
        #     if 0 in tile_array.shape:
        #         vbPrint(f'empty array generated: {tile_array.shape}. Skipping')
        #         continue

         # ## Creation of Label chips
            # split_array = split_raster(
            #     image_array=tile_array,                        #current tile to split into chips
            #     chipX = chipDimX,                            #chip width
            #     chipY = chipDimY,                            #chip height
            #     minScrapPercent = 0,                     #0 means every tile_array will be padded before splitting (otherwise chips along bot/right edges are discarded)
            #     verbose = config['verbose']
            # )
        
            # if we specify a tileset to save then save the tiles. 
            # otherwise if the source tiles are Tif, the tileset would already contain the required JPEG tiles 
            # if ts is not None:
            #     save_tile(
            #         array=tile_array,              #tile array
            #         save_directory=labelTilesDir,        #folder location to save tile in 
            #         tile_name=tile_name,            #chips will be named as: region_tile_{tileXpos}_{tileYpos}
            #         profile = tile_meta,                 #rasterio requires a 'profile'(meta data) to create/write files
            #         save_fmt = target_tile_file_format,   #format to save tiles as (ex: png,jpeg,geoTiff) 
            #         verbose = config['verbose']
            #     )
            
            ## Save the Label Chips
            # save_chips(
            #     array=split_array,                                     #split tile containing chips
            #     save_directory=chipsDir,                             #folder location to save chips in 
            #     tile_name=tile_name,                                   #chips will be named as: tileName_tileXpos_tileYpos_chipXpos_chipYpos
            #     profile = tile_meta,                                   #rasterio requires a 'profile'(meta data) to create/write files
            #     chip_file_format = chip_file_format,                                   #format to save chips as (ex: png,jpeg,tiff) 
            #     verbose = config['verbose']
            # )
    
        
        # keep the chips of a single tile in memory 
        label_chips=[]

        if tile_filepath.suffix == '.JPEG':
            
            # open file and use cv2 to create a black and white image as we dont need the three channels. 
            # Please note that in semantic segmentation in general we may have more than two classes (sidewalk and background) 
            # and in that case we would not convert to black and white the label tiles. 
            label_tile = cv2.imread(str(tile_filepath),1)
            label_tile = cv2.cvtColor(label_tile, cv2.COLOR_BGR2GRAY)
            (thresh, label_tile) = cv2.threshold(label_tile, 127, 255, cv2.THRESH_BINARY) # returned thresh is not used

            # split the tile into chips 
            label_tile = label_tile.reshape(label_tile.shape[0], label_tile.shape[1], 1)
            label_chips_array = patchify(label_tile, (chipDimY, chipDimY, 1),  step=chipWindowStride)

            for i in range(label_chips_array.shape[0]):
                for j in range(label_chips_array.shape[1]):
                    
                    chip_index += 1

                    label_chip_array = label_chips_array[i, j, :, :]

                    # scaled_label_chip_array = min_max_scaler.fit_transform(label_chip_array.reshape(-1, label_chip_array.shape[-1])).reshape(label_chip_array.shape)

                    label_chip_array = label_chip_array[0] #Drop the extra unecessary dimension that patchify adds.                               
                    label_chips.append(label_chip_array)

                    filename = tile_filepath.stem + '_' + str(i) + '_' + str(j) + str(tile_filepath.suffix) 
                    annotation_index  += 10000 * chip_index

                    raster_coco.genImgJSON(np.squeeze(label_chip_array), filename, 
                        chip_index, band_no=1, 
                        annotation_idx=annotation_index, 
                        annotation_threshold=sidewalk_annotation_threshold)
        
        # For testing on few tiles only
        # if  tile_filepath.name == 'UTM_X24_Y72.JPEG':
        #     break
        # label_chips_array = np.array(label_chips)

    #filepath = '%s/annotations_%s.json'%(annDir, train_or_val)
    annotation_filepath = '%s/annotations.json'%(annDir)

    with open(annotation_filepath, 'w') as outfile:
        json.dump(labelCOCOJSON, outfile)
    #label_tiles_array = np.array(label_tiles)

    # Summary
    if ts is not None:
        # tileFiles = __listdir(directory=tilesDir,
        #                       extensions=['all']) 
        # vbPrint(f"Number of files created in {tilesDir}: {len(tileFiles)}\n{'-'*4}Image Tiles made successfully{'-'*4}")
        
        
        chipFiles = __listdir(directory=chipsDir,
            extensions=['jpeg', 'tif']) 
                           
    vbPrint(f"Number of files created in {chipsDir}: {len(chipFiles)}\n{'-'*4}Image Chips made successfully{'-'*4}")



if __name__ == '__main__':
    """
    Argument Formats:
        --<KEY>=<VALUE>     -> Used to set config
        -<PipelineStep>     -> Used to set pipeline Steps

    Base Arguments:
        --project_root                -> [dataset name] The datset name. This will be used to create the file directory and refer to the dataset.
        --ts                -> [tileset name] Name of the tileset. This will be used to create the unique tiles and images folder.
                                    Separate tile and image folders are needed to separate training data from inference data.
                                    This also allows speration of inference set. Ex of values for ts: train,test, XYZ_County_inference, PQR_Inference, etc.
        -<PipelineStep>     -> [Function Name] name of the function to call in the pipeline step. Multiple can be passed to create the pipeline.

    LoadTiles Arguments:
        --s3td              -> [s3 URL] The s3 URI of the input image tiles. Ex: s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/DOM2015/Selected_500/

    genAnnotations Arguments:
        --tvRatio           -> The train/validation split ratio. For ex: 0.8 means out of 100 images, 80 will be train and 20 will be validation. 

    genEvalJSON Arguments:
        --trained_model     -> [path] Path to the trained model. Ex: weights/DVRPCResNet50_8_88179_interrupt.pth
        --config            -> [config name] name of the config to be used from config.py. Example dvrpc_config
        --score_threshold   -> [float] The score threshold to use for inferences. Example: 0.0

    genTileGeoJSON Arguments:
        --annJSON           -> [path] name of the annotations .JSON file. Example: DVRPC_test.json
        --infJSON           -> [path] name of the created inference .JSON file. Example: DVRPCResNet50.json
        --genGeoJSON        -> [0 or 1] Skips generation of geoJSON if set to 0. Default is 1
        --genImgs           -> [0 or 1] Skips generation of images if set to 0. Default is 1 

    Misc Arguments:
        --verbose           -> [0 or 1] Setting it to 1 allows print statements to run. Default: 1
        --pef               -> [float] (Used for long processes) Determines after how much increase in the factor of completion of a process, a vbPrint is made. 
                                    For example: 0.1 means at every 10% completion, a print is made. (Only works if --verbose is set to 1). Default: 0.1

    Examples:
        Training Pipeline
        * python pipeline_e2e.py --project_root=project_root1 --ts=train -loadTiles --s3td=test -genImgChips -genLabelChips -genAnnotations

        MEVP Pipeline
        * python pipeline_e2e.py --project_root=project_root1 --ts=inf1 -loadTiles --s3td=test -genImgChips -genAnnotations
        * python -genEvalJSON --trained_model=weights/DVRPCResNet50_8_88179_interrupt.pth --config=dvrpc_config --score_threshold=0.0
        * python pipeline_e2e.py --project_root=project_root1 --ts=inf1 -genTileGeoJSON --annJSON=DVRPC_test.json --infJSON=DVRPCResNet50.json
    """
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