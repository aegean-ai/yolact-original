"""
Data Pipeline Functions
"""

from pipeline_helper import __time_this, __make_folders, vbPrint, __getWindow, __listdir
from pipeline_config import *
from s3_helper import *
from split_merge_raster import Open_Raster,Split_Raster,Save_Tile,Save_Patches            #These are all called w/in genImgPatches
import numpy as np
from pycocotools import mask
from raster2coco import Raster2Coco
import json


@__time_this
def loadFiles():
    """
    * This function loads all required files from S3.
    * The files required at this stage of the pipeline are the tiles, labels, and their corresponding world files
    * This code may have to change depending on how the Input data is available. 

    Note:
        loads region data from s3 into the directory:
            * ./data/<dataset>/region_<fileset>/
    
    Inputs:    
            ds (str): local dataset folder (root folder)
            fs (str): local fileset folder (sub folder)
            s3td (str): bucket path to files
            fmts (str): list of file formats to search for in s3Dir, gets converted to list. Ex: "jpg, png" -> ['jpg','png'], "all" -> ["all"] (all returns all files)
    """
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
    
    """
    #Examples:
    python3 dataPipeline.py --ds=dataset7 --fs=inputs --fmts=all --s3d=s3://njtpa.auraison.aegean.ai/DOM2015/Selected_500/pipelineTest/ -loadFiles
    
    python3 dataPipeline.py --ds=dataset7 --fs=labels --fmts=all --s3d=s3://njtpa.auraison.aegean.ai/labels_ground_truth/year-2/output/ -loadFiles
    
    python3 dataPipeline.py --ds=dataset7 --fs=vectors --fmts=geojson --s3d=s3://njtpa.auraisonaegean.ai/labels_ground_truth/year-2/vector_files/ -loadFiles 
    """

    
@__time_this  
def genImgPatches():
    """
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
    """

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
        vbPrint(f"Number of files created in {tilesDir}: {len(tileFiles)}\n{'-'*4}Image Tiles made successfully{'-'*4}")
        
        
    patchFiles = __listdir(directory=patchesDir,
                           extensions=['all']) 
                           
    vbPrint(f"Number of files created in {patchesDir}: {len(patchFiles)}\n{'-'*4}Image Patches made successfully{'-'*4}")
    
    """
    Example:
        * python3 dataPipeline.py --ds=dataset5 --fs=labels --fmts=tif --sfmt=png --mWH=5000,5000 -genImgPatches
        * python3 dataPipeline.py --ds=dataset5 --fs=inputs --fmts=jpg --sfmt=png --mWH=5000,5000 -genImgPatches
        * python3 dataPipeline.py --ds=dataset7 --fs=labels --ts=labels --fmts=tif --sfmt=gtiff --mWH=5000,5000 -genImgPatches
        * python3 dataPipeline.py --ds=dataset7 --fs=labels --ts=labels --fmts=tif --sfmt=tiff --mWH=5000,5000 -genImgPatches
        * python3 dataPipeline.py --ds=dataset8 --fs=labels --ts=labels --fmts=gtiff --sfmt=gtiff --mWH=5000,5000 -genImgPatches
        * python3 dataPipeline.py --ds=dataset9 --fs=labels --ts=labels9 --fmts=gtiff --sfmt=gtiff --mWH=5000,5000 -genImgPatches
    """

def genAnnotations():
    """
    Generates the train and test annotations files using the image and label patches
    
    Requirements:
        * label patches should be available in <ds>/labelPatches_<ts>/..
        * While not required by this function, corresponding image patches should be available in <ds>/imagePatches_<ts>/..

    Generates:
        * Two annotation files to be used by the model to train:
            1. <ds>/annotations_<ts>/annotations_train.json
            2. <ds>/annotations_<ts>/annotations_test.json

    """
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

    vbPrint('Annotations made and saved successfully')
