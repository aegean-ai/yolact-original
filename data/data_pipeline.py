"""
Data Pipeline Functions
"""

from pipeline_helper import __time_this, __make_folders, vbPrint, __getWindow, __listdir
from pipeline_config import *
from s3_helper import *
from split_merge_raster import open_raster,split_raster,save_tile,save_chips            #These are all called w/in genImgChips
import numpy as np
from pycocotools import mask
from raster2coco import Raster2Coco
import json

import traceback
import sys

from s3_helper import s3_to_local, local_to_s3
from pipeline_helper import *



@__time_this
def loadFiles():
    """
    * This function loads all required files from S3.
    * The files required at this stage of the pipeline are the tiles, labels, and their corresponding world files
    
    Note:
        loads region data from s3 into the directory:
            * ./data/<dataset>/region_<fileset>/
    
    Inputs:    
            ds (str): local dataset folder (root folder)
            fs (str): local fileset folder (sub folder)
            s3td (str): bucket path to files
            fmts (str): list of file formats to search for in s3_uri, gets converted to list. Ex: "jpg, png" -> ['jpg','png'], "all" -> ["all"] (all returns all files)
    """
    ##----------------------------Configuration/Setup---------------------------##
    ds = config['ds']                                                           #dataset string. root folder of files
    fs = config['fs']                                                           #fileset string. folder to hold world files
    s3_uri = config['s3d']                                                       #AWS URI for folder located in S3 bucket 
    
    file_formats = config['fmts'].lower().replace(' ','').split(',')            #convert string of file formats to list of formats
    loadDir = '%s/region_%s'%(ds,fs)                                             #root folder + subfolder
    
    vbPrint('Load File(s) Starting...')
                                                                                
    __make_folders(rootfolder=ds,                                               ## Making the dirs
                   subfolder=loadDir)                                           #method for making project rootfolder/subfolders

    ##----------------------------Load Files------------------------------------##
    
                                                                                ##Retrieve aws bucket object    
    s3_to_local(s3_uri=s3_uri,                                              #S3 Bucket access. Uses boto3 + local aws credentials to access bucket
                     desired_formats=file_formats,
                     load_dir=loadDir,
                     verbose=config['verbose'])
    
    
    vbPrint('Loading Region File(s) Completed')
    
    """
    #Examples:
    python3 data_pipeline.py --ds=dataset7 --fs=inputs --fmts=all --s3d=s3://njtpa.auraison.aegean.ai/DOM2015/Selected_500/pipelineTest/ -loadFiles
    
    python3 data_pipeline.py --ds=dataset7 --fs=labels --fmts=all --s3d=s3://njtpa.auraison.aegean.ai/labels_ground_truth/year-2/output/ -loadFiles
    
    python3 data_pipeline.py --ds=dataset7 --fs=vectors --fmts=geojson --s3d=s3://njtpa.auraisonaegean.ai/labels_ground_truth/year-2/vector_files/ -loadFiles 
    """

    
@__time_this  
def genImgChips():
    """
    Notes: 
        Generates image chips by splitting the tiles
        Prerequisite: loadFiles() has been run or tiles are present in the data/<dataset>/tiles directory
    
    Inputs: 
        ds (str): dataset folder. aka root folder for project
        fs (str): fileset folder. aka subfolder to hold chips
        mWH (int): maximum Width/Height of tiles
        fmts (str): what raster file types should be converted to chips 
                        (this is important since image files are associated w/ other file type like an .xml,.cpg,.dbf this makes sure those are ignored)
        save_fmt (str): file format for chips. Exs: jpeg, png
    
    outputs:
        folder containing chip image-files and any support files. 
    """

    ##----------------------------Configuration/Setup---------------------------##
    ds = config['ds']                                                           #dataset (root folder)
    fs = config['fs']                                                           #existing fileset (subfolder)
    ts = config['ts']                                                           #optional tileset (subfolder). Default is None, which means no tiles will be saved.
    
    window = __getWindow(window_config=config['mWH'])                           #dictionary w/ max window dimensions (default: width:5000,height:5000) 
    
    file_formats = config['fmts'].replace(' ','').split(',')                    #convert string of file formats to list of formats
    save_fmt = config['sfmt']                                                   #file format for chips when they are saved. (png,jpeg,tif) https://gdal.org/drivers/raster/index.html

    worldDir = '%s/region_%s'%(ds,fs)                                           #directory with files to convert to chips 
    worldFiles = __listdir(directory=worldDir,
                           extensions=file_formats)                             #list of files to convert to chips
    
    if ts is not None:                                                          #subfolder for saving tiles
        tilesDir = '%s/imageTiles_%s'%(ds,ts)
        __make_folders(rootfolder=ds,
                       subfolder=tilesDir)    
    
                           
    chipsDir = '%s/imageChips_%s'%(ds,fs)                                   #subfolder for saving chips
    vbPrint('%i Region files found in %s'%(len(worldFiles),worldDir))           #display the number of files to be processed
    
    
    __make_folders(rootfolder=ds,                                               ##Make folder to hold chips
                   subfolder=chipsDir)                                        #method for making project rootfolder/subfolders
    
    

    
    
    ##----------------------------Create Chips------------------------------------##
    for count,imageName in enumerate(worldFiles):
        image_path = fr'{worldDir}/{imageName}'                                #recreate image path 
        vbPrint(f'Current Image Path: {image_path}')
        
        
        #Open world file & apply window to generate Tiles 
        for tile_array,tile_meta,tile_name in open_raster(path_to_region = image_path,      #current world file
                                                          maxWidth = window['width'],       #this is maxWidth/maxHeight
                                                          maxHeight = window['height'],        #actual height/width may be less since no padding happens when opening
                                                          verbose = config['verbose']):
                                                             
            if 0 in tile_array.shape:
                vbPrint(f'empty array generated: {tile_array.shape}. Skipping')
                continue
            
            
            
            ## Creation of Image chips
            split_array = split_raster(image=tile_array,                        #current tile to split into chips
                                       chipX = 256,                            #chip width
                                       chipY = 256,                            #chip height
                                       minScrapPercent = 0,                     #0 means every tile_array will be padded before splitting (otherwise chips along bot/right edges are discarded)
                                       verbose = config['verbose'])
        
            if ts is not None:
                save_tile(array=tile_array,                                     #tile array
                          save_directory=tilesDir,                              #folder location to save tile in 
                          tile_name=tile_name,                                  #chips will be named as: region_tile_{tileXpos}_{tileYpos}
                          profile = tile_meta,                                  #rasterio requires a 'profile'(meta data) to create/write files
                          save_fmt = save_fmt,                                  #format to save tiles as (ex: png,jpeg,geoTiff) 
                          verbose = config['verbose'])
            
            ##Save Image Chips:
            save_chips(array=split_array,                                     #split tile containing chips
                         save_directory=chipsDir,                             #folder location to save chips in 
                         tile_name=tile_name,                                   #chips will be named as: tileName_tileXpos_tileYpos_chipXpos_chipYpos
                         profile = tile_meta,                                   #rasterio requires a 'profile'(meta data) to create/write files
                         save_fmt = save_fmt,                                   #format to save chips as (ex: png,jpeg,tif) 
                         verbose = config['verbose'])

    
    ##----------------------------Quick Summary------------------------------------##
    if ts is not None:
        tileFiles = __listdir(directory=tilesDir,
                              extensions=['all']) 
        vbPrint(f"Number of files created in {tilesDir}: {len(tileFiles)}\n{'-'*4}Image Tiles made successfully{'-'*4}")
        
        
    chipFiles = __listdir(directory=chipsDir,
                           extensions=['all']) 
                           
    vbPrint(f"Number of files created in {chipsDir}: {len(chipFiles)}\n{'-'*4}Image Chips made successfully{'-'*4}")
    
    """
    Example:
        * python3 data_pipeline.py --ds=dataset5 --fs=labels --fmts=tif --sfmt=png --mWH=5000,5000 -genImgChips
        * python3 data_pipeline.py --ds=dataset5 --fs=inputs --fmts=jpg --sfmt=png --mWH=5000,5000 -genImgChips
        * python3 data_pipeline.py --ds=dataset7 --fs=labels --ts=labels --fmts=tif --sfmt=tif --mWH=5000,5000 -genImgChips
        * python3 data_pipeline.py --ds=dataset7 --fs=labels --ts=labels --fmts=tif --sfmt=tiff --mWH=5000,5000 -genImgChips
        * python3 data_pipeline.py --ds=dataset8 --fs=labels --ts=labels --fmts=tif --sfmt=tif --mWH=5000,5000 -genImgChips
        * python3 data_pipeline.py --ds=dataset9 --fs=labels --ts=labels9 --fmts=tif --sfmt=tif --mWH=5000,5000 -genImgChips
    """

def genAnnotations():
    """
    Generates the train and test annotations files using the image and label chips
    
    Requirements:
        * label chips should be available in <ds>/labelChips_<ts>/..
        * While not required by this function, corresponding image chips should be available in <ds>/imageChips_<ts>/..

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

    labelChipsDir = '%s/labelChips_%s'%(ds,ts)
    annDir = '%s/annotations_%s'%(ds,ts)

    ## Making the dirs
    if(os.path.isdir(annDir)):
        vbPrint('Found dir: %s'%(annDir))
    else:
        vbPrint('Making dir: %s'%(annDir))
        os.mkdir(annDir)

    ## Getting the labels and splitting into training and validation images
    vbPrint('Reading `%s` for label chips'%(labelChipsDir))
    labels = __listdir(labelChipsDir, ['tif'])
    vbPrint('Dataset Size   : %i'%(len(labels)))
    labels = np.asarray(labels)
    np.random.shuffle(labels)
    splitIdx = int(labels.shape[0]*tvRatio)
    trainData,valData = np.split(labels,[splitIdx])
    
    vbPrint('Training Data  : %i'%(len(trainData)))
    vbPrint('Val Data       : %i'%(len(valData)))
    
    # Generate Annotations
    vbPrint('Generating annotations for the training data')
    trainR2C = Raster2Coco(trainData, labelChipsDir)
    trainJSON = trainR2C.createJSON()

    with open('%s/annotations_train.json'%(annDir), 'w+') as outfile:
        json.dump(trainJSON, outfile)
    
    del trainJSON
    del trainR2C

    vbPrint('Generating annotations for the validation data')
    valR2C = Raster2Coco(valData, labelChipsDir)
    valJSON = valR2C.createJSON()

    with open('%s/annotations_val.json'%(annDir), 'w+') as outfile:
        json.dump(valJSON, outfile)
    
    del valJSON
    del valR2C

    vbPrint('Annotations made and saved successfully')



if __name__ == '__main__':
    """
    Argument Formats:``
        --<KEY>=<VALUE>     -> Used to set config
        -<PipelineStep>     -> Used to set pipeline Steps

    Base Arguments:
        --ds                -> [dataset name] The datset name. This will be used to create the file directory and refer to the dataset.
        --ts                -> [tileset name] Name of the tileset. This will be used to create the unique tiles and images folder.
                                    Separate tile and image folders are needed to separate training data from inference data.
                                    This also allows separation of inference set. Ex of values for ts: train,test, XYZ_County_inference, PQR_Inference, etc.
        -<PipelineStep>     -> [Function Name] name of the function to call in the pipeline step. Multiple can be passed to create the pipeline.

    LoadTiles Arguments:
        --s3td              -> [s3 URL] The s3 URI of the input image tiles. Ex: s3://njtpa.auraison.aegean.ai/DOM2015/Selected_500/pipelineTest

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
        Training Pipeline
        * python dataPipeline.py --ds=ds1 --ts=train -loadTiles --s3td=test -genImgChips -genLabelChips -genAnnotations

        MEVP Pipeline
        * python dataPipeline.py --ds=ds1 --ts=inf1 -loadTiles --s3td=test -genImgChips -genAnnotations
        * python -genInferenceJSON --trained_model=weights/DVRPCResNet50_8_88179_interrupt.pth --config=dvrpc_config --score_threshold=0.0
        * python dataPipeline.py --ds=ds1 --ts=inf1 -genInferenceData --annJSON=DVRPC_test.json --infJSON=DVRPCResNet50.json
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