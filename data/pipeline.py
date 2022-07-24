"""
Definitions:
    * Region-File/Region-Image: This is any input image of arbitrary size that is broken down into tiles.
    * Tile Image: Tiles are what a regional image is divided to.  The default tile size is 5000x5000.
    * (es)/ Image : This is the smallest subdivision of a World Image. This is hardcoded 256x256
    * The chip size is set according to the input tensor size for the Yolact Model
    
    * Dataset
      * Each project can be considered one dataset. 
      * It is possible that we want to work in a project similar to
      a previous one, eg to do object detection on a completely new type of images i.e. we want to use the same pipeline for an entirely
      different project. 
      * To ensure separation of this data, a new dataset can be generated for a new project.
      * The dataset is abbreviated as 'ds' throughout the documentation and code.
    
    * Tileset
      * Each batch of tiles can be considered as a tileset. A dataset may have multiple tilesets.
      * Any new batch of data that arrives should be treated like a new tileset. 
      * All Pipeline steps take the dataset and tileset information and process only the mentioned dataset and tileset. 
      This ensures that we don't have to unnecessarily process any old data in the project.
      * The tileset is abbreviated as 'ts' throughout the documentation and code.

General Info:
    * This file has the code to perform all steps of the pipeline.
    * The individual steps can be found in our github documentation: docs.upabove.app/sidewalk/_index.md
    * The configuration of the pipeline as well as the steps to perform are passed as terminal arguments 

Structure:
    This code in this file is present in N sections:
    Imports/Setup
        * Has all imports required throughout the code.
    Helper Functions
        * Common functions used throughout the code.
    Pipeline Functions
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
    * genInferenceJSON
    * genInferenceData #TODO -> GeoJSON projections
    * Cleaning and Metrics #TODO
    * exportData #TODO

    Attributes: 
    
    Todo: 
    
"""

#----------------------------------------------------------------#
#------------------------ Imports/Setup -------------------------#
#----------------------------------------------------------------#


import traceback
import sys

from s3_helper import s3_to_local, local_to_s3
from pipeline_helper import *

#called within download_labels

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