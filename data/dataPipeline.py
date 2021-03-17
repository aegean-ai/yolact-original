import os
import traceback
import sys

config = {
    'pipeline':[],
    'verbose': 1,
    'ds':None,
    'pef': 0.1 # [PrintEveryFactor] Determines after how much increase in the factor of completion of a process, a vbPrint is made. 0.1 means at every 10%, a print is made.
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
        'verbose':int,
        'pef' : float
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

############## PRE TRAINING DATA PIPELINE FUNCTIONS #############

def loadTiles():
    '''
    loads tiles data from s3 into the directory:
        - ./data/<datset>/tiles/
    '''

    ds = config['ds']
    ts = config['ts']
    s3Dir = config['s3td']
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
    '''
    TODO
    - load all tiles from s3 [s3DIR] into the ec2 instance [loadDIr]
    - This can be done by using boto3 for python or executing a shell script (with os.system()) and 'aws s3 sync'
    - It is assumed that the selection of 500 tiles for training is done manually
    '''

    vbPrint('Tiles loaded succesfully')

def genImgPatches():
    '''
    Generates image patches by splitting the tiles
    Prerequisite: loadTiles() has been run or tiles are present in the data/<dataset>/tiles directory
    '''
    ds = config['ds']
    ts = config['ts']
    tilesDir = '%s/tiles_%s'%(ds,ts)
    tiles = os.listdir(tilesDir)
    vbPrint('%i Tile files found in %s'%(len(tiles),tilesDir))
    patchesDir = '%s/patches_%s'%(ds,ts)
    
    ## Making the dirs
    if(os.path.isdir(patchesDir)):
        vbPrint('Found dir: %s'%(patchesDir))
    else:
        vbPrint('Making dir: %s'%(patchesDir))
        os.mkdir(patchesDir)
    
    ## Creation of Image and Label patches
    '''
    TODO
    - Each tile has to be split into 400 patches
    - The patches have to be saved in the [patchesDir] directory
    - The naming of the paches should be such that if we have the name of tile, we can generate the name of its patches
      We could follow the format that has been used for year-1, it has been documented in the 'Formation of image patches'
      section at docs.upabove.app/sidewalk/data-pipeline/_index.md
    - Investigate and implement how to handle the world files
    '''

    vbPrint('Image Patches made successfuly')


'''
INVESTIGATION NEEDED FOR genLabelPatches AND genAnnotations:
    - We know that training uses images and annotations.json (and not the binary label rasters)
    - How are the label raster patches being generated? Do we have label tiles ?
    - If we have the tiles, can they be split in memory and their annotations made without having to save them into storage ?
'''
def genLabelPatches():
    '''
    - Generates the label patches
    - For training only
    '''
    vbPrint('Label Patches made successfuly')
    pass

def genAnnotations():
    '''
    - Generates the image and label annotations
    - For training only
    '''
    vbPrint('Image and Label Annotations made successfuly')
    pass

############## POST TRAINING DATA PIPELINE FUNCTIONS #############

'''
Question: We dont need to merge images cause we already have the tiles right ?

Idea:
    Instead of first generating a large number of inferences and then merging them,
    why dont we go tile by tile, making batch inferences for each of its 400 patches, and save it as the merged inference Tile?
    We can always make individual inference patches with eval.py  
'''

def genInferenceTiles():
    '''
    For each tile, generate an inference for each of the patches
    '''
    ds = config['ds']
    ts = config['ts']
    tilesDir = '%s/tiles_%s'%(ds,ts)
    tiles = os.listdir(tilesDir)
    vbPrint('%i Tile files found in %s'%(len(tiles),tilesDir))

    tiles =[0]*2000
    n = len(tiles)

    printAtFactor = config['pef']

    for i,tile in enumerate(tiles):
        facComp = (i+1)/(n)
        if(facComp >= printAtFactor):
            vbPrint('Generating tile inferences: %i/%i %0.2f %%'%(i,n,100*facComp))
            printAtFactor += config['pef']

            


    vbPrint('Inference Tiles generated Successfully')

if __name__ == '__main__':
    '''
    Argument Formats:
        --<KEY>=<VALUE>     -> Used to set config
        -<PipelineStep>     -> Used to set pipeline Steps
    Argument keys:
        --ds            -> [dataset name] The datset name. This will be used to create the file directory and refer to the dataset.
        --ts            -> [tileset name] Name of the tileset. This will be used to create the unique tiles and images folder.
                                          Separate tile and image folders are needed to separate training data from inference data.
                                          This also allows speration of inference set. Ex of values for ts: train,test, XYZ_County_inference, PQR_Inference, etc.
        --s3td          -> [s3 URL] The s3 URI of the input image tiles. Ex: s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/DOM2015/Selected_500/
        -<PipelineStep> -> [Function Name] name of the function to call in the pipeline step. Multiple can be passed to create the pipeline.
        --verbose       -> [0 or 1] Setting it to 1 allows print statements to run. Default: 1
        --pef           -> [float] (Used for long processes) Determines after how much increase in the factor of completion of a process, a vbPrint is made. 
                           For example: 0.1 means at every 10% completion, a print is made. (Only works if --verbose is set to 1). Default: 0.1
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