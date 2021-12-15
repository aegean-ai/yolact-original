"""
Pipeline helper functions. 
"""
import os
from pipeline_config import *
import traceback
import time

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