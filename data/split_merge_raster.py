"""
windowed reading: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

#Note: in laymens terms the intent of split_merge_raster when splitting is to reshape an array
        such that it results in equally sized chips of size (chipX,chipY).
        
        Each chip is stored in an array indexed with a key representing its (chipkXpos,chipYpos) position.
        such that the final array will have a shape (chipXpos,chipYpos,chipXdim,chipYdim)
        
        Number of Padding cells (paddingSize) will satisfy (0 <= xPad <= chipXdim-1), yPad = (0 <= yPad <= chipYdim-1). 
        
        Padded cells will have a constant value of p = 0 representing that they do not belong to original image.
        the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
        
        If the % of original cells in a padded chip is < minScrapPercent then chips along that edge will be discarded (ie 'scrapped')
        **There is no compensation for chips at position NM and may result in minScrapPercent < % of original cells** 
        
                  <chipXdim>
      |  x0  | ... |  xN  |
    _  ______ _____ ______
      |      |     |     p|^
    y0|  b00 | b10 | b0N p|chipYdim
    __|______|_____|_____p|v
     .|                  .|
     .|  original image  .|
     .|                  .|
    __|______|_____|_____.| 
      |      |     |     p|
    yM| b0M  | ... | bNM p|
      |pppppp|ppppp|pppppp|
    ---------------------- 

#Flow:
    inputArray > padd > cut > outPut array

private functions:
    __PaddTiff()
            calls: None
        called in: SplitTiff
        Output flows to: __Cutter
       
public function:
    SplitTiff()
            calls: __PaddTiff,__Cutter
        Called in: Unknown
         Output flows to: unKnown

"""

import numpy as np
import sys
import rasterio as rio
from rasterio import coords
import rasterio.plot                            #---NOTE! For some reason rasterio does not automatically have all submodules available 
import rasterio.mask                            #         They must be imported specificaly



def __init__():
    pass

def __get_size(obj, seen=None):
    #Source: https://goshippo.com/blog/measure-real-size-any-python-object/
    """Recursively finds size of objects"""
    
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([__get_size(v, seen) for v in obj.values()])
        size += sum([__get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += __get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([__get_size(i, seen) for i in obj])
    return size
    
#-------------------Working with Raster Files:
"""
raster.open()
https://rasterio.readthedocs.io/en/latest/api/rasterio.html?highlight=raster.open#rasterio.open
"""

def __enumerate_tiles(width,height,maxWidth,maxHeight):
    #row is y direction (height), col is x direction (width)
    
    colsplit, rowsplit = width//maxWidth, height//maxHeight
    
    for col in range(0,colsplit*maxWidth+1,maxWidth):
        for row in range(0,rowsplit*maxHeight+1,maxHeight):
            
            yield row, row+maxHeight, col, col+maxWidth
            
            
            
def __newTransform(oldtransform,xy,verbose=False):                              #Not Currently need: See Jira UPA-47: Retaining geoLocation Information 
    """
    Notes:
        This function exists right now in the event tiles/chips need to be exported to Arcgis Pro.  
        It ensures that the tiles/chips will end up in the correct location on the map.
        
    https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html   
    
    Inputs:
    oldtransform (affine): rasterio affine object
                xy (dict): cols:left_col (x), rows:top_row (y)  pixel number
        
    """

    
    left,top = rio.transform.xy(oldtransform, rows=xy['rows'], cols=xy['cols'])
    xpixel,ypixel = -1*float(oldtransform.e),float(oldtransform.a)
    newtransform = rio.transform.from_origin(west=left, 
                                             north=top, 
                                             xsize=xpixel,
                                             ysize=ypixel)
                                             
    if verbose:
        print(f'xy: {xy}')
        print(f'left: {left}, top: {top}')
        print(f'oldpixels: {type(oldtransform.a)},{type(oldtransform.e)}')
        print(f'xpx: {xpixel},ypx: {ypixel}')
        print(f'oldtransform:\n{oldtransform}')
        print(f'newtransform:\n{newtransform}')

    return newtransform

def open_raster(path_to_region:str,maxWidth:int=5000, maxHeight:int=5000, verbose:bool=False)-> np.array:
        """
            Note: convert / read the data into a numpy array,
                    return summary of information if needed
                    
                  ***   src.read() will read all bands  
                        currently all functions do accomodate multiband arrays
                        
                        indexes : list of ints or a single int, optional
                                  If `indexes` is a list, the result is a 3D array, but is
                                  a 2D array if it is a band index number.
                                                                                 
                        
                    
            inputs:
                path_to_region (string): File path to region image. 
                verbose (boolean): Used to print summary if needed
            
            outputs:
                tile_array (np.arrary): returned array 
                

        """
        region_name = path_to_region.rsplit('/',1)[1].rsplit('.',1)[0]
        
        with rio.open(path_to_region) as src:
            bands = src.meta['count']
            tile_meta = src.meta.copy()
            
            if verbose:                                                         #Summary info about World file
                print('-'*4,'region summary','-'*4)
                print('region datatypes: ', {i: dtype for i, dtype in zip(src.indexes, src.dtypes)})
                print('color interp: ', src.colorinterp)
                print('tags: ',src.tags())
                print('bounds: ', src.bounds)
                print('extents: ', rio.plot.plotting_extent(src))
                print('res: ',src.res)
                print('shape: ',src.shape)
                print(f'Transform:\n{src.transform}')
                print('units: ', src.units)
                print('meta data:\n',src.meta)
                print('-'*4, 'tile summary' ,'-'*4)
                
            for top_row, bottom_row, left_col, right_col in __enumerate_tiles(src.meta['width'],src.meta['height'],maxWidth=maxWidth,maxHeight=maxHeight):
                
                tile_array = src.read(window = ((top_row,bottom_row),(left_col,right_col)))   #window reads y,x not as x,y
                
                L = len(tile_array.shape)                                       #Checks array shape
                if L == 2:
                    x,y = tile_array.shape
                    newShape = (1,x,y)
                    tile_array = np.reshape(tile_array,newShape)                #ensures array shape is (z,x,y) this is neccessary for split_raster() step
                    
                    
                tile_meta['width']  = tile_array.shape[1]                       #assumes tile has shape (bands,width,height)
                tile_meta['height'] = tile_array.shape[2]
                
                tile_meta['transform'] = __newTransform(oldtransform=src.profile['transform'],       
                                                        xy={'cols':left_col,'rows':top_row},
                                                        verbose=True)
                
                tile_name = fr'{region_name}_tile_{left_col}_{top_row}'          #note that the xy order is different from window read
                
                if verbose:                                                     #Print a summary of stats about Image
                    print(fr'{tile_name}: {type(tile_array)} | tile array shape: {tile_array.shape} | tile array memory size: {__get_size(tile_array)}')
                
                yield tile_array,tile_meta,tile_name
        
        if verbose:
            N = (top_row//maxHeight + 1) * (left_col//maxWidth + 1)
            print(fr'{N} Tiles Scanned for Region File: {path_to_region}','\n')



def __pad_raster(image:np.array,chipX:int,chipY:int,minScrapPercent:float,verbose:bool=False)-> np.array:
    """
    Note:     This function is called w/in split_raster() with the goal of providing padding along the 
                right & bottom edge of the input image such that the final dimensions of the output image are
                integer multiples of chip dimension chipX & chipY
                
                The padding size will satisfy (0 <= paddingSize <= baseX-1) and will have a constant value 
                of (0) representing that the padded cells do not belong to the original image. 
                the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
                
                if (Percent% of the original cells within a padded chip) < minScrapPercent
                the chips along that edge will be 'cut' (ie "scrapped")
        
    Inputs:        
        image (numpy.array): array that will be padded such that shape of the final array 
                             will have integer multiples of blockX & blockY.
        
        chipX (integer)    : the xdim of blocks after split
        chipY (integer)    : the ydim of blocks after split
        
        minScrapPercent (float) : min % of original cells required in edges.
                                  The intent is to prevent slivers of cells with not enough context
                                  resulting in reduced performance metrics. 
        
    Outputs:
        paddedImg (numpy.array): array such that its (xDim,yDim) are integer multiples of (chipX,chipY)
        
    """

    bands, imgX,imgY = image.shape 
    
    scrapX = (imgX % chipX)/chipX                                             # % of cells that are along right edge
    scrapY = (imgY % chipY)/chipY                                             # % of cells that are along bottom edge
    
    padX = chipX-(imgX % chipX) if scrapX >= minScrapPercent else 0           #Amount of cells to add to right edge
    padY = chipY-(imgY % chipY) if scrapY >= minScrapPercent else 0           #Amount of cells to add to bottom edge
    
    xDim = (imgX//chipX + 1)*chipX if padX else (imgX//chipX)*chipX         #xDimension of output image
    yDim = (imgY//chipY + 1)*chipY if padY else (imgY//chipY)*chipY         #yDimension of output image
    
    paddims = ((0,0),(0,padX),(0,padY))
    
    paddedImg = np.pad(array = image,                                           #Apply padding to input image
                       pad_width = paddims,                                     #Pad only bottom & right edges 
                       mode='constant',                                         #Fill with a constant value
                       constant_values = 0)                                     #Contant value of 0
                       

    paddedImg = paddedImg[:,0:xDim,0:yDim]                                      #Crop image, if padX,padY are 0 then this results in
                                                                                # 'scrapping' that edge and losing those cell values
        
    if verbose:                                                                 #Return summary of results 
        print(f"{'-'*5} 'Summary of Padding/Scrapping' {'-'*5}")
        print('xEdge has been padded') if xDim > imgX else print('xEdge has been scrapped') if xDim < imgX else print('xEdge has not been modified')  
        print('yEdge has been padded') if yDim > imgY else print('yEdge has been scrapped') if yDim < imgY else print('yEdge has not been modified')
        print(f'input image shape: {image.shape}')                                            
        print(f'output image shape: {paddedImg.shape}\n')
    
    return paddedImg
    

            

def split_raster(image:np.array,chipX:int,chipY:int,minScrapPercent:float,verbose:bool = False)-> np.array:
    """
    #Note: input numpy.array will be reshaped into blocks such that the shape of 
            resulting blocks will have dimensions (blockX,blockY)
            
            This function will result in the right & bottom edges individually being padded or scrapped.
           
            The padding size will satisfy (0 <= paddingSize <= baseX-1) so that all blocks are of equal size
            padding will have a constant value of (0) representing that the padded cells do not belong to the original image.
            the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
                
            if (Percent% of the original cells within a padded block) < minScrapPercent
            the blocks along that edge will be 'cut' (ie "scrapped").
            
            if minScrapPercent <= 0 then nothing will be scrapped
           
           
    Inputs:        
        image (numpy.array) : array that will be split.
        
        chipX (integer)    : the xdim of chips after split
        chipY (integer)    : the ydim of chips after split
        
        minScrapPercent (float) : min % of original cells required for blocks on edge.
                                  The intent is to prevent slivers of cells with not enough context
                                  resulting in reduced performance metrics. 
         
    Outputs:
        chips (np.array):  reshaped array split into chip. 
                             final shape: (bands,xchipPosition,xchipSize,ychipPosition,ychipSize)       
    """
    
    
    paddedImg = __pad_raster(image,chipX,chipY,minScrapPercent,verbose)        #get padded image
    
    bands, xDim,yDim = paddedImg.shape                                          #padded image dimensions
        
    xSize,ySize = xDim//chipX,yDim//chipY                                     #block x,y arrangment
    
    newShape = (bands,xSize,chipX,ySize,chipY)                                #bands,xchipPosition,xchipSize,ychipPosition,ychipSize
    
    chips = np.reshape(paddedImg,newShape)                                    #reshape into blocks

    
    
    if verbose:                                                                 #print summary
        print(f"{'-'*5} Summary of Splitting {'-'*5}")
        print(f'Input Type: {type(image)}')
        print(f'Input Shape: {image.shape}')
        print(f'Input Memory Sizes: {__get_size(image)}, {image.nbytes}')
        print(f'Output Type: {type(chips)}')
        print(f'Shape Reference: (bands,xSize,chipX,ySize,chipY)')
        print(f'Intended shape: {newShape}')
        print(f'Ouput Shape: {chips.shape}')
        print(f'Output Memory Sizes: {__get_size(chips)}, {chips.nbytes}\n')
        print('memory %change: {0:+8.4f}%\n\n'.format((chips.nbytes/image.nbytes - 1)*100))
    return chips
    




    

def __enumerate_chips(array):
    #<Image Index (Integer)>_<row number>_<col number>
    #shp = (bands,xBlockPosition,xBlockSize,yBlockPosition,yBlockSize) 
    
    xBlocknum = array.shape[1]
    yBlocknum = array.shape[3]
    
    for xBlockPosition in range(xBlocknum):
        for yBlockPosition in range(yBlocknum):
            yield xBlockPosition,yBlockPosition


def __make_worldfile(affine,file_path:str,verbose:bool):
    
    """
    Note:
        to future maintainters there is currently only 1 filetype in the exclusion list (tif)
        if other filetypes that use header files are found, please add them to the exclusion list. 
    
    Inputs:
                 affine: affine transform object
        file_path (str): file path of parent image (must include extension)
    
    Outputs:
        None: results in a world file being created and returns nothing 
        
    Sources;
        per guidlines: http://webhelp.esri.com/arcims/9.3/General/topics/author_world_files.htm
    
    Example World file format:
        20.17541308822119 = A
        0.00000000000000 = D
        0.00000000000000 = B
        -20.17541308822119 = E
        424178.11472601280548 = C
        4313415.90726399607956 = F
    
    """

    splitfp = file_path.rsplit('.',1)                                           #ideally file_path is of form {filename}.{fmt}
    
                                                                                #splitfp should ideally be ['filename','fmt']
    if len(splitfp) >= 2:                                                       #checks for ideal case                                                       
        filename,fmt = splitfp[0],splitfp[1] 
    else:                                                                       #if there is no extension 
        filename = splitfp[0]
        fmt = ''
        if filename == '':                                                      #if file name ends up being empty string something went wrong
            print(f'world file could not be created for: {file_path} | filename of zero length generated') if verbose else None
            return None
            
    if fmt.lower() in ('tif',):                                               #Check exclusion list (files that use header to store world data) 
        print('No world file generated: file type is in exclusion list') if verbose else None
        return None        
    elif len(fmt) >= 3:                                                         #Uses 1st & 3rd letters of extension 
        f1,f2 = fmt[0],fmt[2]
        fmt = f'{f1}{f2}w'
        file_path = f'{filename}.{fmt}'
    elif len(fmt) == 0:                                                         #if image has no extension, w is appended to file name (per guidelines)
        file_path = f'{filename}w'
    else:                                                                       #Append 'w' without any other modification
        fmt = f'{fmt}w'
        file_path = f'{filename}.{fmt}'
    
    
    data = f'{affine.a}\n{affine.d}\n{affine.b}\n{affine.e}\n{affine.c}\n{affine.f}'  
    
    with open(file_path,'w') as worldfile:                                      #Create world file
        worldfile.write(data)        
    print(f'world file create: {file_path}') if verbose else None
    
    
   
def save_chips(array:np.array, save_directory:str, chip_file_format:str, tile_name:str, profile:dict, verbose:bool=False)->None: 
    """
    Notes:
        Method for saving image chips
          does not currently update affine transform the upper left pixel location will have to be worked out for each chip
          
    * Note for future maintainters: save_tile & save_chips have very similar processes and inputs. If one function is updated or changed, 
    please review the other function to ensure that update is not needed there.     
        
    Inputs:
        array (np.array): tile array 
        save_directory (str): name of subfolder that will hold chips 
        chip_file_format (str): file type for image chips. ex JPEG/PNG etc
        tile_name (str):  naming convention '{world_name}_tile_{left_col}_{top_row}'
        profile (dict): tile_meta
        
    
    References:      
        Image profiles: https://rasterio.readthedocs.io/en/latest/topics/profiles.html
        Supported drivers: https://gdal.org/drivers/raster/index.html
        Saving Images: https://rasterio.readthedocs.io/en/latest/topics/writing.html           
    """    
    

    path_to_save = fr'{save_directory}/{tile_name}'
    
    tile_affine = profile['transform']

    if chip_file_format == 'jpg':
        profile['driver'] = 'JPEG'
    elif chip_file_format=='tiff':
        profile['driver'] = 'GTiff'
    
    profile['count'] = array.shape[0]                                               #band count
    profile['width'] = array.shape[2]                                               #xPatchSize
    profile['height'] = array.shape[4]                                              #yPatchSize
    profile['photometric'] = 'RGB'
    
    print(f'updated raster meta data:\n{profile}') if verbose else None
    
    for row_num, col_num in __enumerate_chips(array):                                     #row_num, col_num: the chip position in the sliced array (not directly a pixel reference)
        chip_path = fr'{path_to_save}_chip_{row_num}_{col_num}.{profile["driver"]}'       #chip filename: {world_name}_tile_{left_col}_{top_row}_chip_{row_num}_{col_num}.{fmt}
        
        left_col = col_num*profile['width'] 
        top_row = row_num*profile['height']
        
        profile['transform'] = __newTransform(oldtransform= tile_affine,            #updates the affine to set lat/long of upper left pixel 
                                              xy={'cols':left_col,'rows':top_row},
                                              verbose=True)        
        
        with rio.open(chip_path,'w',**profile) as dst:                             #create file
            left_col, top_row = col_num*profile['width'], row_num*profile['height'] #slice out chip
            dst.write(array[:,row_num,:,col_num,:])                                 #array[bands,xPatchPosition,xPatchSize,yPatchPosition,yPatchSize] 
            __make_worldfile(profile['transform'], chip_path,verbose)              #method for creating a world file. (a document that contains key affine info)
        
    print(f'Save Completed to {path_to_save}\n') if verbose else None
    
    
    
def save_tile(array:np.array, save_directory:str, save_fmt:str, tile_name:str, profile:dict, verbose:bool=False)->None: 
    """
    Notes:
        Method for saving tiles
          does not currently update affine transform the upper left pixel location will have to be worked out for each chip
    
    *Note for future maintainters: save_tile & save_chips have very similar processes and inputs. 
                                    If one function is updated or changed, please review the other function to ensure that update is not needed there.  
        
    Inputs:
        array (np.array): tile array 
        save_directory (str): name of subfolder that will hold tile
        save_fmt (str): file type for image tile. ex JPEG/PNG etc
        tile_name (str): naming convention '{region_name}_tile_{left_col}_{top_row}'
        profile (dict): tile_meta
          
    Outputs:
        None: will result in a tile being saved as tile_name to save_directory in specified save_fmt format
        
        
    References:      
              image profiles: https://rasterio.readthedocs.io/en/latest/topics/profiles.html
           Supported drivers: https://gdal.org/drivers/raster/index.html
               Saving Images: https://rasterio.readthedocs.io/en/latest/topics/writing.html           
    """    
    
    
    profile['driver'] = 'JPEG' if save_fmt == 'jpg' else save_fmt.upper()       #rasterio recognizes JPEG for jpg
    profile['count'] = array.shape[0]                                           #band count
    profile['width'] = array.shape[1]                                           #xTileSize
    profile['height'] = array.shape[2]                                          #yTileSize
    profile['photometric'] = 'RGB'
    
    print(f'updated raster meta data:\n{profile}') if verbose else None
    
    
    tile_path = fr'{save_directory}/{tile_name}.{profile["driver"]}'

    if array.max():
        print('Array Has Data') if verbose else None
        with open(fr'{save_directory}/info.txt','a') as info:
            info.write(f'{tile_name}.{profile["driver"]}: {array.sum()}\n')
            
    with rio.open(tile_path,'w',**profile) as dst:
        dst.write(array)                                                        #tile array
        __make_worldfile(profile['transform'], tile_path,verbose)                       #method for creating a world file. (a document that contains key affine info)
        
    print(f'Save Completed to {save_directory}\n') if verbose else None
