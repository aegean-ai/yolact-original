"""
windowed reading: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html

#Note: in laymens terms the intent of split_merge_raster when splitting is to reshape an array
        such that it results in equally sized patches of size (patchX,patchY).
        
        Each patch is stored in an array indexed with a key representing its (patchkXpos,patchYpos) position.
        such that the final array will have a shape (patchXpos,patchYpos,patchXdim,patchYdim)
        
        Number of Padding cells (paddingSize) will satisfy (0 <= xPad <= patchXdim-1), yPad = (0 <= yPad <= patchYdim-1). 
        
        Padded cells will have a constant value of p = 0 representing that they do not belong to original image.
        the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
        
        If the % of original cells in a padded patch is < minScrapPercent then patches along that edge will be discarded (ie 'scrapped')
        **There is no compensation for patches at position NM and may result in minScrapPercent < % of original cells** 
        
                  <patchXdim>
      |  x0  | ... |  xN  |
    _  ______ _____ ______
      |      |     |     p|^
    y0|  b00 | b10 | b0N p|patchYdim
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
    
    for col in range(0,colsplit*maxWidth-1,maxWidth):
        for row in range(0,rowsplit*maxHeight-1,maxHeight):
            
            yield row, row+maxHeight, col, col+maxWidth
            
            
            
def __newTransform(oldtransform,xy,verbose=False):                              #Not Currently need: See Jira UPA-47: Retaining geoLocation Information 
    """
    Notes:
        This function exists right now in the event tiles/patches need to be exported to Arcgis Pro.  
        It ensures that the tiles/patches will end up in the correct location on the map.
        
    https://rasterio.readthedocs.io/en/latest/api/rasterio.transform.html    
    """

    
    left,top = xy
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

def Open_Raster(path_to_world:str,maxWidth:int=5000, maxHeight:int=5000, verbose:bool=False)-> np.array:
        """
            Note: convert / read the data into a numpy array,
                    return summary of information if needed
                    
                  ***   src.read() will read all bands  
                        currently all functions do accomodate multiband arrays
                        
                        indexes : list of ints or a single int, optional
                                  If `indexes` is a list, the result is a 3D array, but is
                                  a 2D array if it is a band index number.
                                                                                 
                        
                    
            inputs:
                path_to_world (string): File path to world image. 
                verbose (boolean): Used to print summary if needed
            
            outputs:
                tile_array (np.arrary): returned array 
                

        """
        world_name = path_to_world.rsplit('/',1)[1].rsplit('.',1)[0]
        
        with rio.open(path_to_world) as src:
            bands = src.meta['count']
            tile_meta = src.meta.copy()
            
            if verbose:                                                         #Summary infor about World file
                print('-'*4,'world summary','-'*4)
                print('world datatypes: ', {i: dtype for i, dtype in zip(src.indexes, src.dtypes)})
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
                    tile_array = np.reshape(tile_array,newShape)                #ensures array shape is (z,x,y) this is neccessary for Split_Raster() step
                    
                    
                tile_meta['width']  = tile_array.shape[1]                       #assumes tile has shape (bands,width,height)
                tile_meta['height'] = tile_array.shape[2]
                
                #tile_meta['transform'] = __newTransform(oldtransform=src.profile['transform'],       #This function is not currently need See Jira UPA-47
                #                                        xy=src.xy(col=left_col,row=top_row),
                #                                       verbose=True)
                
                tile_name = fr'{world_name}_tile_{left_col}_{top_row}'          #note that the xy order is different from window read
                
                if verbose:                                                     #Print a summary of stats about Image
                    print(fr'{tile_name}: {type(tile_array)} | tile array shape: {tile_array.shape} | tile array memory size: {__get_size(tile_array)}')
                
                yield tile_array,tile_meta,tile_name
        
        if verbose:
            N = (top_row//maxHeight + 1) * (left_col//maxWidth + 1)
            print(fr'{N} Tiles Scanned for World File: {path_to_world}','\n')



def __PadRaster(image:np.array,patchX:int,patchY:int,minScrapPercent:float,verbose:bool=False)-> np.array:
    """
    Note:     This function is called w/in Split_Raster() with the goal of providing padding along the 
                right & bottom edge of the input image such that the final dimensions of the output image are
                integer multiples of patch dimension patchX & patchY
                
                The padding size will satisfy (0 <= paddingSize <= baseX-1) and will have a constant value 
                of (0) representing that the padded cells do not belong to the original image. 
                the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
                
                if (Percent% of the original cells within a padded patch) < minScrapPercent
                the patches along that edge will be 'cut' (ie "scrapped")
        
    Inputs:        
        image (numpy.array): array that will be padded such that shape of the final array 
                             will have integer multiples of blockX & blockY.
        
        patchX (integer)    : the xdim of blocks after split
        patchY (integer)    : the ydim of blocks after split
        
        minScrapPercent (float) : min % of original cells required in edges.
                                  The intent is to prevent slivers of cells with not enough context
                                  resulting in reduced performance metrics. 
        
    Outputs:
        paddedImg (numpy.array): array such that its (xDim,yDim) are integer multiples of (patchX,patchY)
        
    """

    bands, imgX,imgY = image.shape 
    
    scrapX = (imgX % patchX)/patchX                                             # % of cells that are along right edge
    scrapY = (imgY % patchY)/patchY                                             # % of cells that are along bottom edge
    
    padX = patchX-(imgX % patchX) if scrapX >= minScrapPercent else 0           #Amount of cells to add to right edge
    padY = patchY-(imgY % patchY) if scrapY >= minScrapPercent else 0           #Amount of cells to add to bottom edge
    
    xDim = (imgX//patchX + 1)*patchX if padX else (imgX//patchX)*patchX         #xDimension of output image
    yDim = (imgY//patchY + 1)*patchY if padY else (imgY//patchY)*patchY         #yDimension of output image
    
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
    

            

def Split_Raster(image:np.array,patchX:int,patchY:int,minScrapPercent:float,verbose:bool = False)-> np.array:
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
        
        patchX (integer)    : the xdim of patches after split
        patchY (integer)    : the ydim of patches after split
        
        minScrapPercent (float) : min % of original cells required for blocks on edge.
                                  The intent is to prevent slivers of cells with not enough context
                                  resulting in reduced performance metrics. 
         
    Outputs:
        patches (np.array):  reshaped array split into patch. 
                             final shape: (bands,xpatchPosition,xpatchSize,ypatchPosition,ypatchSize)       
    """
    
    
    paddedImg = __PadRaster(image,patchX,patchY,minScrapPercent,verbose)        #get padded image
    
    bands, xDim,yDim = paddedImg.shape                                          #padded image dimensions
        
    xSize,ySize = xDim//patchX,yDim//patchY                                     #block x,y arrangment
    
    newShape = (bands,xSize,patchX,ySize,patchY)                                #bands,xpatchPosition,xpatchSize,ypatchPosition,ypatchSize
    
    patches = np.reshape(paddedImg,newShape)                                    #reshape into blocks

    
    
    if verbose:                                                                 #print summary
        print(f"{'-'*5} Summary of Splitting {'-'*5}")
        print(f'Input Type: {type(image)}')
        print(f'Input Shape: {image.shape}')
        print(f'Input Memory Sizes: {__get_size(image)}, {image.nbytes}')
        print(f'Output Type: {type(patches)}')
        print(f'Shape Reference: (bands,xSize,patchX,ySize,patchY)')
        print(f'Intended shape: {newShape}')
        print(f'Ouput Shape: {patches.shape}')
        print(f'Output Memory Sizes: {__get_size(patches)}, {patches.nbytes}\n')
        print('memory %change: {0:+8.4f}%\n\n'.format((patches.nbytes/image.nbytes - 1)*100))
    return patches
    



def __Merge_and_Unpad(blocks:np.array)-> np.array: 
    """
        Note: Will rebuild image by removing padding and merging blocks.
                *Can not recover lost cells due to scrapping*
                
        inputs:
            blocks (np.array): array of blocks to be joined & unpadded
            
        output:
            unpadded (np.array): joined & unpadded array
            
    """
    
    bands,blockXpos,blockXdim,blockYpos,blockYdim = blocks.shape                                  #Contains information to rebuild original image
    mergedImg = np.reshape(blocks,(bands,blockXpos*blockXdim,blockYpos*blockYdim))                #Joins blocks into a single image
    
    xMax = ((mergedImg[:,::-1,:].cumsum(axis=1) != 0).sum(axis = 1)).max() 
    yMax = ((mergedImg[:,:,::-1].cumsum(axis = 2) != 0).sum(axis = 2)).max()                                                                                            #This finds padding by assuming padding occurs along bottom/right edges
    
    unpaddedImg = mergedImg[:,0:xMax,0:yMax]                                                      #Removes padding from bottom/right edge
    
    return unpaddedImg 

def Merge_Blocks(blocks:np.array,mergeXdim:int,mergeYdim:int,minScrapPercent:float,verbose:bool = False)-> np.array:
    """
    #Note: To make block sizes larger blocks are merged into a full image, padding is removed,
            finally the image can be broken back down to blocks of larger size. 
    
    inputs:
        blocks (np.array):  array containing image blocks that will be joined,unpadded,then split into larger block sizes
        mergeXdim (integer): the X dimension of final blocks 
        mergeYdim (integer): the Y dimension of final blocks
        minScrapPercent (float): min % of original cells required for blocks on edge.
                                 The intent is to prevent slivers of cells with not enough context
                                 resulting in reduced performance metrics. 
                                 
        verbose (boolean):  returns a summary of outputs
        
    outputs:
        mergedImg (np.array): the final array with larger blocks. 
        
    """
    
    mergedImg = __Merge_and_Unpad(blocks)                                                        #Rebuilds Image
    mergedImg = Split_Raster(mergedImg,mergeXdim,mergeYdim,minScrapPercent,verbose)                #Splits & Padds to a new size
    
    if verbose:
        print(f'\nmerged shape: {mergedImg.shape}')

    return mergedImg
    

def __enumerate_patches(array):
    #<Image Index (Integer)>_<row number>_<col number>
    #shp = (bands,xBlockPosition,xBlockSize,yBlockPosition,yBlockSize) 
    
    xBlocknum = array.shape[1]
    yBlocknum = array.shape[3]
    
    for xBlockPosition in range(xBlocknum):
        for yBlockPosition in range(yBlocknum):
            yield xBlockPosition,yBlockPosition

    
def Save_Patches(array:np.array, save_directory:str, save_fmt:str, tile_name:str, profile:dict, verbose:bool=False)->None: 
    """
    Notes:
        Method for saving image patches
          does not currently update affine transform the upper left pixel location will have to be worked out for each patch
        
        
    Inputs:
            array (np.array): tile array 
        save_directory (str): name of subfolder that will hold patches 
              save_fmt (str): file type for image patches. ex JPEG/PNG etc
             tile_name (str):  naming convention '{world_name}_tile_{left_col}_{top_row}'
              profile (dict): tile_meta
          
    
    References:      
              image profiles: https://rasterio.readthedocs.io/en/latest/topics/profiles.html
           Supported drivers: https://gdal.org/drivers/raster/index.html
               Saving Images: https://rasterio.readthedocs.io/en/latest/topics/writing.html           
    """    
    

    path_to_save = fr'{save_directory}/{tile_name}'
    
    profile['driver'] = 'JPEG' if save_fmt == 'jpg' else save_fmt.upper()
    profile['count'] = array.shape[0]                                           #band count
    profile['width'] = array.shape[2]                                           #xPatchSize
    profile['height'] = array.shape[4]                                          #yPatchSize
    profile['photometric'] = 'RGB'
    
    print(f'updated raster meta data:\n{profile}') if verbose else None
    
    for row_num, col_num in __enumerate_patches(array):
        patch_path = fr'{path_to_save}_patch_{row_num}_{col_num}.{profile["driver"]}'
        
        with rio.open(patch_path,'w',**profile) as dst:
            dst.write(array[:,row_num,:,col_num,:])                             #array[bands,xPatchPosition,xPatchSize,yPatchPosition,yPatchSize] 
    
    print(f'Save Completed to {path_to_save}\n') if verbose else None
    

def Split_Tile(rasterfile:str,path_to_save:str,maxWidth:int,maxHeight:int,fileformat:str='PNG',verbose:bool=False):
    
    with rio.open(rasterfile) as src:
        tile_profile = {}
        tile_profile['meta'] = src.meta.copy()
    

        for row,height,col,width in __enumerate_tiles(src.meta['width'],src.meta['height'],maxWidth=maxWidth,maxHeight=maxHeight):
            tile_path = fr"{path_to_save}/tile_{col}_{row}.{fileformat.lower()}"
            tile = src.read(window=((row,height),(col,width)))                      #((rows),(cols))
                                                                                    #tuple containing the indexes of the rows at which
                                                                                    #the window starts and stops and the second is a tuple
                                                                                    #containing the indexes of the columns at which the window
                                                                                    #starts and stops
            if 0 in tile.shape:
                print(f'empty array found: {tile.shape} | x:{col},y:{row} | array: {tile}') if verbose else None
                continue                                                                                    
            elif len(tile.shape) not in (2,3):
                print(f'tile is of unexpected shape: {tile.shape} Should be either (x,y) or (b,x,y) |x: {row}  y:{col}') if verbose else None
                continue
            
            try:#this try block is mostly just for trouble shooting and should not be strictly necessary. 
               
                Min,Max = tile.min(), tile.max()
                if Max > 0:
                    print(f'tile is not all zeros: tile shape:{tile.shape} | {col} {row}: {Min} {Max}') if verbose else None
                else:
                    print(f'tile is all zeros: tile shape: {tile.shape} | {col} {row}: {Min} {Max}') if verbose else None
            except:
                print(f'min/max error: {tile.shape} | {col} {row}') if verbose else None
                print(f'meta:\n{tile_profile}\n')   if verbose else None
                continue            
            
            L = len(tile.shape)
            
            tile_profile['meta']['width'] = tile.shape[L-2]
            tile_profile['meta']['height'] = tile.shape[L-1]
            tile_profile['meta']['driver'] = fileformat        

            try:
                with rio.open(tile_path,'w',**tile_profile['meta']) as dst:
                    dst.write(tile)
            except:
                print(f'write error: tile shape: {tile.shape} | {col} {row}: {Min} {Max}') if verbose else None
                
