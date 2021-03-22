"""
#Note: in laymens terms the intent of split_merge_tiff when splitting is to reshape an array
        such that it results in equally sized blocks of size (blockX,blockY).
        
        Each block is stored in an array indexed with a key representing its (blockXpos,blockYpos) position.
        such that the final array will have a shape (blockXpos,blockYpos,blockXdim,blockYdim)
        
        Number of Padding cells (paddingSize) will satisfy (0 <= xPad <= blockXdim-1), yPad = (0 <= yPad <= blockYdim-1). 
        
        Padded cells will have a constant value of p = 0 representing that they do not belong to original image.
        the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
        
        If the % of original cells in a padded block is < minScrapPercent then blocks along that edge will be discarded (ie 'scrapped')
        **There is no compensation for block at position NM and may result in minScrapPercent < % of original cells** 
        
                  <blockXdim>
      |  x0  | ... |  xN  |
    _  ______ _____ ______
      |      |     |     p|^
    y0|  b00 | b10 | b0N p|blockYdim
    __|______|_____|_____p|v
     .|                  .|
     .| original image   .|
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
        
    __Cutter()
            calls: None
        called in: SplitTiff
       Output flows to: SplitTiff
       
public function:
    SplitTiff()
            calls: __PaddTiff,__Cutter
        Called in: Unknown
         Output flows to: unKnown



"""

import numpy as np
import sys
import rasterio as rio
import rasterio.plot                            #---NOTE! For some reason rasterio does not automatically have all submodules available 
import rasterio.mask                            #         They must be imported specificaly
#from dataPipeline import vbPrint


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

def Open_Tiff(path_to_tiff: str, verbose:bool = False)-> np.array:
        """
            Note: convert / read the data into a numpy array,
                    return summary of information if needed
                    
                  ***   src.read() will read all bands  
                        currently all functions do accomodate multiband arrays
                        
                        indexes : list of ints or a single int, optional
                                  If `indexes` is a list, the result is a 3D array, but is
                                  a 2D array if it is a band index number.
                                                                                 
                        
                    
            inputs:
                path_to_tiff (string): File path to tiff image. 
                verbose (boolean): Used to print summary if needed
            
            outputs:
                tiffArray (np.arrary): returned array 
                
            
        """
        

        with rio.open(path_to_tiff) as src:
            tiffArray = src.read()                                              #return all bands (3D array) 
            profile = src.profile
        
            if verbose:                                                         #Print a summary of stats about Image
                print('-'*4,'Tiff Summary','-'*4)
                print(type(tiffArray))
                print(f'profile:\n{profile}')
                print('datatypes: ', {i: dtype for i, dtype in zip(src.indexes, src.dtypes)})
                print('color interp: ', src.colorinterp)
                print('tags: ',src.tags())
                print('bounds: ', src.bounds)
                print('extents: ', rio.plot.plotting_extent(src))
                print('res: ',src.res)
                print('shape: ',src.shape)
                print('array shape: ',tiffArray.shape)
                print('Transform:\n',src.transform)
                print('units: ', src.units)
                print(f'Array memory size: {__get_size(tiffArray)}\n')
            
        return tiffArray,profile

def __PadTiff(image:np.array,blockX:int,blockY:int,minScrapPercent:float,verbose:bool=False)-> np.array:
    """
    Note:     This function is called w/in SplitTiff() with the goal of providing padding along the 
                right & bottom edge of the input image such that the final dimensions of the output image are
                integer multiples of block dimension blockX & blockY
                
                The padding size will satisfy (0 <= paddingSize <= baseX-1) and will have a constant value 
                of (0) representing that the padded cells do not belong to the original image. 
                the dtype of the array is assumed to be unsigned (ex: uint8) and this is the lowest value
                
                if (Percent% of the original cells within a padded block) < minScrapPercent
                the blocks along that edge will be 'cut' (ie "scrapped")
        
    Inputs:        
        image (numpy.array): array that will be padded such that shape of the final array 
                             will have integer multiples of blockX & blockY.
        
        blockX (integer)    : the xdim of blocks after split
        blockY (integer)    : the ydim of blocks after split
        
        minScrapPercent (float) : min % of original cells required in edges.
                                  The intent is to prevent slivers of cells with not enough context
                                  resulting in reduced performance metrics. 
        
    Outputs:
        paddedImg (numpy.array): array such that its (xDim,yDim) are integer multiples of (blockX,blockY)
        
    """
    bands, imgX,imgY = image.shape
    
    scrapX = (imgX % blockX)/blockX                                             # % of cells that are along right edge
    scrapY = (imgY % blockY)/blockY                                             # % of cells that are along bottom edge
    
    padX = blockX-(imgX % blockX) if scrapX >= minScrapPercent else 0           #Amount of cells to add to right edge
    padY = blockY-(imgY % blockY) if scrapY >= minScrapPercent else 0           #Amount of cells to add to bottom edge
    
    xDim = (imgX//blockX + 1)*blockX if padX else (imgX//blockX)*blockX         #xDimension of output image
    yDim = (imgY//blockY + 1)*blockY if padY else (imgY//blockY)*blockY         #yDimension of output image
    
    paddedImg = np.pad(array = image,                                           #Apply padding to input image
                       pad_width = ((0,0),(0,padX),(0,padY)),                   #Pad only bottom & right edges 
                       mode='constant',                                         #Fill with a constant value
                       constant_values = 0)                                     #Contant value of 0
    
    paddedImg = paddedImg[:,0:xDim,0:yDim]                                      #Crop image, if padX,padY are 0 then this results in
                                                                                #  'scrapping' that edge and losing those cell values
    
    if verbose:                                                                 #Return summary of results 
        print(f"{'-'*5} 'Summary of Padding/Scrapping' {'-'*5}")
        print('xEdge has been padded') if xDim > imgX else print('xEdge has been scrapped') if xDim < imgX else print('xEdge has not been modified')  
        print('yEdge has been padded') if yDim > imgY else print('yEdge has been scrapped') if yDim < imgY else print('yEdge has not been modified')
        print(f'input image shape: {image.shape}')                                            
        print(f'output image shape: {paddedImg.shape}\n')
    
    return paddedImg
    

            

def Split_Tiff(image:np.array,blockX:int,blockY:int,minScrapPercent:float,verbose:bool = False)-> np.array:
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
        
        blockX (integer)    : the xdim of blocks after split
        blockY (integer)    : the ydim of blocks after split
        
        minScrapPercent (float) : min % of original cells required for blocks on edge.
                                  The intent is to prevent slivers of cells with not enough context
                                  resulting in reduced performance metrics. 
         
    Outputs:
        blocks (dictionay): reshaped array split into blocks. 
                            final shape: (#of blocks, baseX,baseY)       
    """
    
    
    paddedImg = __PadTiff(image,blockX,blockY,minScrapPercent,verbose)          #get padded image

    bands, xDim,yDim = paddedImg.shape                                          #padded image dimensions
                                
    xSize,ySize = xDim//blockX,yDim//blockY                                     #block x,y arrangment
    
    newShape = (bands,xSize,blockX,ySize,blockY)                                #band,xblockPosition,xblockSize,yblockPosition,yblockSize
    
    blocks = np.reshape(paddedImg,newShape)                                     #reshape into blocks

    
    
    if verbose:                                                                 #print summary
        print(f"{'-'*5} Summary of Splitting {'-'*5}")
        print(f'Input Type: {type(image)}')
        print(f'Input Shape: {image.shape}')
        print(f'Input Memory Size: {__get_size(image)},{image.nbytes}')
        print(f'Output Type: {type(blocks)}')
        print(f'Shape Reference: (bands,xSize,blockX,ySize,blockY)')
        print(f'intended shape: {newShape}')
        print(f'Ouput Shape: {blocks.shape}')
        print(f'Output Memory Size: {__get_size(blocks)},{blocks.nbytes}\n')
        print('memory %change: {0:+8.4f}%\n\n'.format((blocks.nbytes/image.nbytes - 1)*100))
    return blocks
    




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
    mergedImg = Split_Tiff(mergedImg,mergeXdim,mergeYdim,minScrapPercent,verbose)                #Splits & Padds to a new size
    
    if verbose:
        print(f'\nmerged shape: {mergedImg.shape}')

    return mergedImg
    

def __enumerate_Blocks(array):
    #<Image Index (Integer)>_<row number>_<col number>
    #shp = (bands,xBlockPosition,xBlockSize,yBlockPosition,yBlockSize) 
    
    xBlocknum = array.shape[1]
    yBlocknum = array.shape[3]
    
    for xBlockPosition in range(xBlocknum):
        for yBlockPosition in range(yBlocknum):
            yield xBlockPosition,yBlockPosition

    
def Save_Tiff(array:np.array,save_directory:str,save_fmt:str,image_index:str,profile:dict,verbose:bool=False):
    #image profiles: https://rasterio.readthedocs.io/en/latest/topics/profiles.html
    #Supported drivers: https://gdal.org/drivers/raster/index.html
    #Saving Images: https://rasterio.readthedocs.io/en/latest/topics/writing.html#
    
    
    image_index = image_index.rsplit('.')[0]
    path_to_save = fr'{save_directory}/{image_index}'
    
    profile['driver'] = 'JPEG' if save_fmt == 'jpg' else save_fmt.upper()
    profile['count'] = array.shape[0]                                           #band count
    profile['width'] = array.shape[2]                                           #xBlockSize
    profile['height'] = array.shape[4]                                          #yBlockSize
    profile['photometric'] = 'RGB'
    
    print(f'updated profile: {profile}') if verbose else None
    
    for row_num, col_num in __enumerate_Blocks(array):
        block_path = fr'{path_to_save}_{row_num}_{col_num}.{profile["driver"]}'
        
        with rio.open(block_path,'w',**profile) as dst:
            dst.write(array[:,row_num,:,col_num,:])                             #array[bands,xBlockPosition,xBlockSize,yBlockPosition,yBlockSize] 
    
    print('Save Completed\n') if verbose else None
    
    