"""
Info:
    Coordinate system: EPSG::4326     see http://epsg.io/4326
    gdal : Geospatial Data Abstraction Library
      ogr: OpenGis simple features Reference implementation
    http://gismanual.com/relational/99-049_OpenGIS_Simple_Features_Specification_For_SQL_Rev_1.1.pdf
    
    
Sources:
    https://desktop.arcgis.com/en/arcmap/10.3/manage-data/geodatabases/feature-class-basics.htm
    https://gis.stackexchange.com/questions/346895/create-a-new-raster-tiff-file-which-is-masked-based-on-the-geojson-file
    https://stackoverflow.com/questions/47846178/how-to-rasterize-simple-geometry-from-geojson-file-using-gdal
    https://gis.stackexchange.com/questions/211611/python-gdal-handling-big-rasters-and-avoid-memoryerrors
    https://gis.stackexchange.com/questions/348510/generate-an-empty-large-raster-using-gdal
    https://gdal.org/programs/gdal_polygonize.html

    https://morphocode.com/using-ogr2ogr-convert-data-formats-geojson-postgis-esri-geodatabase-shapefiles/

    command line examples: 
        ogr2ogr -f "ESRI Shapefile" sidewalks_fromgeoJSON1.shp "sidewalks.geoJSON"
        ogr2ogr -f "ESRI Shapefile" vectors/sidewalks_fromgeoJSON2.shp "geoJSON/DVRPC_Sidewalks.geojson"


(y,x) = (lat,lng)
"""


import json
import time
import os

import ogr
import osgeo.gdal as gdal
from calcDist import dist, getPoints


#used in __convert_dims
def __checkPassed(a,b,c,verbose):
    """
    Notes:
        internal logic to determine if correct information was passed in __convert_dims
            when finding cellsize -> arrdim or arrdim -> cellsize conversion
        
    Inputs:
              a: cells
              b: arrdims
              c: lnglat_deltas
        verbose: whether to print status message
        
    Outputs:
        bool: return a boolean specifiying if check was passed or not. 
    """
    
    if (c is None) and (type(c) not in(tuple,list)) or (len(c) !=2):
        print(f'latlong: {c} must be a tuple or list of len 2') if verbose else None
        return False
    if (a is None) or (b is not None):
        print('error: parameter mismatch, can only pass: cells or arrdims not both') if verbose else None
        return False
    if (type(a) not in (tuple,list)) or (len(a) != 2):
        print(f'error: parameter {a} is not a tuple or list of length 2') if verbose else None
        return False
    
    return True

    
def __convert_dims(cells=None,arrdims=None,lnglat_deltas=None,verbose=False):               #Used in geojson_to_geotiff 
    """ 
    Notes:
        used to find raster-array dimensions (remember rasters are a grid)  from cell sizes, or vice versa
        cells: (x,y) = (lng,lat)
        
    Inputs:
              cells (tuple): cell sizes (x,y)
            arrdims (tuple): array shape
      lnglat_deltas (tuple): longitude & latitude deltas of region extents (delta_longitude,delta_latitude)
             verbose (bool): whether to return status message
             
    Outputs:
        ydim,xdim: either returns arraydims (int,int) or cellsizes (float,float)
        
    """
    ydim,xdim = None,None
    if __checkPassed(cells,arrdims,lnglat_deltas,verbose):                      
        message = 'returning arr dims:'
        xdim = int(lnglat_deltas[0]/cells[0])                                   #xarr = dlng/xcell
        ydim = int(lnglat_deltas[1]/cells[1])                                   #yarr = dlat/ycell
    elif __checkPassed(arrdims,cells,lnglat_deltas,verbose):
        message = 'returning cells'
        xdim = lnglat_deltas[0]/arrdims[0]                                      #xcell = dlng/xarr
        ydim = lnglat_deltas[1]/arrdims[1]                                      #ycell = dlat/yarr
    else:
        message = f'parameter not compatable>  cells:{cells} | arrdims:{arrdims} | deltas:{lnglat_deltas}'
        
    if verbose:
        print(message)
        
    return ydim,xdim

def __time_this(start=None,crash:bool=False,verbose:bool=False):                #Used within geojson_to_geotiff since it is resource/time intensive
    
    if start is None:
        start = time.perf_counter()
        return start
    else:
        end = time.perf_counter()
        
        delta = end-start
        ms = round((delta%1)*1000,2)
        
        ty_res = time.gmtime(delta)
        res = f"{time.strftime('h:%H m:%M s:%S',ty_res)} ms:{ms}"

        print(f'function took: {res}') if verbose else None
        
        return start,res
        


def __crashreport(start,res,file):                                                  #used for geojson_to_geotiff function 
        print('error occured')
        name = file.rsplit('.',1)[0]
        size = '{0:6.2f}'.format(os.path.getsize(file)/10**6)                       #find current file size of geotif @ time of crash
        
        with open(f'crash_time_{name}_{start}.txt','w') as report:
            report.write(f'writting to {name} took: {res} | file size: {size} GB')  #write a report file in the event of a crash (overrides previous)
            

def __get_output_filename(input_file,output_folder,output_file,frmt):               #Function for splicing input name/creating output filename

    if output_file is None:                                                         #if no output filename is provided, the default is to use input name
        input_name = input_file.rsplit('.',1)[0].rsplit('/',1)[1]
        output_file = f'{output_folder}/{input_name}.{frmt}'
    
    else:
        output_file = f'{output_folder}/{output_file}.{frmt}'
    
    return output_file
    
#---------------------------------------Functions:


def geojson_to_geotiff(geojson_file: str, destination_folder:str, raster_file: str = None, verbose:bool=False) -> None: 
    
    """
    Notes:
        ***IMPORTANT: this is a very resource intensive task. Can take many hours to complete depending on the size of region / number of features
                       a file with 351514 features took h:07 m:34 s:12 ms:733.56 to complete and resulted in a geoTif file that was 10GB in size.     
                       
        sources used:
            https://gis.stackexchange.com/questions/346895/create-a-new-raster-tiff-file-which-is-masked-based-on-the-geojson-file
            https://stackoverflow.com/questions/47846178/how-to-rasterize-simple-geometry-from-geojson-file-using-gdal
            https://gis.stackexchange.com/questions/211611/python-gdal-handling-big-rasters-and-avoid-memoryerrors
            https://gis.stackexchange.com/questions/348510/generate-an-empty-large-raster-using-gdal
            

            
    Inputs:
                geojson_file (str): the geojson file to convert to geoTiff
          destination_folder (str): the folder to store geoTiff
      raster_file (optional) (str): file name to use. Default is to use source filename
                    verbose (bool): whether to print information
                  
    Outputs:
            None: does not return anything, but results in a new raster file being created.
    
    """
    
    
    if verbose:
        with open(geojson_file) as gjson:                               #This reads geoJSON file to get basic information
            data = json.load(gjson)           
        print(f'json keys: {data.keys()}')
        print(f'Num of Features: {len(data["features"])}')


    source_ds = ogr.Open(geojson_file)                                  #This actually reads the geoJSON to rasterize     
    source_layer = source_ds.GetLayer()

                                             
    extents = tuple(source_layer.GetExtent())                           #(longitutde min, longitude max, lattitude min, lattitude max)
    x_min,x_max,y_min,y_max = extents            


    bl,ul,ur = getPoints(extents)                                       #pt: (x,y) = (lng,lat)
    dlng =  extents[1] - extents[0]                                     #delta longitutude
    dlat =  extents[3] - extents[2]                                     #delta latitude
    
    h,w = dist(bl,ul,'km'), dist(ul,ur,'km')                            #Hight & Width in km of Region
    rwh = w/h                                                           #Width/Height ratio
    

    if rwh >= 1:                                                        #allow cell sizes to maintain aspect ratio of region
        ycelldim = 1.00*10**-6
        xcelldim = rwh*ycelldim
    else:
        xcelldim = 1.00*10**-6
        ycelldim = rwh*xcelldim        

    cols,rows = __convert_dims(cells=(xcelldim,ycelldim),
                               lnglat_deltas=(dlng,dlat),
                               verbose=verbose)
    
    
    raster_file = __get_output_filename(input_file = geojson_file,
                                       output_folder=destination_folder,
                                       output_file=raster_file,
                                       frmt = 'tiff')
    
    if verbose:
        print(f'extents: {extents}') 
        print(f'region size: Height: {h} km | Width: {w} km')
        print(f'w/h ratio: {w/h}')
        print('cell dimensions: Height: {0:5.3E}m, Width: {1:5.3E}m'.format(ycelldim,xcelldim))
        print('cell spatial reference: Height: {0:5.3f}m, Width: {1:5.3f}m'.format(h*1000*ycelldim,w*1000*xcelldim))
        print(f'cols,rows: {cols},{rows}')
        print(f'approximate tiles: {(cols*rows)/(5000**2)}')
        print(f'out file name: {raster_file}')
        
    
    start =__time_this()                                                        #Creates an empty raster to burn vector data into
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_file,cols,rows,1,gdal.GDT_Byte,options=['COMPRESS=LZW', 'TILED=YES','BIGTIFF=YES'])
    __time_this(start)
    
    target_ds.SetGeoTransform([ul[0], xcelldim, 0, ul[1], 0, -ycelldim])        #Sets the affine matrix
    
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)                                                      #Sets the 'No data' value to 0 (for when gdal encounters an NA, Null etc)



    start =__time_this()
    try:
        gdal.RasterizeLayer(target_ds, [1], source_layer,burn_values=[255],options=['ALL_TOUCHED=TRUE','BIGTIFF=YES','COMPRESS=LZW'])
        if verbose:
            print('--- geojson_to_geotiff completed ---')
    except:
        start,res = __time_this(start)
        __crashreport(start,res,raster_file)    
    finally:
        start,res = __time_this(start)
      


#----------------------------These set of functions dont directly 'use python' but instead pass a gdal/ogr command to system command line.

def geotif_to_shape(geotif_file:str,destination_folder:str,shape_file:str=None,verbose:bool=False)->None:
    
    """
    Notes:
        ***IMPORTANT: This conversion can take a long time. 
    
    Inputs:
               geotif_file (str): raster file (GTiff) to convert to vector file (shp)
        destination_folder (str): the folder to store vector file (shp)
     shape_file (optional) (str): file name to use. Default is to use source filename
                  verbose (bool): whether to print status messages
                  
    
    Outputs:
            None: does not return anything, but results in a new raster file being created.
   
    
    """
    
    output_file = __get_output_filename(input_file = geotif_file,
                                        output_folder=destination_folder,
                                        output_file=shape_file,
                                        frmt = 'shp')                 
    
   #command = f"gdal_polygonize.py <raster_file> <out_file>"
    
    command = f"gdal_polygonize.py {geotif_file} {output_file}"
    start =__time_this()
    os.system(command)
    start,res = __time_this(start)
    
    if verbose:
        print(f'output file: {output_file}')
        print(f'command: {command}')
        print('--- geotif_to_shape completed ---')           




def geojson_to_shape(geojson_file: str, destination_folder:str, shape_file: str = None, verbose:bool=False)->None:
    """
    Notes:
        This is a simple conversion and should not take a lot of time/resources
    
    """
    
    output_file = __get_output_filename(input_file = geojson_file,
                                        output_folder=destination_folder,
                                        output_file=shape_file,
                                        frmt = 'shp')                 
    
    #command = ogr2ogr -f <output format> <destination filename> <source filename>
    command = f"ogr2ogr -f GeoJSON {output_file} {geojson_file}"
    os.system(command)
    
    if verbose:
        print(f'output file: {output_file}')
        print(f'command: {command}')
        print('--- geojson_to_shape completed ---')


def shape_to_geojson(shape_file:str,destination_folder:str,geojson_file:str=None,verbose:bool=False)->None:
    """
    Notes:
        This is a simple conversion and should not take a lot of time/resources
    
    """
    output_file = __get_output_filename(input_file = shape_file,
                                        output_folder=destination_folder,
                                        output_file=geojson_file,
                                        frmt = 'geoJSON')                 
    
    #command = ogr2ogr -f <output format> <destination filename> <source filename>
    command = f"ogr2ogr -nlt LINESTRING -skipfailures  {output_file} {shape_file}"
    os.system(command)
    
    if verbose:
        print(f'output file: {output_file}')
        print(f'command: {command}')
        print('--- shape_to_geojson completed ---')
        
        
