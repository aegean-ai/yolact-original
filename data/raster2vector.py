"""
Notes:
    raster2vector allows raster files (jpg,png,GeoTif) to be converted to a vector file (.shp)
    it does this using rasterio to generate vector shapes and geopandas to save into correct file type (.shp)
    
    the supported vector files can be found using:
        import fiona
        print(fiona.supported_drivers)
      
        output:
        {'AeronavFAA': 'r', 'ARCGEN': 'r', 'BNA': 'rw', 'DXF': 'rw', 'CSV': 'raw', 'OpenFileGDB': 'r', 'ESRIJSON': 'r', 'ESRI Shapefile': 'raw', 'GeoJSON': 'raw', 
        'GeoJSONSeq': 'rw', 'GPKG': 'raw', 'GML': 'rw', 'OGR_GMT': 'rw', 'GPX': 'rw', 'GPSTrackMaker': 'rw', 'Idrisi': 'r', 'MapInfo File': 'raw', 'DGN': 'raw', 'OGR_PDS': 'r', 
        'S57': 'r', 'SEGY': 'r', 'SUA': 'r', 'TopoJSON': 'r'}
    
    geoPandas default is 'ESRI Shapefile'

Sources:
    https://geopandas.readthedocs.io/en/latest/docs/reference/api/geopandas.GeoDataFrame.to_file.html
    https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons
    https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.to_file.html#geopandas.GeoDataFrame.to_file
    https://geopandas.org/docs/reference/geodataframe.html

"""

import rasterio as rio
from rasterio.features import shapes
import geopandas as gpd

def convert_raster2vector(raster_file_name:str,vector_file_name:str,verbose=False)->None:
    """
    Warning!:
        This function currently can only be used to convert tiles/patche-images. 
        It needs to be updated to handle world-images
        
    Inputs: 
        raster_file_name (str): filename of raster to be converted
        vector_file_name (str): filename of vector file to be created
                verbose (bool): whether to print summary/debug info     
    
    outputs:
        None: Results in a vector file being created with the desired file name
    
    Sources:
        https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons
    
    """
    
    
    #---Get vector properties:
    with rio.open(raster_file_name) as src:
        transform = src.transform 
        image = src.read()                                                      #Warning!: This is the line that causes trouble if there are too many pixels in image. 
        results = [{'properties':{'raster_val':v},'geometry':s} for i,(s,v) in enumerate(shapes(image, mask=None,transform=transform))]
           
        
    #---Save to Shapefile:
    gdf_vectors = gpd.GeoDataFrame.from_features(results)                       #Convert to geoDataframe
    gdf_vectors.to_file(vector_file_name)                                       #Save shapefile

    if verbose:                                                                 #Print summary
        print('transform:\n{transform}')
        print(f'results length: {len(results)}')
        print(f'result peek: {results[0]}')
        print(f'geoDataframe:\n{gdf_vectors.head}')