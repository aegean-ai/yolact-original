import numpy as np
import os
import rasterio as rio    
from osgeo import gdal, osr, ogr        
from os import listdir
from os.path import isfile, join
from pathlib import Path
import fnmatch
import geopandas as gpd
import shapely



from centerline.converters import create_centerlines, get_ogr_driver
from centerline.exceptions import UnsupportedVectorType


from math import cos, sin, asin, sqrt, radians

def _partitionGeoJSON(geoJSONrootPath:str, tileindexRootPath:str, 
    partitionedGeoJSONRootPath:str,target_crs:str, sidewalk_annotation_threshold:int):
    """
    Partitions a large geoJSON vector file of features into multiple geoJSON files, each one bounded by a tile extent. Each tile extent is obtained from a separate tile index geojson file.   
    
    This function is called by genLabelTiles and the transformer argument is there to allow the generation of partitioned geoJSONs according to the CRS of the tile - both the tile and geoJSON must be 
    in the same CRS for the function to genLabelTiles function to work.  

    Inputs:
        Features geoJSON file that covers the whole region of the tileset. 
        Tile index geoJSON file containing the geography info for each tile in the imagery tileset.
        Target CRS of the partitioned geoJSONs
        
    Outputs:
        Split geoJSON feature files  in a subdirectorty called 'partitioned-geoJSON'- as many as the tiles in the tileset that contain features. 
    
    """

    print("Loading feature and tile-index geoJSONs to data frames")

    # read the input vector features file
    input_features_file = open(geoJSONrootPath)
    features_gdf = gpd.read_file(input_features_file)
    
    # read the input tile index file
    tile_index_file = open(tileindexRootPath)
    tile_index_gdf = gpd.read_file(tile_index_file)
 
    # print(features_gdf.head())
    # print(tile_index_gdf.head())

    # discard the other columns as they are not needed
    features_gdf = features_gdf['geometry']

    print(features_gdf.head())
    print(tile_index_gdf.head())

    # Partitioning 

    print('Partitioning started .... ')

    # The geometry type is converted to simpler Polygons from Multi-Polygons

    mask_tiles_gdf = tile_index_gdf.explode()

    # print(mask_tiles_gdf.head())

    # ---
    # The following code has been commented out as it partitions the geoJSON without creating a spatial index. 
    # As such is 100x slower than the implementation with spatial indexing and almost infeasible for large problems such as this.

    # for ind  in mask_tiles_gdf.index:
    #     # Get the POLYGON from the data frame
    #     tile_geometry = mask_tiles_gdf['geometry'][ind[0]][0]
    #     shapely_tile_geometry = shapely.wkt.loads(tile_geometry.wkt)

    #     features_clipped = features_gdf.clip(shapely_tile_geometry)
        
        
    #     if not features_clipped.empty:
    #         print(mask_tiles_gdf['TILENAME'][ind])
    #         output_filename = os.path.join(geoJSONrootPath, mask_tiles_gdf['TILENAME'][ind] + '.geojson')
    #         features_clipped.to_file(output_filename, driver='GeoJSON')

    # ---

    print('Creating a spatial index of the features ... ')

    spatial_index = features_gdf.sindex

    print('Creating the geoJSON partition for tiles that we  have features ...')
    tiles=[]
    for ind  in mask_tiles_gdf.index:
        
        # Get the POLYGON from the data frame
        tile_geometry = mask_tiles_gdf['geometry'][ind[0]][0]
        shapely_tile_geometry = shapely.wkt.loads(tile_geometry.wkt)
        
        possible_matches_index = list(spatial_index.intersection(shapely_tile_geometry.bounds))
        possible_matches = features_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(tile_geometry)]

        # Clip is expensive so its done only after the geometries have been filtered. 
        # Without clip the features extend slightly into adjacent tiles. 
        features_clipped = precise_matches.clip(shapely_tile_geometry)

        # Create a tile geoJSON only if there are features therein 
        # and if the number of features (in terms of rows in the geopandas data frame exceed a threshold

        if not features_clipped.empty:
             if len(features_clipped) > int(sidewalk_annotation_threshold):
            
                print(mask_tiles_gdf['TILENAME'][ind])
                output_filename = os.path.join(partitionedGeoJSONRootPath, mask_tiles_gdf['TILENAME'][ind] + '.geojson')

                # set the crs 
                features_clipped.set_crs(target_crs, allow_override=True)

                # reproject to target crs the coordinates of the features
                features_clipped = features_clipped.to_crs(target_crs)

                features_clipped.to_file(output_filename, driver='GeoJSON')
                tiles.append(output_filename)
        
    return tiles

def _centerline(runner, polygon_geojson_pathfile, line_geojson_pathfile):
    
    runner.invoke(
        create_centerlines, [polygon_geojson_pathfile, line_geojson_pathfile]
    )




def find_files(directory, pattern): 
    """
    Returns a list of the directories that the pattern is found
    """
    p = Path(directory)
    return list(p.glob('**/' + pattern))



def calc_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    m = 6371*1000 * c
    feet = int(m / 0.3048)
    
    return feet