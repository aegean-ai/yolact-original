import fnmatch
import os
import data.split_merge_vector
import geopandas as gpd
#from shapely.geometry import Polygon
import shapely.wkt

imageryRootPath = '/workspaces/data/njtpa.auraison.aegean.ai/imagery/dvrpc-tiles/2015/TIFF_UTM'
geoJSONrootPath='/workspaces/data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4'
tileindexRootPath='/workspaces/data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4'
tiles=[]

# for root, dirs, files in os.walk(imageryRootPath):
#     for filename in fnmatch.filter(files, '*.tif'):
#         tiles.append(os.path.join(root, filename))

# print(tiles)
# print('Number of imagery geoTIFF tiles', len(tiles))

print("Loading feature and tile-index geoJSONs to data frames")

data.split_merge_vector._partitionGeoJSON(
    os.path.join(geoJSONrootPath, 'DVRPC_Sidewalks.geojson'),
    os.path.join(geoJSONrootPath, 'dvrpc-tile-index.geojson') 
)

# # discard the other columns as they are not needed
# features_gdf = features_gdf['geometry']

# print(features_gdf.head())

# print(tile_index_gdf.head())

# # Partitioning 
# print('Partitioning started .... ')

# # The geometry type is converted to simpler Polygons from Multi-Polygons

# mask_tiles_gdf = tile_index_gdf.explode()

# print(mask_tiles_gdf.head())


# # The following code has been commented out as it partitions the geoJSON without creating a spatial index. 
# # As such is 100x slower than the implementation with spatial indexing and almost infeasible for large problems such as this.

# # for ind  in mask_tiles_gdf.index:
# #     # Get the POLYGON from the data frame
# #     tile_geometry = mask_tiles_gdf['geometry'][ind[0]][0]
# #     shapely_tile_geometry = shapely.wkt.loads(tile_geometry.wkt)

# #     features_clipped = features_gdf.clip(shapely_tile_geometry)
    
    
# #     if not features_clipped.empty:
# #         print(mask_tiles_gdf['TILENAME'][ind])
# #         output_filename = os.path.join(geoJSONrootPath, mask_tiles_gdf['TILENAME'][ind] + '.geojson')
# #         features_clipped.to_file(output_filename, driver='GeoJSON')

# print('Creating a spatial index of the features ... ')

# spatial_index = features_gdf.sindex

# print('Creating the geoJSON partition for tiles that we  have features ...')

# for ind  in mask_tiles_gdf.index:
    
#     # Get the POLYGON from the data frame
#     tile_geometry = mask_tiles_gdf['geometry'][ind[0]][0]
#     shapely_tile_geometry = shapely.wkt.loads(tile_geometry.wkt)
    
#     possible_matches_index = list(spatial_index.intersection(shapely_tile_geometry.bounds))
#     possible_matches = features_gdf.iloc[possible_matches_index]
#     precise_matches = possible_matches[possible_matches.intersects(tile_geometry)]

#     # Clip is expensive so its done only after the geometries have been filtered. 
#     # Without clip the geometries extend slightly into adjacent tiles until they end. 
#     features_clipped = precise_matches.clip(shapely_tile_geometry)

#     if not features_clipped.empty:
#         #print(precise_matches)
#         print(mask_tiles_gdf['TILENAME'][ind])
#         output_filename = os.path.join(geoJSONrootPath, mask_tiles_gdf['TILENAME'][ind] + '.geojson')
#         features_clipped.to_file(output_filename, driver='GeoJSON')


