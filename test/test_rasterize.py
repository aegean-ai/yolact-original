import pprint
import rasterio
from rasterio import features
import geopandas as gpd
import os
import numpy as np
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union
from rasterio.mask import raster_geometry_mask
import math
# with rasterio.open('./segmentation/models/yolact-original/test/13547682814_f2e459f7a5_o.png') as src:
#     blue = src.read(3)

# mask = blue != 255
# shapes = features.shapes(blue, mask=mask)
# pprint.pprint(next(shapes))

# # Output
# # pprint.pprint(next(shapes))
# # ({'coordinates': [[(71.0, 6.0),
# #                    (71.0, 7.0),
# #                    (72.0, 7.0),
# #                    (72.0, 6.0),
# #                    (71.0, 6.0)]],
# #   'type': 'Polygon'},
# # 253)

# shapes = features.shapes(blue, mask=mask, transform=src.transform)

# image = features.rasterize(
#             ((g, 255) for g, v in shapes),
#             out_shape=src.shape,
#             transform=src.transform)

# with rasterio.open(
#         './segmentation/models/yolact-original/test/rasterized-results.tif', 'w',
#         driver='GTiff',
#         dtype=rasterio.uint8,
#         count=1,
#         width=src.width,
#         height=src.height) as dst:
#     dst.write(image, indexes=1)


# -----

vector_gpd = gpd.read_file('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/labelTiles_ts4/UTM_X24_Y76.geojson')
vector_gpd = vector_gpd.set_crs('EPSG:6565', allow_override=True)
buffered_vector_gpd = vector_gpd.copy()

 # Buffer the features    (with buffer_value=4)    
# buffered_vector_gpd.geometry = vector_gpd['geometry'].buffer(4)
# # overwrite the per tile geoJSON with buffered geometries
# buffered_vector_gpd.to_file('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/labelTiles_ts4/UTM_X24_Y76.geojson', driver = 'GeoJSON')

# Get list of geometries for all features in vector file
geom = [shapes for shapes in buffered_vector_gpd.geometry]
# geom_is_valid = rasterio.features.is_valid_geom(geom)
# print(geom_is_valid)

# import gdal
# raster = gdal.Open("raster.tif", gdal.GA_Update)
# vector = "../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4/UTM_X24_Y76.geojson"
# OPTIONS = gdal.RasterizeOptions(burnValues=[255])
# gdal.Rasterize(raster, vector, options=OPTIONS)
# driver = gdal.GetDriverByName('GTiff')

from rasterio.warp import calculate_default_transform, reproject, Resampling

dst_crs = 'EPSG:6565'

with rasterio.open('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4/UTM_X24_Y76.tif') as src:
    
    # template_raster = src.read(1)
    # out_image, out_transform = rasterio.mask(src, geom, crop=True, nodata=0)

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4/reprojected_UTM_X24_Y76.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

with rasterio.open('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4/UTM_X24_Y76.tif') as src:

    with rasterio.Env():

        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src.profile

        # # And then change the band count to 1, set the
        # # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rasterio.uint8,
            count=1, # single band
            compress='lzw'
        )

        # get extent of all polygons
        xmin,ymin,xmax,ymax = buffered_vector_gpd.geometry.total_bounds.tolist()

        # set raster resolution and use that to get width and height
        res = 1
        width = 7000 #math.ceil((xmax-xmin)/res)
        height = 7000 # math.ceil((ymax-ymin)/res)

        # get the affine transformation for our empty raster
        transform = rasterio.transform.from_origin(west=xmin,north=ymax,xsize=res,ysize=res)

        mask = rasterio.features.rasterize(
            [(x.geometry, 255) for i, x in buffered_vector_gpd.iterrows()],
            transform=transform,
            out_shape=src.shape,
            default_value = 255,
            all_touched = True
        ) 
        print(sum(mask))

        label_tile_filepath = os.path.join('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/labelTiles_ts4/test_UTM_X24_Y76_label.tif') 

        with rasterio.open(label_tile_filepath, 'w', **profile) as masktif:
            masktif.write(mask.astype(rasterio.uint8), 1)


# # read in polygon file
# gdf = gpd.read_file('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/labelTiles_ts4/UTM_X24_Y76.geojson')

# # get extent of all polygons
# xmin,ymin,xmax,ymax = gdf.geometry.total_bounds.tolist()

# # set raster resolution and use that to get width and height
# res = 1 # just an arbitrary resolution i chose for my EPSG:4326 polygons
# width = int((xmax-xmin)/res)
# height = int((ymax-ymin)/res)

# # get the affine transformation for our empty raster
# transform = rasterio.transform.from_origin(xmin,ymax,res,res)

# # create rasterio dataset with empty raster
# with rasterio.open('new.tif','w',driver='GTiff',height=height,width=width,
#                    count=1,dtype='uint8',crs='EPSG:6565',
#                    transform=transform) as empty:

#     # loop through polygon geodataframe creating rasters from polygons
#     for ind,row in gdf.iterrows():
#         mask,mask_tform,window = raster_geometry_mask(empty,[row['geometry']],invert=True)
#         mask = mask.astype('uint8')*255 # "Values" value, 0 elsewhere
        
#         # write out mask as a raster, use metadata from empty dataset for parameters
#         # outpath = f'raster.tif' # raster0.tif, raster1.tif, etc
#         # with rasterio.open(outpath,'w',**empty.meta) as dst:
#         empty.write(mask, indexes=1)

#     shapes = features.shapes(empty.read(1), mask=mask)


# # out_shape = template_raster.shape
# # transform = src.transform

# # # mask = rasterio.features.geometry_mask(geom, out_shape, transform, all_touched=True, invert=True)

# # # mask = mask != True

# # # print(mask)

# # mask = rasterio.features.rasterize(
# #             geom,
# #             out_shape = template_raster.shape,
# #             fill = 0,
# #             out = None,
# #             transform = src.transform,
# #             all_touched = False,
# #             default_value = 255,
# #             dtype = rasterio.uint8
# #         )

# # print(out_image)
# # meta = src.meta.copy()
# # # meta.update(nodata=0, count=1, dtype=rio.uint8)
# # label_tile_filepath = os.path.join('../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/labelTiles_ts4/test_UTM_X24_Y76_label.tif') 
# # with rasterio.open(label_tile_filepath, 'w', **meta) as dst:
# #     dst.write(out_image, indexes=1) # write band 1

# # raster_path = '../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4/UTM_X24_Y76.tif'
# # geojson_path = '../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/region_fs4/test-tile-index.geojson'
# # output_path= '../data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/labelTiles_ts4'
# # filename = 'test_UTM_X24_Y76_label.tif'

# # def generate_mask(raster_path, shape_path, output_path, file_name):
    
# #     """Function that generates a binary mask from a vector file (shp or geojson)
    
# #     raster_path = path to the .tif;

# #     shape_path = path to the shapefile or GeoJson.

# #     output_path = Path to save the binary mask.

# #     file_name = Name of the file.
    
# #     """
    
# #     #load raster
    
# #     with rasterio.open(raster_path, "r") as src:
# #         raster_img = src.read()
# #         raster_meta = src.meta
# #         src = src.to_crs('epsg:6565')
    
# #     #load o shapefile ou GeoJson
# #     train_df = gpd.read_file(shape_path)
    
    
# #     #Verify crs
# #     if train_df.crs != src.crs:
# #         print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,train_df.crs))
        
        
# #     #Function that generates the mask
# #     def poly_from_utm(polygon, transform):
# #         poly_pts = []

# #         poly = cascaded_union(polygon)
# #         for i in np.array(poly.exterior.coords):

# #             poly_pts.append(~transform * tuple(i))

# #         new_poly = Polygon(poly_pts)
# #         return new_poly
    
    
# #     poly_shp = []
# #     im_size = (src.meta['height'], src.meta['width'])
# #     for num, row in train_df.iterrows():
# #         if row['geometry'].geom_type == 'Polygon':
# #             poly = poly_from_utm(row['geometry'], src.meta['transform'])
# #             poly_shp.append(poly)
# #         else:
# #             for p in row['geometry']:
# #                 poly = poly_from_utm(p, src.meta['transform'])
# #                 poly_shp.append(poly)

# #     mask = rasterio.features.rasterize(shapes=poly_shp,
# #                      out_shape=im_size, transform=src.transform)
    
# #     #Salve
# #     mask = mask.astype("uint8")
    
# #     bin_mask_meta = src.meta.copy()
# #     bin_mask_meta.update({'count': 1})
# #     os.chdir(output_path)
# #     with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
# #         dst.write(mask * 255, 1)

# # generate_mask(raster_path=raster_path, shape_path=geojson_path, output_path=output_path, file_name=filename)
