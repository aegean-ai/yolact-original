import fnmatch
import os
import webdataset as wds
import geopandas as gpd
import subprocess

imageryRootPath = '/workspaces/data/njtpa.auraison.aegean.ai/imagery/dvrpc-tiles/2015/TIFF_UTM'
tar_url = '/workspaces/data/njtpa.auraison.aegean.ai/pipeline-runs/project_root4/imageChips_ts4/dvrpc-2022-val.tar'
webdatasetRootPath='/workspaces/data/njtpa.auraison.aegean.ai/pipeline-runs/project_root6/imageTiles_ts6'

#tiles=[]

# for root, dirs, files in os.walk(imageryRootPath):
#     for filename in fnmatch.filter(files, '*.tif'):
#         tiles.append(os.path.join(root, filename))

# print(tiles)
# print('Number of imagery geoTIFF tiles', len(tiles))


print("Loading tar file")

dataset = wds.WebDataset(tar_url)

for sample in dataset:
    print('Chip name : ', sample['__key__'])
    print('Chip world file : ', sample['jgw'])
    print('Chip auxiliary metadata : ', sample['jpeg.aux.xml'])

    print('\n')


