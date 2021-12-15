"""
Model Evaluation and Verification Functions
"""

import os
from pipeline_helper import *
import json
import pyproj
import numpy as np
from pycocotools import mask
import cv2
from raster2coco import Raster2Coco
import wget                                                                     #called within download_labels
from zipfile import ZipFile        

def genInferenceJSON():
    """
    Uses shell commands to run the eval.py for the configured parameters of the pipeline
    This generates inferences in a JSON file saved in <ds>/inferencesJSON_<ts>/
    """
    ds = config['ds']
    ts = config['ts']

    detPath = '%s/inferencesJSON_%s/'%(ds,ts)
    if(os.path.isdir(detPath)):
        vbPrint('Found dir: %s'%(detPath))
    else:
        vbPrint('Making dir: %s'%(detPath))
        os.mkdir(detPath)

    shCmd = 'cd .. && python eval.py --trained_model="%s" \
        --config=%s  --web_det_path="%s" \
        --score_threshold=%f --top_k=15  --output_web_json'%(
            config['trained_model'],
            config['config'],
            'data/'+detPath,
            config['score_threshold']
        )

    if(config['verbose']==0):
        shCmd += ' --no_bar'

    vbPrint('Initializing Inferences')
    os.system(shCmd)
    vbPrint('Completed')

def genInferenceData():
    """
    GenInferenceData Summary:
        Generates up to 2 types of inference data from the inferences JSON output by the model which is generated by genInferenceJSON():
        * Inference GeoJSON: a single geoJSON that has all the inferences as vector polygons. The file is a .geojson in <ds>/inferenceGeoJSON_<ts>
        * Inference Tiles: Merged inferences for each of the 5000x5000 tiles, each tile is available as a .png images in <ds>/inferenceTiles_<ts>
        Requirements
        * The required annotation file must be present in <ds>/annotations_<ts>
        * The inferences JSON generated by the model must be present in <ds>/inferencesJSON_<ts>/..
    
    GenInferenceData Algorithm:
        * The detections are available based on image_id. This id does not correspond to any tile or patch identification directly.
        * The annotations file has, for each image_id, the associated tile number and patch co-ordinates encoded into the image name.
        * We first read the entire annotations file and create a HashMap that stores, for each tile: a list of all patches' information.
        * The information for a patch includes its image_id, its tile number, and its patch co-ordinates
        * Once this hashMap is prepared, we are ready to iterate on a tile by tile basis:
            * For each tile, we have the patch info for all patches. For each patch:
                * We have the tile number which is the 'tile' itself
                * We have raw image present with the name 'tile_<patch Co-ordinate X>_<patch Co-ordinate Y>'
                * We have the associated patch detections with the 'image_id' in the inferences JSON
            * Essentially: We have a mapping between the tile and image_ids for all patches in the tile, for all tiles.
            * For each patch:
                * We have the detection vectors. This is in a COCO vector format and needs to be rasterized.
                * For the 256x256 patch, we have many detections each with a score.
                * We generate an empty image 256x256 image filled with 0s.
                * For each detection, we rasterize the detection and add it to the raster image if it is above the configurable threshold.
                * Sidewalk pixels are marked 255. If multiple detections mark the same pixel as a sidewalk, it is still 255.
                * 255 was chosen instead of 1 for the sake of simplicity: If viewed as an image, this raster shows the sidewalk in white
                  with a black background very clearly.
            * Essentially: For each 'image_id' (i.e. for each patch), we now have a rasterized inference.
            * For each tile:
                * We now go through the all of patches, pull their rasters from the HashMap.
                * Since the patch information includes the co-ordinates of a patch in a tile, we know the indexes of the 256x256 block in the 5120x5120 tile.
                * Using this info, we generate a single 5120x5120 raster for each tile. This is then cropped to 5000x500px similar to year-1.
                * Generation of the GeoJSON:
                    * Since we know the tile i.e. the path and name of the tile image, we can infer the path and name of the respective world file
                    * The Raster can be converted into polygon vectors
                    * For each detected polygon of >3 points, it is added into the geoJSON.
                    * The geoJSON file is initialized at the start of the code, opened for processing each tile and closed. 
        * Once all the tiles are processed, the appropriate line endings are appended to the geoJSON.

    GenInferenceData Notes:
        * The saving images can be skipped the argument '--genImgs=0'
        * The generation of geoJSON can be skipped with the argument '--genGeoJSON=0'
        * If both arguments are provided, nothing is saved. (Possible use-case is for debugging)
    """
    ds = config['ds']
    ts = config['ts']
    tilesDir = '%s/tiles_%s'%(ds,ts)
    inferenceTilesDir = '%s/inferenceTiles_%s'%(ds,ts)
    inferenceGeoJSONDir = '%s/inferenceGeoJSON_%s'%(ds,ts)
    geoJSONFilename = '%s_%s.geojson'%(ds,ts)
    genImgs = config['genImgs']
    genGeoJSON = config['genGeoJSON']

    # Making the dirs
    if(os.path.isdir(inferenceTilesDir)):
        vbPrint('Found dir: %s'%(inferenceTilesDir))
    else:
        vbPrint('Making dir: %s'%(inferenceTilesDir))
        os.mkdir(inferenceTilesDir)

    if(os.path.isdir(inferenceGeoJSONDir)):
        vbPrint('Found dir: %s'%(inferenceGeoJSONDir))
    else:
        vbPrint('Making dir: %s'%(inferenceGeoJSONDir))
        os.mkdir(inferenceGeoJSONDir)

    detFilePath = '%s/inferencesJSON_%s/%s'%(ds,ts,config['infJSON'])
    annFilePath = '%s/annotations_%s/%s'%(ds,ts,config['annJSON'])
    #annFilePath = '%s/inference_annotations/%s'%(ds,config['annJSON']) # For testing. Can be reverted once the data-pipeline part is completed.

    if(genGeoJSON):
        # Initialize GeoJSONFile
        with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'w+') as geoJSONf:
            geoJSONf.write('{"type": "FeatureCollection","name": "sidewalk_detections_%s_%s","features": ['%(ds,ts))

    # Generates a HashMap that stores the image tiles information
    # The program iterates through this to generate each tile
    tileMap = dict()
    vbPrint('Loading the annotations JSON')
    with open(annFilePath) as f:
        inpImageData = json.load(f)
        for img in inpImageData['images']:
            tile,i,j =  img['file_name'].split('.')[0].split('_')
            if(tile not in tileMap):
                tileMap[tile] = []
            tileMap[tile].append((img['id'],int(i),int(j)))
        del inpImageData
    
    vbPrint('Generated information for %i tiles'%(len(tileMap.keys())))

    vbPrint('Loading the inference JSON')
    with open(detFilePath) as f:
        infData = json.load(f)['images']
    
    
    patchShape = (config['patchDimX'],config['patchDimY'])
    infTileShape = (patchShape[0]*config['colsSplitPerTile'],
                    patchShape[1]*config['rowsSplitPerTile'])

    vbPrint('Expected Patch dimensions: %ix%i'%(patchShape[0],patchShape[1]))
    vbPrint('Expected Tiles dimensions: %ix%i'%(infTileShape[0],infTileShape[1]))

    # Creating a hashmap for the image dets with the image ids as keys
    # Each prediction is a merged image of many detections
    # Only detections above the threshold are selected to be merged
    vbPrint('Generating det masks with a threshold of %0.2f'%(config['det_score_threshold']))
    infPatchMap = dict()
    i=0
    n=len(infData)
    peFactor = config['pef']
    
    for img in infData:
        i += 1
        pred = np.zeros(patchShape)
        for det in img['dets']:
            if(det['score']>config['det_score_threshold']):
                detMask = mask.decode(det['mask'])
                pred[detMask==1]=255
        
        infPatchMap[img['image_id']] = pred

        # Prints progress at every 'pef' factor of completion
        # PEF stands for 'Print-Every factor'
        facComp = (i+1)/(n)
        if(facComp >= peFactor):
            vbPrint('Generating detection masks: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
            peFactor += config['pef']

    del infData
    
    vbPrint('Detection Masks made')
    vbPrint('Generating inference tiles from detection masks')

    # Rows and columns a tile is split into.
    # Default is 20,20
    rows = config['rowsSplitPerTile']
    cols = config['colsSplitPerTile']

    #### epsg: 3702 is close but thats Wyoming east
    #  
    # Initialize transformer
    #   From:     epsg projection 32111 - nad83 / new jersey
    #   To:       lat longs
    projectionTransformer = pyproj.Transformer.from_crs("epsg:32111","epsg:4326",always_xy=False)

    i = 0
    sidewalkCount = 0
    n = len(tileMap.keys())
    peFactor = config['pef']

    """
    For each tile, gets the detection masks and merges them into one image.
    It seems that many patches are missing in inferences. It is assumed these don't have any detections and are skipped.
    """
    for i,tile in enumerate(tileMap.keys()):
        # Skipping tile if unexpected number of patches exist for the tile in the annotations JSON
        expectedPatches = rows*cols
        if(len(tileMap[tile])!=expectedPatches):
            vbPrint('Found %i patches for tile %s. Skipping'%(len(tileMap[tile]), tile))
        else:
            # The inference tile. Initialized with 0. Masks are overlayed at their respective locations.
            infTile = np.zeros((rows*patchShape[0],cols*patchShape[1]),dtype=np.uint8)
            for patchID,row,col in tileMap[tile]:
                # Overlays the patch mask at the correct location in infTile
                # Skip any patches with no inference. It is assumed that no inference data is generated for patches with 0 detections.
                if(patchID in infPatchMap):
                    # Cropshift is the extra pixels to the left and top in the final row and column of the image inference
                    # This is only done due to the way the year-1 project was 'padded'.
                    # This will be need to be removed when correct padding is applied.
                    cropShiftX = 0
                    cropShiftY = 0
                    if(row==rows):
                        cropShiftX = 120
                    if(col==cols):
                        cropShiftY = 120
                    infTile[(row-1)*patchShape[0]:((row)*patchShape[0])-cropShiftX, (col-1)*patchShape[1]:((col)*patchShape[1])-cropShiftY] = infPatchMap[patchID][cropShiftX:,cropShiftY:]
            
            # Cropping image into the final tile dimension
            infTile = infTile[:config['tileDimX'],:config['tileDimY']]

            if(genGeoJSON):
                # Getting affine transform parameters from the world file
                with open('%s/%s.JGw'%(tilesDir,tile)) as worldFile:
                    rows = worldFile.read().split('\n')

                A = float(rows[0])
                D = float(rows[1])
                B = float(rows[2])
                E = float(rows[3])
                C = float(rows[4])
                F = float(rows[5])

                converter = Raster2Coco(None,None)
                binMask = np.zeros_like(infTile)
                binMask[infTile==255]=1
                vectors = converter.binaryMask2Polygon(binMask)
                JSONRows = ''
                for sidewalk in vectors:
                    # Skipping any triangles
                    if(len(sidewalk) >= 4):
                        sidewalkCount += 1
                        # Applying affine transform
                        '''
                        print(('-'*10)+'\nSidewalk pixels vector')
                        print(sidewalk)
                        
                        print(('-'*10)+'\nUTM Vector')
                        print(['[%f,%f]'%(
                                ((A*x) + (B*y) + C,
                                (D*x) + (E*y) + F)
                        ) for y,x in sidewalk])
                        '''
                        vecLi = ['[%f,%f]'%(
                            projectionTransformer.transform(
                                ((A*x) + (B*y) + C),
                                ((D*x) + (E*y) + F)
                            )[::-1]
                        ) for x,y in sidewalk]
                        #print(('-'*10)+'\nlat long vector')
                        #print(vecLi)

                        vecStr = '[[%s]]'%(','.join(vecLi))
                        #print(('-'*10)+'\nFinal vector string for ')
                        #print('%s'%(vecStr))
                        rowStr = ',\n{"type":"Feature","properties":{"objectid":%i}, "geometry":{ "type": "Polygon", "coordinates":%s}}'%(sidewalkCount,vecStr)
                        
                        # Skip comma for first sidewalk
                        if(sidewalkCount == 1):
                            rowStr = rowStr[1:]

                        JSONRows += '%s'%(rowStr)
                    
                if(JSONRows != ''):
                    with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'a+') as geoJSONf:
                        geoJSONf.write(JSONRows)

            if(genImgs):
                # Writing the image tile
                # File format kept as png because it has lossless compression and to work well with rasterio if needed.
                cv2.imwrite('%s/%s.png'%(inferenceTilesDir,tile),infTile)

            # Prints progress at every 'pef' factor of completion
            # PEF stands for 'Print-Every factor'
            facComp = (i+1)/(n)
            if(facComp >= peFactor):
                vbPrint('Generating inferences tile: %i/%i\t%0.2f %%'%(i+1,n,100*facComp))
                peFactor += config['pef']

    if(genGeoJSON):
        #print('TILE: ',tile)
        with open("%s/%s"%(inferenceGeoJSONDir,geoJSONFilename),'a+') as geoJSONf:
            geoJSONf.write('\n]}')

    if(genGeoJSON and genImgs):
        vbPrint('Inference GeoJSON and Tiles generated Successfully')
    elif(genGeoJSON):
        vbPrint('Inference GeoJSON generated Successfully')
    elif(genImgs):
        vbPrint('Inference Tiles generated Successfully')
    else:
        vbPrint('Inference Data ran successfully. Nothing saved due to arguments passed.')
