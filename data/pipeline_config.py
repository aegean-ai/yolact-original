"""
This is the default configuration of parameters for the pipeline
"""

config = {
    'pipeline':[], # Steps of the pipeline
    'verbose': 1, # If set to 0, does not print detailed information to screen
    'ds':None, # The Dataset
    'fs': None, # The Fileset
    'ts':None, # The Tileset
    'pef': 0.1, # Print-every-factor: Prints percentage completion after ever 'pef' factor of completion.
    'score_threshold':0.0, # During evaluation of inferences from raw data using Model: Minimum score threshold for detecting sidewalks.
    'det_score_threshold':0.0, # During generation of inference data from already generated inferenceJSON: Minimum sidewalk detection score threshold to include detection. 
    'rowsSplitPerTile':20, # Expected rows per tile
    'colsSplitPerTile':20, # Expected columns per tile
    'chipDimX':256, # Patch dimension X
    'chipDimY':256, # Patch dimension Y
    'tileDimX':5000, # Tile Dimension X
    'tileDimY':5000, # Tile Dimension Y
    'mWH': '5000,5000', # maximum Width/Height of tiles: used in genImageChips
    'fmts':'jpg,png,tif', # image formats to consider when reading files
    'sfmt':'jpeg', # Save format for generated chips in genImageChips
    'tvRatio':0.8, # The train-validation ratio used for generating training annotations
    'genImgs':1, # in genInferenceData Setting this to 0 skips saving images 
    'genGeoJSON':1, # in genInferenceData Setting this to 0 skips generation of geoJSON 
    'aws_sso_flag': False
}
