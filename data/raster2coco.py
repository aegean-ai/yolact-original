from osgeo import gdal, gdalnumeric
from skimage import measure
import numpy as np
import datetime
from shapely.geometry import Polygon

# Disabling speedups because of error when using Cython speedups
from shapely import speedups
speedups.disable()

def raster2array(rasters,band_no=1):
    """
    Arguments:
    rasters            A gdal Raster object
    band_no         band numerical order
    Example :
    raster = gdal.Open(rasterfn)
    raster2array(raster,1)
    """
    bands = rasters.RasterCount
    if band_no>0 and band_no <=bands:
        band = rasters.GetRasterBand(band_no)
        array = band.ReadAsArray()
    else:
        array = rasters.ReadAsArray()

    return array

# This function will convert the rasterized clipper shapefile
# to a mask for use within GDAL.
def imageToArray(i):
    
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    
    a=gdalnumeric.fromstring(i.tobytes(),'b')

    a.shape=i.im.size[1], i.im.size[0]
    
    return a

#
def coord2pixelOffset(geotransform, x, y):
    
    """
    Arguments:
    
    geotransform  - A gdal transform object
    x               world coordinate x
    y               world coordinate y
    return  pixel position in image
    Example :
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    coord2pixel(geotransform,xCoord,yCoord)
    """

    #left top
    originX = geotransform[0]
    originY = geotransform[3]

    #pixel resolution
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]

    #ax rotate (here not used)
    rotateX = geotransform[2]
    rotateY = geotransform[4]


    xOffset = int((x - originX) / pixelWidth)
    yOffset = int((y - originY) / pixelHeight)
    return xOffset, yOffset

def getUniqueValue(inL, column):
        # intilize a null list
        unique_list = []

        if (len(inL) == 0): return unique_list
        count = len(inL[0])
        if column > count: return unique_list
        # traverse for all elements
        for x in inL:
            # check if exists in unique_list or not
            if x[column - 1] not in unique_list:
                unique_list.append(x[column - 1])
        return unique_list

def pixeloffset2coord(geoTransform,pixel_xOffset,pixel_yOffset):
    """
    geoTransform: a gdal geoTransform object
    pixel_xOffset:
    pixel_yOffset:
    return:  coords
    """

    #left top
    originX = geoTransform[0]
    originY = geoTransform[3]

    #pixel resolution
    pixelWidth = geoTransform[1]
    pixelHeight = geoTransform[5]


    # calculate coordinates
    coordX = originX+ pixelWidth*pixel_xOffset
    coordY = originY+pixelHeight*pixel_yOffset

    return coordX,coordY


INFO = {
    "description": "Sidewalk Validation Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2022,
    "contributor": "Aegean AI",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'sidewalk',
        'supercategory': '',
    }
]

class Raster2Coco():
    
    def __init__(self, files_array, dir, has_gt):
        
        self.files_array = files_array
        self.input_dir = dir
        self.has_gt = has_gt

    def createJSON(self):

        self.cocoJSON = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
       
        return(self.cocoJSON)

    def genDirAnnotations(self, band_number):

        annotation_idx = 1
        for img_idx, labelFile in enumerate(self.files_array):
            
            raster = gdal.Open('%s/%s'%(self.input_dir,labelFile))
            raster_array = raster2array(raster, band_number)

            self.genImgJSON(raster_array, labelFile,  img_idx+1, 1, annotation_idx + 10000 * img_idx, annotation_threshold=7)
        
        return(self.cocoJSON)
    
    def genImgJSON(self, raster_array, filename, img_idx, band_no=1, annotation_idx=1, annotation_threshold=7):
        
        
        img_size = [raster_array.shape[0],raster_array.shape[1]]

        #create image_info
        image_info = self.create_image_info(img_idx, filename, img_size)
        
        self.cocoJSON["images"].append(image_info)

        if self.has_gt==True:
            # create annotations
            polygons = self.binaryMask2Polygon(raster_array)

            for idx,polygon in enumerate(polygons):
                # # TODO: understand how to optimize the threshold below
                # if polygon.size > annotation_threshold:
                category_info = {'id':1,"is_crowd":0}
                annotation_info = self.create_annotation_info(idx+annotation_idx, img_idx, category_info, polygon, img_size)
                
                self.cocoJSON["annotations"].append(annotation_info)


    def binaryMask2Polygon(self,binaryMask):

        polygons =[]

        padded_binary_mask = np.pad(binaryMask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask)
        contours = np.subtract(contours, 1)

        def closeContour(contour):
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack((contour, contour[0]))
            return contour

        for contour in contours:
            contour = closeContour(contour)
            contour = measure.approximate_polygon(contour, 1)

            if len(contour)<3:
                continue
            contour = np.flip(contour,axis =1)
            # segmentation = contour.ravel().tolist()
            #
            # # after padding and subtracting 1 we may get -0.5 points in our segmentation
            # segmentation = [0 if i < 0 else i for i in segmentation]
            # polygons.append(segmentation)
            polygons.append(contour)
        return polygons

    def create_image_info(self,
            image_id, 
            file_name, 
            image_size,
            date_captured=datetime.datetime.utcnow().isoformat(' '),
            license_id=1, coco_url="", flickr_url=""
        ):

        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
        }

        return image_info

    def create_annotation_info(self,
            annotation_id, 
            image_id, 
            category_info, 
            segmentation,
            image_size=None, 
            tolerance=2, 
            bounding_box=None
        ):
        try:
            polygon = Polygon(np.squeeze(segmentation))
            # print(type(polygon))
            area =polygon.area

            segmentation = segmentation.ravel().tolist()

            # # after padding and subtracting 1 we may get -0.5 points in our segmentation
            bbx =[0 if i < 0 else int(i) for i in list(polygon.bounds)]
            segmentation = [0 if i < 0 else int(i) for i in segmentation]

            annotation_info = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_info["id"],
                "iscrowd": category_info["is_crowd"],
                "area": area,
                "bbox": bbx,
                "segmentation": [segmentation],
                "width": image_size[0],
                "height": image_size[1],
            }
            
        except Exception as e:
            print("Error in create_annotation_info():", e)

        return annotation_info