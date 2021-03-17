import numpy as np
import json
from pycocotools import mask
import cv2

scoreThreshold = 0.0
imgShape = (256,256)

with open('data/dvrpc/inferences/masks/DVRPCResNet50.json') as f:
    data = json.load(f)

with open('data/dvrpc/annotations/DVRPC_test.json') as f:
    inpImageData = json.load(f)
    imgIDNameMap = {}
    for img in inpImageData['images']:
        imgIDNameMap[img['id']] = img['file_name']
    del inpImageData

def genBinaryMasks(savePredImage=True, saveInputImage=True, saveOverlay=True, swapColors=True, printScore=False):
    global imgData
    global imgIDNameMap
    n = 0
    totScore = 0
    for imgData in data['images']:
        pred = np.zeros(imgShape)
        for det in imgData['dets']:
            if(det['score']>scoreThreshold):
                n += 1
                totScore += det['score']
                rle = det['mask']
                x = mask.decode(rle)
                pred[x==1]=255

        if(savePredImage):
            cv2.imwrite('data/dvrpc/inferences/binaryMasks/%s.jpg'%(imgData['image_id']),pred)

        inpImg = None
        if(saveInputImage):
            inpImg = cv2.imread('data/dvrpc/images/%s'%(imgIDNameMap[imgData['image_id']]))
            if(swapColors):
                inpImg[:, :, [0, 1, 2]] = inpImg[:, :, [0, 2, 1]]
                '''
                0,1,2 #
                1,0,2 #
                2,0,1 #
                0,2,1 # Close but still not correct
                1,2,0 #
                2,1,0 #
                '''
            cv2.imwrite('data/dvrpc/inferences/binaryMasks/%s_inpImg.jpg'%(imgData['image_id']),inpImg)

        if(saveOverlay):
            if(inpImg is None):
                overlayImg = cv2.imread('data/dvrpc/images/%s'%(imgIDNameMap[imgData['image_id']]))
            else:
                overlayImg = inpImg
            overlayImg[pred==255]=(255,0,0)
            cv2.imwrite('data/dvrpc/inferences/binaryMasks/%s_overlayImg.jpg'%(imgData['image_id']),overlayImg)

    if(printScore):
        print('Average score: %f'%(totScore/n))

if __name__ == '__main__':
    genBinaryMasks()