import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from yolact import Yolact

#modelPath = 'weights/External config_0_10000.pth'
#modelPath = 'weights/External config_1_20000.pth'
modelPath = 'weights/config_2_25721_interrupt.pth'

'''
imgPaths = [
    'inferenceTestData/trainImgs/1010_1_1.tif',
    'inferenceTestData/trainImgs/1010_1_2.tif',
    'inferenceTestData/trainImgs/1010_1_3.tif',
    'inferenceTestData/trainImgs/1010_1_4.tif',
]
labelPaths = [
    'inferenceTestData/trainLabels/1010_1_1.tif',
    'inferenceTestData/trainLabels/1010_1_2.tif',
    'inferenceTestData/trainLabels/1010_1_3.tif',
    'inferenceTestData/trainLabels/1010_1_4.tif',
]

model = Yolact()
weightsAndState = torch.load(modelPath, map_location=torch.device('cpu'))
model.load_state_dict(weightsAndState)

model.eval()

def genLabelVis(label):
    labelVis = np.zeros_like(label)
    labelVis[label==1] = 255
    return(labelVis)

#def pred(img):
    #model.predict([img])

for img,label in zip(imgPaths,labelPaths):
    img = np.array(cv2.imread(img))
    label = np.array(cv2.imread(label))
    labelVis = genLabelVis(label)
    predImg = pred(img)
    cv2.imshow('Image',img)
    cv2.imshow('Label',labelVis)
    #cv2.imshow('Prediction',predImg)
    cv2.waitKey(0)
'''
def saveNPArr(arr, npArrPath):
    with open(npArrPath, 'wb') as f:
        np.save(f, np.array(arr), allow_pickle=True)

imgPaths = [
    'data/dvrpc/images/1010_1_1.tif',
    'data/dvrpc/images/1010_1_2.tif',
    'data/dvrpc/images/1010_1_3.tif',
    'data/dvrpc/images/1010_1_4.tif',
]
predsSavePath = 'data/dvrpc/inferences/test.npy'

model = Yolact()
weightsAndState = torch.load(modelPath, map_location=torch.device('cpu'))
model.load_state_dict(weightsAndState)

imgs = [np.array(cv2.imread(img)) for img in imgPaths]

preds = model.predict(imgs)
saveNPArr(preds, predsSavePath)

for pred in preds:
    print('Prediction shape: ', pred.shape)