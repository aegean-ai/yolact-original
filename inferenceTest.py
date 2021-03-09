import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from yolact import Yolact
'''
#modelPath = 'weights/External config_0_10000.pth'
#modelPath = 'weights/External config_1_20000.pth'
modelPath = 'weights/External config_2_25721_interrupt.pth'

device = torch.device('cpu')
model = Yolact()
model.load_state_dict(torch.load(PATH, map_location=device))

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

def saveNPArr(arr, npArrPath):
    with open(npArrPath, 'wb') as f:
        np.save(f, np.array(arr), allow_pickle=True)

def genLabelVis(label):
    labelVis = np.zeros_like(label)
    labelVis[label==1] = 255
    return(labelVis)


for img,label in zip(imgPaths,labelPaths):
    img = np.array(cv2.imread(img))
    label = np.array(cv2.imread(label))
    labelVis = genLabelVis(label)
    #predImg = pred(img)
    cv2.imshow('Image',img)
    cv2.imshow('Label',labelVis)
    cv2.waitKey(0)
'''

from yolact import Yolact

#modelPath = 'weights/External config_0_10000.pth'
modelPath = 'weights/External config_1_20000.pth'
#modelPath = 'weights/External config_2_25721_interrupt.pth'

imgPaths = [
    'data/dvrpc/images/1010_1_1.tif',
    'data/dvrpc/images/1010_1_2.tif',
    'data/dvrpc/images/1010_1_3.tif',
    'data/dvrpc/images/1010_1_4.tif',
]

savePaths = [
    'data/dvrpc/inferences/1010_1_1.tif',
    'data/dvrpc/inferences/1010_1_2.tif',
    'data/dvrpc/inferences/1010_1_3.tif',
    'data/dvrpc/inferences/1010_1_4.tif',
]

predsSavePath = 'data/dvrpc/inferences/test.npy'

net = Yolact()
#net.load_weights(modelPath)
net.load_state_dict(torch.load(modelPath), strict=False)
net.eval()

import eval
for imgPath,savePath in zip(imgPaths,savePaths):
    print('Evaluating:',imgPath.split('/')[-1])
    eval.evalimage(net=net, path=imgPath, save_path=savePath)