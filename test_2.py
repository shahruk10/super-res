import os
import sys
import cv2
import glob 
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt

import metrics
from keras.models import load_model
from utilsTrain import ensureDir, loadConfig
from utilsTest import enhance, calcPSNR

grayscale = False
scaleFactor = 1
plot = False

modelName = 'VDSR_001'
modelArch = 'VDSR_001'
modelFolder = 'H:/Projects/SR/models'
weightsFolder = modelFolder + '/' + modelName
bestModelPath = weightsFolder + '/' + 'best.hdf5'

stepFactor = 2
scale = []
window = []
weights = [bestModelPath]

img_folder = 'H:/Projects/SR/images/val/*.jpg'
imgPaths = glob.glob(img_folder)

inputDim = (16,16,3)
outputDim = (32,32,3)
scale.append(outputDim[0]/inputDim[0])
window.append(inputDim[0])

step = window[0]//int(stepFactor)

print("\nLoading Models:")
print(weights)
print("\n")
models = [load_model(w, custom_objects={'PSNRLoss':metrics.PSNRLoss} ) for w in weights]

#p = imgPaths[0]
psnr = []

for p in imgPaths:
    if grayscale:
        img = cv2.imread(p,0)
    else:
        img = cv2.imread(p)

    scaleFactor = int(scaleFactor)
    inputImg = cv2.resize(img, (int(img.shape[1]/(scale[0]*scaleFactor)), int(img.shape[0]/(scale[0]*scaleFactor))), interpolation=cv2.INTER_CUBIC)
    enhancedImg = inputImg

    if scaleFactor > 1:
        for s in range(int(np.log2(scaleFactor))):
            enhancedImg = enhance(models, enhancedImg, scale, window, step)     # recursively upscale
    else:
        #print('lalaa')
        enhancedImg = enhance(models[0], enhancedImg, scale[0], window[0], step)         # single pass

    if enhancedImg.shape != img.shape:
        shape = enhancedImg.shape
        img = img[:shape[0], :shape[1], :]

    psnr.append(calcPSNR(img, enhancedImg))

    if plot:
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(img[:,:,[2,1,0]])
        plt.subplot(1,2,2)
        plt.title("Enhanced")
        plt.imshow(enhancedImg[:,:,[2,1,0]])
        plt.show()

print("\n", "".join(['-']*70))
for path,val in zip(imgPaths,psnr):
    print("%s :\t %2.3f" % (path, val))
print("".join(['-']*70), "\n")
print("Average PSNR : ", np.mean(psnr))