# Library Imports

import h5py
import numpy as np
import cv2
import glob
from utilsDb import getPatches

# Configure Input, Paths and Parameters

img_folder = 'H:/Projects/SR/images/train/*.jpg'
savePath = 'C:/HDF5/super-res/32_16_32_64_train.hdf5'

imgPaths = glob.glob(img_folder)

no_images = len(imgPaths)
num_patches = 32
xSize = 32
ySize = 32
downSize = 16

totalPatches = len(imgPaths)*num_patches

compress = True
normalize = True
color = True

xshape = (xSize,xSize,3)
yshape = (ySize,ySize,3)

xname = 'X'
yname = 'Y'

flushSize = 20
shuffle = False

# Sampling Funcs

def sampleImage(img, downSize, upSize):
    down = cv2.resize(img,(downSize, downSize), interpolation = cv2.INTER_CUBIC)
    up = cv2.resize(down, (upSize, upSize), interpolation = cv2.INTER_CUBIC)
    return up

def sampleImages(Images,downSize,upSize):
    sampled = []
    for img in Images:
        sample = sampleImage(img,downSize,upSize)
        sampled.append(sample)
    sampled = np.asarray(sampled)
    return sampled


# Creating and Configuring the database

db = h5py.File(savePath, mode = 'w')

dbParams = {}
dbParams['dtype'] = np.float32 if normalize else np.uint8

if compress:
        dbParams['compression'] = 'gzip'
        dbParams['compression_opts'] = 9

db.create_dataset(xname, (totalPatches,)+xshape, **dbParams)
db.create_dataset(yname, (totalPatches,)+yshape, **dbParams)

# Loop Through Data

didx = 0

dataX = {}
dataY = {}

dataX[xname] = None
dataY[yname] = None

#idx = 0
#imgPath = imgPaths[0]

for idx, imgPath in enumerate(imgPaths):

    print("\t%d / %d \t\t\t" % ((idx+1), len(imgPaths)), end='\r')

    if color:
        img = cv2.imread(imgPath)
    else:
        img = cv2.imread(imgPath,0)

    patchesX = getPatches(img, num_patches, xSize, normalize = normalize)

    if dataX[xname] is None:
        dataX[xname] = patchesX
    else:
        dataX[xname] = np.concatenate((dataX[xname], patchesX), axis = 0)

    patchesY = sampleImages(patchesX, downSize, ySize)

    if dataY[yname] is None:
        dataY[yname] = patchesY
    else:
        dataY[yname] = np.concatenate((dataY[yname], patchesY), axis = 0)


    # flushing data to disc

    if idx % flushSize == flushSize-1:
        patchCount = (flushSize * num_patches)
        
        #  shuffling patches
        if shuffle:
            sidx = np.arange(patchCount)
            np.random.shuffle(sidx)
            
            for xname in dataX:
                dataX[xname] = dataX[xname][sidx,...]
            for yname in dataY:
                dataY[yname] = dataY[yname][sidx,...]

        print("\n\nFlushing to disk\n")
        for xname in dataX:
            db[xname][didx: didx + patchCount, ...] = dataX[xname][...]
            dataX[xname] = None
        for yname in dataY:
            db[yname][didx: didx + patchCount, ...] = dataY[yname][...]
            dataY[yname] = None
        
        didx += patchCount


leftover = len(imgPaths) % flushSize
if leftover:
    patchCount = (leftover * num_patches)
    
    #shuffling patches
    if shuffle:
        sidx = np.arange(patchCount)
        np.random.shuffle(sidx)
        
        for xname in dataX:
            dataX[xname] = dataX[xname][sidx,...]
        for yname in dataY:
            dataY[yname] = dataY[yname][sidx,...]

    print("\n\nFlushing to disk\n")
    for xname in dataX:
        db[xname][didx: didx + patchCount, ...] = dataX[xname][...]
        dataX[xname] = None
    for yname in dataY:
        db[yname][didx: didx + patchCount, ...] = dataY[yname][...]
        dataY[yname] = None

db.close()

print("\nDone")