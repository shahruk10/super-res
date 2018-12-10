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


def main():
    parser = ArgumentParser()
    
    parser.add_argument("cfgPath", help="Path(s) to model config.toml; if multiple, seperate by comma")
    
    parser.add_argument("fileList", help="Path to file containing image file paths to test model on")
    
    parser.add_argument("--plot", dest="plot", help="Will show each original and enhanced image", action='store_true', default=False)
    
    parser.add_argument("--stepFactor", dest="stepFactor", default=2,
                        help="Inverse fraction of the window size to use as step size, using 1 implies no overlap between windows")

    parser.add_argument("--scaleFactor", dest="scaleFactor", default=1,
                        help="Additional factor by which to first downscale by cv2 and then upscale by model, used to recursively                           apply model to get larger image")
    
    parser.add_argument("--use-grayscale", dest="grayscale", action='store_true', default=False,
                        help="load images as grayscale")
    
    parser.add_argument("--use-all-folds", dest="useAllFolds", action='store_true', default = False,
                        help="Setting this flag will utilize models of all folds in the ensemble; \
                              otherwise on the model named best.hdf5 in the model folder is used" )

    args = parser.parse_args()

    cfgPaths = args.cfgPath.split(",")
    cfgs = [loadConfig(p) for p in cfgPaths]

    with open(args.fileList, 'r') as f:
        imgPaths = f.readlines()
    imgPaths = [p.strip('\n') for p in imgPaths]
    
    print("\nTesting on %d images" % len(imgPaths))

    scale = []
    window = []
    weights = []
    for c in cfgs:
        inputDim = c['model_params']['inputDim']
        outputDim = c['model_params']['outputDim']
        scale.append(outputDim[0]/inputDim[0])
        window.append(inputDim[0])

        modelFolder = os.path.join(c['model_arch']['modelDir'], c['model_arch']['modelName'])
        if args.useAllFolds:
            weights.extend( sorted(glob.glob(os.path.join(modelFolder, 'best*.hdf5'))) )
        else:
            weights.append( os.path.join(modelFolder, 'best.hdf5') )
    
    assert len(np.unique(scale)) == 1, "ERROR: all models dont have same scale factor"
    assert len(np.unique(window)) == 1, "ERROR: all models dont have same input dimension"
    
    scale = scale[0]
    window = window[0]
    step = window//int(args.stepFactor)
    print("\nLoading Models:")
    print(weights)
    print("\n")
    models = [load_model(w, custom_objects={'PSNRLoss':metrics.PSNRLoss} ) for w in weights]

    psnr = []
    for p in imgPaths:
        if args.grayscale:
            img = cv2.imread(p,0)
        else:
            img = cv2.imread(p)

        scaleFactor = int(args.scaleFactor)
        inputImg = cv2.resize(img, (int(img.shape[1]/(scale*scaleFactor)), int(img.shape[0]/(scale*scaleFactor))), interpolation=cv2.INTER_CUBIC)
        enhancedImg = inputImg

        if scaleFactor > 1:
            for s in range(int(np.log2(scaleFactor))):
                enhancedImg = enhance(models, enhancedImg, scale, window, step)     # recursively upscale
        else:
            enhancedImg = enhance(models, enhancedImg, scale, window, step)         # single pass

        if enhancedImg.shape != img.shape:
            shape = enhancedImg.shape
            img = img[:shape[0], :shape[1], :]

        psnr.append(calcPSNR(img, enhancedImg))

        if args.plot:
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

if __name__ == '__main__' :
    main()