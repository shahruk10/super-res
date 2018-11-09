import os
import cv2
import glob
import h5py
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction.image import extract_patches_2d

def shuffleArrays(X):
    ''' shuffles an array or list of arrays with the same shape in axis 0 '''
    
    if isinstance(X, list):
        totalSamples = X[0].shape[0]
        shuffle_idx = np.arange(totalSamples)
        np.random.shuffle(shuffle_idx)
        for idx in range(len(X)):
            if X[idx].shape[0] == totalSamples:
                X[idx] = X[idx][shuffle_idx]
    else:
        totalSamples = X.shape[0]
        shuffle_idx = np.arange(totalSamples)
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]

    return X

def getFileIndicesForCV(K, fileList, randomState=27):
    ''' returns K lists of files indexes, dividing files into K random non overlapping groups '''

    kf = KFold(n_splits=K, random_state=randomState, shuffle=True)
    fileIndex = np.arange(len(fileList))
    foldIndices = [val for _, val in kf.split(fileIndex)]

    return foldIndices

def getDatasetInfo(cfg):
    ''' gets the names and shapes of datasets speicified in config '''

    datasetInfo = {'originalPatches' : [], 'resizedPatches' : []}
    
    for patchSize in cfg['original_patches']:
        dinfo = {}
        dinfo['name'] = 'org_%d' % patchSize
        dinfo['size'] = patchSize
        dinfo['shape'] = (patchSize, patchSize, 3,) if cfg['color'] else (patchSize, patchSize,)
        datasetInfo['originalPatches'].append(dinfo)
    
    for patchSize in cfg['resized_patches']:
        dinfo = {}
        dinfo['name'] = 'rsz_%d' % patchSize
        dinfo['size'] = patchSize
        dinfo['shape'] = (patchSize, patchSize, 3,) if cfg['color'] else (patchSize, patchSize,)
        datasetInfo['resizedPatches'].append(dinfo)       

    return datasetInfo

def getPatches(img, n, size, normalize=False, randomState=1306):
    ''' returns n patches taken from random locations picked from uniform distribution.
    Each patch is a square with height and width equal to size. If normalize is true, values
    are converted to float32 and divided by 255.0 '''

    patches = extract_patches_2d(img, patch_size=(size,size), max_patches=n, random_state=randomState)

    if normalize:
        patches = np.array(patches, dtype=np.float32)
        patches /= 255.0
    
    return patches

def resizePatches(patches, size, interpolation='bicubic'):
    ''' resizes a given list/array of patches to a square patch of height and width equal to size.
    The interpolation method can be specified : bicubic / bilinear '''

    if interpolation == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif interpolation == 'linear':
        interpolation = cv2.INTER_LINEAR

    return np.array([cv2.resize(p, (size,size), interpolation=interpolation) for p in patches])
     
def createHDF5(imgPaths, cfg, savePath, shuffle=True, flushSize=1500, normalize=False, compress=False):
    ''' creates hdf5 datbase file of image patches given list of image paths. Images are read as color (BGR).
    The patches are cropped at random locations sampled from a uniform distribution. They are resized down to
    the size(s) and interpolation method(s) speicified in cfg (contents of config file).   
    
    If compress is true, will compress entries by gzip. Will lower read time but may give 10:1 compression
    especially for sparse data. '''

    # file level shuffle
    if shuffle:
        imgPaths = shuffleArrays(imgPaths)

    print("\nTotal Images \t:", imgPaths.shape[0])

    print("Setting up hdf5 database")
    db = h5py.File(savePath, mode='w')
    dbParams = {}
    dbParams['dtype'] = np.float32 if normalize else np.uint8
    if compress:
        dbParams['compression'] = 'gzip'
        dbParams['compression_opts'] = 9
    
    totalPatches  = len(imgPaths) * cfg['num_patches']
    datasetInfo = getDatasetInfo(cfg)

    # creating space for all the data and initializing
    # dictionaries to store patches before flushing to disk
    dataOriginal = {}
    dataResized = {}
    for dinfo1 in datasetInfo['originalPatches']:
        xname = dinfo1['name']
        dataOriginal[xname] = None
        db.create_dataset(xname,  (totalPatches,) + dinfo1['shape'], **dbParams)

        for dinfo2 in datasetInfo['resizedPatches']:
            if dinfo1['size'] != dinfo2['size']:
                yname = "%s_%s" % (xname, dinfo2['name'])
                dataResized[yname] = None
                db.create_dataset(yname,  (totalPatches,) + dinfo2['shape'], **dbParams)

    didx = 0          # database index

    print("\nStoring features in hdf5 database ...")
    try:
        for idx, imgPath in enumerate(imgPaths):

            print("\t%d / %d \t\t\t" % ((idx+1), len(imgPaths)), end='\r')

            # reading image
            if cfg['color']:
                img = cv2.imread(imgPath)
            else:
                img = cv2.imread(imgPath,0)

            for dinfo1 in datasetInfo['originalPatches']:
                # getting patches
                xname = dinfo1['name']
                patchesOriginal = getPatches(img, cfg['num_patches'], dinfo1['size'], normalize=normalize)
                if dataOriginal[xname] is None:
                    dataOriginal[xname] = patchesOriginal
                else:
                    dataOriginal[xname] = np.concatenate((dataOriginal[xname], patchesOriginal),axis=0)

                # resizing patches
                for dinfo2 in datasetInfo['resizedPatches']:
                    yname = "%s_%s" % (xname, dinfo2['name'])
                    if yname in dataResized:
                        patchesResized = resizePatches(patchesOriginal, dinfo2['size'], cfg['interpolation'])
                        if dataResized[yname] is None:
                            dataResized[yname] = patchesResized
                        else:
                            dataResized[yname] = np.concatenate((dataResized[yname], patchesResized), axis=0)
                            
            # shuffling and flushing to disk
            if idx % flushSize == flushSize-1:
                patchCount = (flushSize * cfg['num_patches'])
                
                #  shuffling patches
                if shuffle:
                    sidx = np.arange(patchCount)
                    np.random.shuffle(sidx)
                    
                    for xname in dataOriginal:
                        dataOriginal[xname] = dataOriginal[xname][sidx,...]
                    for yname in dataResized:
                        dataResized[yname] = dataResized[yname][sidx, ...]

                print("\n\nFlushing to disk\n")
                for xname in dataOriginal:
                    db[xname][didx: didx + patchCount, ...] = dataOriginal[xname][...]
                    dataOriginal[xname] = None
                for yname in dataResized:
                    db[yname][didx: didx + patchCount, ...] = dataResized[yname][...]
                    dataResized[yname] = None
                
                didx += patchCount      

                    
    except Exception as err:
        print("\n\nERROR:", err)
        print(imgPath)
        db.close()
        os.remove(savePath)
        return

    # flushing remainder
    leftover = len(imgPaths) % flushSize
    if leftover:
        patchCount = (leftover * cfg['num_patches'])
        
        #  shuffling patches
        if shuffle:
            sidx = np.arange(patchCount)
            np.random.shuffle(sidx)
            
            for xname in dataOriginal:
                dataOriginal[xname] = dataOriginal[xname][sidx,...]
            for yname in dataResized:
                dataResized[yname] = dataResized[yname][sidx, ...]

        print("\n\nFlushing to disk\n")
        for xname in dataOriginal:
            db[xname][didx: didx + patchCount, ...] = dataOriginal[xname][...]
            dataOriginal[xname] = None
        for yname in dataResized:
            db[yname][didx: didx + patchCount, ...] = dataResized[yname][...]
            dataResized[yname] = None

    db.close()

    print("\nDone")

def createKFoldHDF5(K, imgPaths, cfg, saveDir, saveName, shuffle=True, flushSize=500, normalize=False, compress=False):
    ''' creates K hdf5 datbase file with each database having patches from different set of files, 
    given list of paths to original images and config. If compress is true, will compress entries by gzip. 
    Will lower read time but may give 10:1 compression especially for sparse data. '''

    foldIndices = getFileIndicesForCV(K, imgPaths)

    for i, fidx in enumerate(foldIndices):
        print("\n" + "".join(['-']*25), "Generating Fold %03d" % (i+1) + "".join(['-']*25) + "\n")

        savePath = os.path.join(saveDir, '%s_fold_%02d.hdf5' % (saveName,i+1))
        createHDF5(imgPaths[fidx], cfg, savePath, shuffle=True, flushSize=flushSize, normalize=normalize, compress=compress)
