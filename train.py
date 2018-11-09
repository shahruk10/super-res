import os
import gc
import sys
import h5py
import numpy as np
from argparse import ArgumentParser

from modelLib import makeModel
from utilsTrain import ensureDir, loadConfig, saveConfig, summarizeLogs
from utilsTrain import dataGenerator, sequentialDataGenerator, getSampleCount
from utilsTrain import getOptimizer, getMetric

from keras import backend
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# setting RNG seeds
tf.set_random_seed(1306)
np.random.seed(1306)

def main():

    parser = ArgumentParser()
    parser.add_argument("cfgPath", help="Path to config.toml")
    parser.add_argument("-l",   dest="loadModel", help="Load weights of model", default=False, action='store_true')
    parser.add_argument("--vb", dest="verbose", help="Print more stuff", default=False,  action='store_true')
    parser.add_argument("--sg", dest="sequentialGenerator", help="Use this flag if database doesn't fit in memory", default=False, action='store_true')
    args = parser.parse_args()

    # loading config file ...
    cfg = loadConfig(args.cfgPath)
   
    # ... and unpacking variables
    try:
        dictget = lambda d, *k: [d[i] for i in k]

        xName, yName, trainDbPath, valDbPath = dictget(cfg['database'], 'x', 'y', 'train', 'val')
        
        modelArch, modelDir, modelName = dictget(cfg['model_arch'], 'modelArch', 'modelDir', 'modelName')

        epochs, batchSize, patience, numWorkers, maxMemory,lossFunc, metricFuncs = dictget(cfg['training_params'],
                                                                                            'epochs',
                                                                                            'batchSize',
                                                                                            'patience',
                                                                                            'numWorkers',
                                                                                            'maxMemory',
                                                                                            'lossFunc',
                                                                                            'metricFuncs')
    except KeyError as err:
        print("\n\nERROR: not all parameters defined in config.toml : ", err)
        print("Exiting ... \n\n")
        sys.exit(1)

    # setting up directories
    modelFolder = os.path.join(modelDir, modelName)
    saveConfig(args.cfgPath, modelFolder)
    
    if len(valDbPath) != 0:
        print("\n\nValidation sets provided : ", valDbPath)

    # starting cross fold training
    kfolds = len(trainDbPath)
    for fold_idx in range(kfolds):
 
        print("\n" + "".join(['-']*25), "Starting Fold %03d" % (fold_idx+1) + "".join(['-']*25) + "\n")
        weightsFolder = os.path.join(modelFolder, "weights_%03d" % (fold_idx+1))
        bestModelPath = os.path.join(weightsFolder, "best_fold_%03d.hdf5" % (fold_idx+1))
        ensureDir(bestModelPath)

        # creating model
        model = makeModel(modelArch, cfg['model_params'], verbose=args.verbose)

        # loading model weights
        if args.loadModel:
            print("\n\nLoading Model Weights:\t %s" % modelName)
            model.load_weights(bestModelPath)
            log = np.genfromtxt(os.path.join(modelFolder,  modelName + '_trainingLog_%03d.csv' % (fold_idx+1)), delimiter=',', dtype=str)[1:, 0]
            epochStart = len(log)
        else:
            epochStart = 0

        opt = getOptimizer(cfg['optimizer_params'])

        if lossFunc not in ['mse','mae','mape', 'logcosh', 'msle']:
            lossFunc = getMetric(lossFunc)
        
        for idx, func in enumerate(metricFuncs):
            if func == 'psnr':
                metricFuncs[idx] = getMetric(func)

        model.compile(loss=lossFunc, optimizer=opt, metrics=metricFuncs)

        # starting up training generator
        if kfolds > 1 and len(valDbPath) == 0:
            trainDbs = [trainDbPath[idx] for idx in range(kfolds) if idx != fold_idx]
        else:
            trainDbs = trainDbPath

        trainSamples = getSampleCount(trainDbs, xName)
        if args.sequentialGenerator:
            trainGen = sequentialDataGenerator(trainDbs, xName, yName, batchSize,numWorkers=numWorkers,maxMemoryUsage=maxMemory)
        else:
            trainGen = dataGenerator(trainDbs, xName, yName, batchSize)

        monitorMetric = 'val_loss'
        if len(valDbPath) == 0 and kfolds > 1:
            valDb = [trainDbPath[fold_idx]]
        elif len(valDbPath) != 0:
            valDb = valDbPath
        else:
            monitorMetric = 'loss'
        
         # callbacks
        check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:03d}.hdf5"), monitor=monitorMetric, mode='auto')
        check2 = ModelCheckpoint(bestModelPath, monitor=monitorMetric, save_best_only=True, mode='auto')
        check3 = EarlyStopping(monitor=monitorMetric, min_delta=0.01, patience=patience, verbose=0, mode='auto')
        check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog_%03d.csv' % (fold_idx+1)), separator=',', append=True)
        check5 = ReduceLROnPlateau(monitor=monitorMetric, factor=0.1, patience=patience//3, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-10)

        print("\nInitiating Training:\n")
        # using a validation set
        if kfolds > 1 or len(valDbPath) != 0:
            valSamples = getSampleCount(valDb, xName)
            if args.sequentialGenerator:
                valGen = sequentialDataGenerator(valDb, xName, yName, batchSize,numWorkers=2,maxMemoryUsage=maxMemory)
            else:
                valGen = dataGenerator(valDb, xName, yName, batchSize)

            model.fit_generator(trainGen, steps_per_epoch=(trainSamples // batchSize), epochs=epochs, initial_epoch=epochStart,
                                validation_data=valGen, validation_steps=(valSamples // batchSize), 
                                callbacks=[check1, check2, check3, check4, check5], verbose=1)

            valGen.close()

        # no validation set
        else:       
            model.fit_generator(trainGen, steps_per_epoch=(trainSamples // batchSize), epochs=epochs, initial_epoch=epochStart,
                                callbacks=[check1, check2, check3, check4, check5],verbose=1)

        trainGen.close()
        gc.collect()
        backend.clear_session()

        # closing hdf5 db in case it was left open due to unexpected shutdown
        for obj in gc.get_objects(): 
            try:
                if isinstance(obj, h5py.File):
                    try:
                        obj.close()
                    except:
                        pass # Was already closed
            except:
                pass    # elements in garbage without __class__ attribute

    summarizeLogs(modelFolder)

if __name__ == "__main__":
    main()
