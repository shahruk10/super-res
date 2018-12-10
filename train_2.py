from utilsTrain import *
from modelLib import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

modelName = 'VDSR_001'
modelArch = 'VDSR_001'
modelFolder = 'H:/Projects/SR/models'
weightsFolder = modelFolder + '/' + modelName
bestModelPath = weightsFolder + '/' + 'best.hdf5'
ensureDir(bestModelPath)

#optimizer_params = ''
metricFuncs = ['psnr']

trainDbPath = 'C:/HDF5/super-res/32_16_32_64_train.hdf5'
valDbPath = 'C:/HDF5/super-res/32_16_32_64_test.hdf5'

batchSize = 256

patience = 1000

xName = 'X'
yName = 'Y'

model = makeModel(modelArch)

epochStart = 0
epochs = 1000

#opt = getOptimizer(optimizer_params)

opt = Adam(lr = 0.001)

lossFunc = 'mse' 
#lossFunc = getMetric(lossFunc)

for idx, func in enumerate(metricFuncs):
    if func == 'psnr':
        metricFuncs[idx] = getMetric(func)

model.compile(loss=lossFunc, optimizer=opt, metrics=metricFuncs)

valDb = valDbPath
trainDbs = trainDbPath
trainSamples = getSampleCount(trainDbs, xName)
valSamples = getSampleCount(valDb, xName)

trainGen = dataGenerator(trainDbs, xName, yName, batchSize)

fold_idx = 0

monitorMetric = 'val_loss'

check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:03d}.hdf5"), monitor=monitorMetric, mode='auto')
check2 = ModelCheckpoint(bestModelPath, monitor=monitorMetric, save_best_only=True, mode='auto')
check3 = EarlyStopping(monitor=monitorMetric, min_delta=0.01, patience=patience, verbose=0, mode='auto')
check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog_%03d.csv' % (fold_idx+1)), separator=',', append=True)
check5 = ReduceLROnPlateau(monitor=monitorMetric, factor=0.1, patience=patience//3, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-10)

print("\nInitiating Training:\n")

valGen = dataGenerator(valDb, xName, yName, batchSize)

model.fit_generator(trainGen, steps_per_epoch=(trainSamples // batchSize), epochs=epochs, initial_epoch=epochStart,
                    validation_data=valGen, validation_steps=(valSamples // batchSize), 
                    callbacks=[check1, check2, check3, check4, check5], verbose=1)

valGen.close()
trainGen.close()

