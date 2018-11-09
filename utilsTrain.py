import os
import sys
import gc
import glob
import toml
import shutil
import numpy as np
import h5py
import threading

if sys.version_info[0] < 3:     # python 2/3 compatible
    from Queue import Queue
else:
    from queue import Queue

import metrics
from matplotlib import pyplot as plt
from keras.optimizers import Adam, Nadam, Adamax, Adagrad, Adadelta, RMSprop, SGD


def ensureDir(filePath):
    ''' checks if the folder at filePath exists. If not, it creates it. '''

    directory = os.path.dirname(filePath)

    if not os.path.exists(directory):
        os.makedirs(directory)

def loadConfig(cfgPath):
    ''' loads toml file containing configurations for model training '''

    with open(cfgPath, 'r') as df:
        tomlString = df.read()
    cfg = toml.loads(tomlString)
    return cfg

def saveConfig(cfgPath, savePath):
    ''' copies config file to model directory '''

    ensureDir(os.path.join(savePath, 'config.toml'))
    shutil.copy(cfgPath, os.path.join(savePath, 'config.toml'))

def getMemoryPerBatch(dbPath, xName, yName, batchSize):
    ''' calculates and return the memory requirement in bytes of single batch of data '''

    with h5py.File(dbPath, "r") as db:
        xMem = (db[xName][0].nbytes) * batchSize
        yMem = (db[yName][0].nbytes) * batchSize

    return xMem + yMem

def getCurrentRAMUsage():
    ''' returns total, used and free memory at the time of call (RAM + Swap). Works in Linux based OS only'''

    totMem, usedMem, freeMem = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    return totMem,usedMem, freeMem

def getSampleCount(dbPath,xName):
    ''' returns the number of data samples in hdf5 db '''
    
    if not isinstance(dbPath, list):
        dbPath = [dbPath]

    count = 0
    for path in dbPath:
        with h5py.File(path, "r") as db:
            count += db[xName].shape[0]

    return count

def getOptimizer(params):
    ''' returns the learning optimizer based on given params '''

    opt_params = { p:params[p] for p in params if p != 'optimizer'}

    if params['optimizer'] == 'adam':
        opt = Adam(**opt_params)
    elif params['optimizer'] == 'nadam':
        opt = Nadam(**opt_params)
    elif params['optimizer'] == 'adamax':
        opt = Adamax(**opt_params)
    elif params['optimizer'] == 'sgd':
        opt = SGD(**opt_params)
    elif params['optimizer'] == 'rmsprop':
        opt = RMSprop(**opt_params)
    elif params['optimizer'] == 'adagrad':
        opt = Adagrad(**opt_params)
    elif params['optimizer'] == 'adadelta':
        opt = Adadelta(**opt_params)

    return opt

def getMetric(metric):
    ''' returns the required metric from metrics.py '''
    
    if metric == 'psnr':
        return metrics.PSNRLoss

def dataGenerator(dbPaths, xName, yName, batchSize):
    ''' This generator queues and outputs batches from hdf5 database(s). The name of the x entries and y entries must be given as well as the batchSize. '''

    if not isinstance(dbPaths,list):
        dbPaths = [dbPaths]

    databases = [h5py.File(path, "r") for path in dbPaths]

    try:
        while True:
            np.random.shuffle(databases)
            for db in databases:
                X = db[xName][...]
                Y = db[yName][...]
                # creating list of indices of each batch
                totalSamples = X.shape[0]
                idx = np.arange(totalSamples)
                batches = [idx[i: ((i + batchSize) if (i+batchSize)<totalSamples else (totalSamples-1))] for i in range(0, totalSamples, batchSize)]
                np.random.shuffle(batches)

                for batch in batches:
                    x = X[batch, ...]
                    y = Y[batch, ...]
                    shuffle_idx = np.arange(len(batch))
                    np.random.shuffle(shuffle_idx)
                    yield (x[shuffle_idx,...],y[shuffle_idx,...])
                
                del X
                del Y
                gc.collect()
    except Exception as err:
        print(err)
        # closing db - triggers exception in dataQueuer
        for db in databases:
            db.close()
        return

class DataQueuer (threading.Thread):
    ''' queues up training/testing data for training/testing generator. Used by sequentialDataGenerator '''
    
    def __init__(self, db, xName, yName, Q, batchIndexes, workerID):
        threading.Thread.__init__(self)
        self.databases = db
        self.xName = xName
        self.yName = yName
        self.Q = Q
        self.batchIndexes = batchIndexes
        self.ID = workerID

    def run(self):
        ''' puts batches of data on the queue until databases are closed '''
        try:
            while True:
                for db, sidx in zip(self.databases, self.batchIndexes):
                    for i in sidx[self.ID]:
                        x = db[self.xName][i, ...]
                        y = db[self.yName][i, ...]
                        
                        # always shuffle within a batch
                        shuffle_idx = np.arange(len(i))
                        np.random.shuffle(shuffle_idx)
                        x = x[shuffle_idx]
                        y = y[shuffle_idx]

                        # print("Queued")
                        self.Q.put([x, y])
        
        except ValueError:
            # db closed from outside -> generator closed -> job done -> time to exit
            print("Closing Queueing Thread # %d" % self.ID)
            return
        except Exception as err:
            # some other error
            print(err)
            return

def sequentialDataGenerator(dbPaths, xName, yName, batchSize, shuffle=True, maxMemoryUsage = 8, numWorkers = 4):
    ''' This generator queues and outputs batches from hdf5 database(s) sequentially without loading everything to 
    RAM at the same time. Use this if the database does not fit in your computer's memory. The max RAM usage for 
    queueing batches may be speicied by maxMemoryUsage (GB). The number of threads dedicated to queueing batches 
    is specified by numWorkers '''

    if not isinstance(dbPaths,list):
        dbPaths = [dbPaths]

    databases = []
    batchIndexes = []
    for idx, path in enumerate(dbPaths):
        databases.append(h5py.File(path, "r"))
        totalSamples = databases[idx][xName].shape[0]
        idx = np.arange(totalSamples)

        # creating list of indices of each batch
        sidx = [idx[i: ((i + batchSize) if (i+batchSize)<totalSamples else (totalSamples-1))] for i in range(0, len(idx), batchSize)]
        
        #  padding batch number to distribute evenly among workers
        if len(sidx) % numWorkers:
            padding = numWorkers - (len(sidx) % numWorkers)
            # repeating batches
            for i in range(padding):
                sidx.append(sidx[i])

        # chunking batches for each worker
        n = len(sidx) // numWorkers
        sidxChunks = [sidx[i:i + n] for i in range(0, len(sidx), n)]
        batchIndexes.append(sidxChunks)

    memPerBatch = getMemoryPerBatch(dbPaths[0],xName,yName,batchSize) / (1024.0 * 1024.0 * 1024.0)      # converting to GB
    maxQueueSize = int(maxMemoryUsage / memPerBatch)                                                
    Q = Queue(maxsize = maxQueueSize)
    # print("Max Generator Queue Size : ", maxQueueSize, " using upto ", maxMemoryUsage , "GB\n")

    # staring threads for queueing data from disk
    threads = [ DataQueuer(databases,xName,yName,Q,batchIndexes,n) for n in range(numWorkers) ]
    for t in threads:
      t.setDaemon(True)
      t.start()

    try:
        # while generator is being iterated over, 
        # get data from queue and yield them to caller.
        # do this until GeneratorExit exception raised, 
        # or any other error 
        while True:
            x,y = Q.get()
            Q.task_done()       # emptying a space in the queue
            yield (x,y)

    # GeneratorExit exception raised when normally closing
    except:
      # closing db - triggers exception in dataQueuer
      for db in databases:
        db.close()
      # getting elements from queue to remove block of .put() in threads
      for n in range(numWorkers):
        x, y = Q.get_nowait()
        Q.task_done()
      return

def summarizeLogs(modelDir, verbose=True):
    ''' summarizes training results '''

    logFiles = sorted(glob.glob( os.path.join(modelDir, os.path.basename(modelDir) + "*.csv")))

    summary = { 'min_val_loss' : [],
                'min_val_loss_epoch': [],
            }

    plt.figure(figsize=(15, 6), dpi=300)
    for fidx, lg in enumerate(logFiles):
        fid = 'fold_%02d' % (fidx+1)
        log = np.genfromtxt(lg, delimiter=',', names=True) 
        summary[fid] = {'train_loss' : log['loss'], 'val_loss' : log['val_loss']}
        summary['min_val_loss'].append(np.min(log['val_loss']))
        summary['min_val_loss_epoch'].append(np.argmin(log['val_loss']))
        
        plt.plot(log['epoch'], log['val_loss'], 'o-',label="Fold %d" % (fidx+1), ms=2)

    summary['avg_val_loss'] = np.mean(summary['min_val_loss'])
    summary['std_val_loss'] = np.std(summary['min_val_loss'])

    cfg = loadConfig(os.path.join(modelDir,'config.toml'))
    summary['lr'] = cfg['optimizer_params']['lr']
    summary['opt'] = cfg['optimizer_params']['optimizer']
    summary['opt_params'] = str(cfg['optimizer_params'])

    with open(os.path.join(modelDir, os.path.basename(modelDir) + "_summary.txt"), 'w') as df:
        
        df.write("Min Val Loss,")
        for idx in range(len(logFiles)):
            df.write("%f," % (summary['min_val_loss'][idx]))
        df.write("\n")

        df.write("Min Val Loss Epoch,")
        for idx in range(len(logFiles)):
            df.write("%d," % (summary['min_val_loss_epoch'][idx]))
        df.write("\n")
        
        df.write("Avg Val Loss, %f\n" % summary['avg_val_loss'])
        df.write("Std Val Loss, %f\n" % summary['std_val_loss'])
        df.write("Learning Rate, %f\n" % summary['lr'])
        df.write("Optimizer, %s\n" % summary['opt'])
        df.write("Optimizer Params, %s\n" % summary['opt_params'])

    plt.grid()
    plt.title(os.path.basename(modelDir), color='#606060', fontweight="semibold", fontsize=12, family='sans-serif')
    plt.xlabel("Epochs", color='#606060', fontweight="medium", fontsize=11, family='sans-serif')
    plt.ylabel("Validation Loss", color='#606060', fontweight="medium", fontsize=11, family='sans-serif')
    plt.legend(loc='best', fontsize=10, frameon=True)
    plt.savefig(os.path.join(modelDir, os.path.basename(modelDir) + '_val_loss.png'), dpi=300)

    if verbose:
        print("".join(["-"]*50))
        print("Average Val Loss : %f" % summary['avg_val_loss'])
        print("Std Dev Val Loss : %f" % summary['std_val_loss'])
