import os
import sys
import glob
import numpy as np
from utilsTrain import loadConfig, saveConfig, ensureDir
from utilsDb import createHDF5, createKFoldHDF5
from argparse import ArgumentParser

def main():
    ''' will make an hdf5 database using the ifeat/ofeats at featDir '''

    parser = ArgumentParser()
    parser.add_argument("configPath", help="Path tp config_db file")
    args = parser.parse_args()

    if args.configPath is None:
        print("\n\nRequired arguments not set !")
        parser.print_help()
        print("Exiting ... \n\n")
        sys.exit(1)

    cfg = loadConfig(args.configPath)

    if cfg['shuffle']:
        print("Shuffling On")
    else:
        print("Shuffling Off")
    
    if cfg['normalize']:
        print("Will normalize values to between 0 and 1.0")
    else:
        print("Pixel values will be between 0-225")

    if cfg['compress']:
        print("Will apply GZip Compression\n")
    else:
        print("No compression applied\n")


    # creating directories
    saveDir = os.path.join(cfg['save_dir'], cfg['save_name'])
    savePath = os.path.join(saveDir, cfg['save_name'])
    ensureDir(savePath)

    saveConfig(args.configPath, saveDir)

    print("\nWill use images from : ")
    print(cfg['image_sets'])
    
    imgPaths = []
    for imgSet in cfg['image_sets']:
        imgPaths.extend(sorted(glob.glob(os.path.join(imgSet, '*'))))
    imgPaths = np.array(imgPaths)

    if cfg['kfold'] == 0:
        createHDF5(imgPaths,
                   cfg,
                   savePath + '.hdf5',
                   shuffle=cfg['shuffle'],
                   flushSize=cfg['flush_size'],
                   compress=cfg['compress'],
                   normalize=cfg['normalize'])
    else:
        print("Creating database for %d folds ...\n" % int(cfg['kfold']))
        createKFoldHDF5(cfg['kfold'],
                        imgPaths, 
                        cfg,
                        saveDir,
                        cfg['save_name'],
                        shuffle=cfg['shuffle'],
                        flushSize=int(cfg['flush_size']),
                        compress=cfg['compress'],
                        normalize=cfg['normalize'])

if __name__ == '__main__':
    main()
