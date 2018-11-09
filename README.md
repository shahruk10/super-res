# Super-resolution imaging using DNNs

## Installing dependencies

- Anaconda3 is recommended as the default python enviroment. 

- Run `pip` from the terminal/command prompt on the requirement file to install the needed dependencies. For cpu based tensorflow :

  ```console
  foo@foobar:~$ pip install -r requirements-cpu.txt
  ```
- For Tensorflow GPU support, first need to install CUDA 9.0 and CuDNN 7.1. See https://www.tensorflow.org/install/gpu for instructions. Then run :
  ```console
  foo@foobar:~$ pip install -r requirements-gpu.txt
  ```
## Generating database

- First, need to create `hdf5` databses which are very space efficient and quick to read from disk

- Use `makeHDF5.py` to create these files. You must provide the path to the `config_db.toml` file which contains the settings for your database

    ```console
    foo@foobar:~$ python makeHDF5.py ./config_db.toml
    ```
- Some key settings inside the `config_db.toml` file :

  - `image_sets` ~ a list of paths to folders containing the images to include in the database

  - `kfold`  ~ the number of cross folds. If kfold is set to 5, the script will generate 5 `hdf5` files each with randomly distributed patches from different image files. Patches from any image are found only in one `hdf5` file.

  - `num_patches` ~ the number of patches to take from each image

  - `original_patches` ~ the crop size for each patch. Can list several sizes, but they must be smaller than original image size. All patches are square.

  - `resized_patches` ~ the size to which each of the original patches are resized to. Can list several sizes, but only sizes where original size is greater will be taken into account.


  - `normalize` ~ if set to `true`, all image patches will be scaled to values between 0.0 and 1.0. 

## Defining models

- Models are defined in `modelLib.py` 

- Some configuration of the model like input size and output size etc. can be parameterized and set from `config_train.toml` for fast experimentation

## Training models

- Once you have `hdf5` files and models defined, use `train.py` to train your models

  ```console
  foo@foobar:~$ python train.py ./config_train.toml
  ```

- Some key settings inside the `config.toml are :

  - `x` ~ this is the name of the database entry in the `hdf5` files to use as input (e.g. `org_64_rsz_32` for patches of size 64 x 64 from the original image which were resized to 32 x 32)

  - `y` ~ this is the name of the database entry in the `hdf5` files to use as the target for the model (e.g. `org_64` for patches of size 64 x 64)

  - `train` ~ list of paths to `hdf5` files which will be used as training set. If more than one path provided, cross validation will be performed

  - `val` ~ list of paths to `hdf5` files which will override the default test set during cross validation

  - `modelArch` ~ the name of the model as defined in `modelLib.py`

  - `modelDir` and `modelName` ~ path at which to save model weights and other results 
