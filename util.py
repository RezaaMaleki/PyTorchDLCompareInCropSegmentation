"""

Util file for misc functions

"""
from constants import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse
import pickle
import pandas as pd
import time
import random
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from constants import *
import torch

def random_seed(seed_value, use_cuda):
    if seed_value is not None:
        np.random.seed(seed_value) # cpu vars
        torch.manual_seed(seed_value) # cpu  vars
        random.seed(seed_value) # Python
        if use_cuda: 
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value) # gpu vars
            torch.backends.cudnn.deterministic = True  #needed
            torch.backends.cudnn.benchmark = False

def dates2doy(dates):
    """ Transforms list of dates in YYYY-MM-DD format to a vector of days of year
    """
    data = []
    # convert each date to doy
    for date in dates:
        y, m, d = date.split('-')
        doy = datetime(int(y), int(m), int(d)).timetuple().tm_yday
        data.append(doy)
    # save dates as npy array
    data = np.array(data)
    return data

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def softmax(x):
    """
    Computes softmax values for a vector x.

    Args: 
      x - (numpy array) a vector of real values

    Returns: a vector of probabilities, of the same dimensions as x
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def create_categorical_df_col(df, from_col, to_col):
    """
    Creates a categorical column in a dataframe from an existing column

    For example, column 'classes' of possibilities 'cat', 'dog', 'bird'
    can be categorized in a new column 'class_nums' of possibilities 0, 1, 2.

    Args:
        df - pandas data frame
        from_col - (str) specifies column name that you wish to categorize 
                         into integer categorical values
        to_col - (str) specifies column name that will be added with the
                       new categorical labels

    Returns: 
       df - pandas data frame with additional column of categorical labels
    """
    df[from_col] = pd.Categorical(df[from_col])
    df[to_col] = df[from_col].astype('category').cat.codes
    return df

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return fig

def split_with_group(df, group, train_frac, test_frac, data_cols, lbl_cols, random_seed=None, shuffle=True, save=False):
    """
    Splits a dataframe into train, val, and test splits while keeping groups
    separated between splits. 

    For example, a data frame may contain the column 'poly_ID' that should be 
    kept separated between dataset splits. 

    train_frac + test_frac must be <= 1. When < 1, the reaminder of the dataset
    goes into a validation set

    Args:
        df - (pandas dataframe) the dataframe to be split into train, val, test splits
        group - (str) the column name to separate by
        train_frac - (float) percentage between 0-1 to put into the training set (train_frac + test_frac <= 1)
        test_frac - (float) percentage between 0-1 to put into the test set (train_frac + test_frac <= 1)
        data_cols - (indexed column(s), i.e. 3:-1) the column(s) of the data frame that contain the data 
        lbl_cols - (int, i.e. -1) the column of the data frame that contains the labels
        random_seed - (int) when splitting and if shuffling the dataset after splitting, use this random_seed to do so
        shuffle - (boolean) if True, shuffle the dataset once it's already split
        save - (boolean) if True, save output splits into a pickle file
    
    Returns:
        X_train - (np.ndarray) training data
        y_train - (np.ndarray) training labels
        X_val - (np.ndarray) validation data
        y_val - (np.ndarray) validation labels
        X_test - (np.ndarray) test data
        y_test - (np.ndarray) test labels
    """

    X = df
    groups = df[group]

    train_inds, test_inds = next(GroupShuffleSplit(n_splits=3, test_size=test_frac, 
                                 train_size=train_frac, random_state=random_seed).split(X, groups=groups))

    val_inds = []
    for i in range(X.shape[0]):
        if i not in train_inds and i not in test_inds:
            val_inds.append(i)

    val_inds = np.asarray(val_inds) 

    if random_seed:
        np.random.seed(random_seed)
    
    if shuffle:
        np.random.shuffle(train_inds)
        np.random.shuffle(val_inds)
        np.random.shuffle(test_inds)

    X_train, y_train = X.values[train_inds, data_cols], X.values[train_inds, lbl_cols].astype(int)
    X_val, y_val = X.values[val_inds, data_cols], X.values[val_inds, lbl_cols].astype(int)
    X_test, y_test = X.values[test_inds, data_cols], X.values[test_inds, lbl_cols].astype(int)

    if save:
        fname = '_'.join('dataset_splits', time.strftime("%Y%m%d-%H%M%S"), '.pickle') 
        with open(fname, "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)

    return X_train, y_train, X_val, y_val, X_test, y_test

    
def crop_ind(y, name_list = [1, 2, 3, 4, 5]):
    """
    Crop row Index for y, just interested in some croptypes
    Args:
        y - y_label (croptype) of train/val/test or pixel arrays
        name_list - what croptypes we are interested in, normally 1-5, just top 5 crops in the country

        Returns:
        crop_index - the index of array or vector to indicate where are the places for the croptype from 1-5
    """

    crop_index = [name in name_list for name in y]
    crop_index = np.where(crop_index)
    return crop_index


def get_y_label(home, country, data_set, data_type, ylabel_dir, raster_npy_dir):
    """
    Get y label for different set small/full, different type train/val/test
    
    Args:
      home - (str) the base directory of data
      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'
      data_set - (str) balanced 'small' or unbalanced 'full' dataset
      data_type - (str) 'train'/'val'/'test'
      ylabel_dir - (str) dir to save ylabel
      raster_npy_dir - (str) string for the mask raster dir 'raster_npy' or 'raster_64x64_npy'

    Output: 
    ylabel_dir/..

    save as grid_nums*row*col 3D array
    """
    gridded_IDs = sorted(np.load(os.path.join(home, country, country+'_'+data_set+'_'+data_type)))

    # Match the Mask
    mask_dir = os.path.join(home, country, raster_npy_dir)
    mask_fnames = [country+'_64x64_'+gridded_ID+'_label.npy' for gridded_ID in gridded_IDs]

    # Geom_ID Mask Array
    mask_array = np.zeros((len(gridded_IDs),64,64))

    for i in range(len(gridded_IDs)): 
        fname = os.path.join(mask_dir,mask_fnames[i])
        # Save Mask as one big array
        mask_array[i,:,:] = np.load(fname)[0:64,0:64]

    output_fname = "_".join([data_set, data_type, 'croptypemask', 'g'+str(len(gridded_IDs)),'r64', 'c64'+'.npy'])

    np.save(os.path.join(ylabel_dir,output_fname), mask_array)
    
    return mask_array


def mask_tif_npy(home, country, csv_source, crop_dict_dir, raster_dir):
    """
    Transfer cropmask from .tif files by field_id to cropmask .npy by crop_id given crop_dict
    
    Args:
      home - (str) the base directory of data
      country - (str) string for the country 'Ghana', 'Tanzania', 'SouthSudan'
      csv_source - (str) string for the csv field id file corresponding with the country
      crop_dict_dir - (str) string for the crop_dict dictionary {0: 'unlabeled', 1: 'groundnuts' ...}
      raster_dir - (str) string for the mask raster dir 'raster' or 'raster_64x64'

    Outputs:
      ./raster_npy/..

    """
    fname = os.path.join(home, country, csv_source)
    crop_csv = pd.read_csv(fname)

    mask_dir = os.path.join(home, country, raster_dir)
    mask_dir_npy = os.path.join(home, country, raster_dir+'_npy')
    mask_fnames = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    mask_ids = [f.split('_')[-1].replace('.tif', '') for f in mask_fnames]
    mask_fnames = [mask_fnames[ID] for ID in np.argsort(mask_ids)]
    mask_ids = np.array([mask_ids[ID] for ID in np.argsort(mask_ids)])

    crop_dict = np.load(os.path.join(home, crop_dict_dir))
    clustered_geom_id = [np.array(crop_csv['geom_id'][crop_csv['crop']==crop_name]) for crop_name in crop_dict.item().values()]

    for mask_fname in mask_fnames: 
        with rasterio.open(os.path.join(mask_dir,mask_fname)) as src:
            mask_array = src.read()[0,:,:]
            mask_array_geom_id = np.unique(mask_array)
            mask_array_crop_id = np.zeros(mask_array.shape)
            mask_array_crop_id[:] = np.nan
            for geom_id in mask_array_geom_id:
                if geom_id>0:
                    crop_num = np.where([geom_id in clustered_geom_id[i] for i in np.arange(len(clustered_geom_id))])[0][0]
                    mask_array_crop_id[mask_array==geom_id] = crop_num
                elif geom_id == 0:
                    mask_array_crop_id[mask_array==geom_id] = 0
            np.save(os.path.join(mask_dir_npy,mask_fname.replace('.tif', '.npy')), mask_array_crop_id)


def fill_NA(X):
    """
    Fill NA values with mean of each band

    Args: 
      X - (numpy array) a vector of real values

    Returns: numpy array the same dimensions as x, no NAs
    """
    X_noNA = np.where(np.isnan(X), ma.array(X, mask=np.isnan(X)).mean(axis=0), X) 
    return(X_noNA)


def weights_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or \
        isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.ConvTranspose2d):
        m.reset_parameters()

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or \
        isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)