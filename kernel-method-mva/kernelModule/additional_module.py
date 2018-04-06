import numpy as np
import pandas as pd
import time
import datetime
from contextlib import contextmanager

def data_process(path):
    """
    Import and preprocessed data (like transform label to become -1 and 1)

    :param path: path to the data folder containing train and test dataset
    :return: preprocessed data
    """
    x0_train = pd.read_csv(path + 'Xtr0.csv', header=None, names=['sequence'])
    x0_test = pd.read_csv(path + 'Xte0.csv', header=None, names=['sequence'])
    y0_train = pd.read_csv(path + 'Ytr0.csv')

    x1_train = pd.read_csv(path + 'Xtr1.csv', header=None, names=['sequence'])
    x1_test = pd.read_csv(path + 'Xte1.csv', header=None, names=['sequence'])
    y1_train = pd.read_csv(path + 'Ytr1.csv')

    x2_train = pd.read_csv(path + 'Xtr2.csv', header=None, names=['sequence'])
    x2_test = pd.read_csv(path + 'Xte2.csv', header=None, names=['sequence'])
    y2_train = pd.read_csv(path + 'Ytr2.csv')

    x0 = x0_train.iloc[:, 0].values.astype(np.str)
    y0 = y0_train.iloc[:, 1].values.astype(float)
    x0_te = x0_test.iloc[:, 0].values.astype(np.str)

    # convert negative example to have label -1 for SVM
    y0[y0 == 0] = -1.0

    x1 = x1_train.iloc[:, 0].values.astype(np.str)
    y1 = y1_train.iloc[:, 1].values.astype(float)
    x1_te = x1_test.iloc[:, 0].values.astype(np.str)

    # convert negative example to have label -1 for SVM
    y1[y1 == 0] = -1.0

    x2 = x2_train.iloc[:, 0].values.astype(np.str)
    y2 = y2_train.iloc[:, 1].values.astype(float)
    x2_te = x2_test.iloc[:, 0].values.astype(np.str)

    # convert negative example to have label -1 for SVM
    y2[y2 == 0] = -1.0
    thres = 1800
    xtrain = np.concatenate([x0[:thres], x1[:thres], x2[:thres]])
    ytrain = np.concatenate([y0[:thres], y1[:thres], y2[:thres]])
    xtest = np.concatenate([x0_te, x1_te, x2_te])

    # extracted from x1_train
    cv_lower, cv_upper = [thres, 2000]  # index of cross_validation data with label

    xcv_sand = np.concatenate([x0[cv_lower:cv_upper], x1[cv_lower:cv_upper], x2[cv_lower:cv_upper]])
    ycv_sand = np.concatenate([y0[cv_lower:cv_upper], y1[cv_lower:cv_upper], y2[cv_lower:cv_upper]])

    return xtrain, ytrain, xtest, xcv_sand, ycv_sand

@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    
    Calculate time taken for running code
    """
    t0 = time.time()
    yield
    print('[{:}] done in {:.8f}s'.format(name, time.time() - t0))


# for saving csv with named according to hour and date
def save_csv(dataframe, file_begin, cols_name=True, fl_format='%.8f', state_index=False):
    '''For saving and storing prediction file systematically
    '''
    print('save', file_begin ,'file...')
    dataframe.to_csv(path_or_buf= file_begin +
                     datetime.datetime.now().strftime("-%y%m%d-%H%M%S") + '.csv',
                     index=state_index,
                     float_format=fl_format,
                     header=cols_name)
    print('finish saving!')
