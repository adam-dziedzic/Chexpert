import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import getpass
import torch
import os
import random as rn


def set_environment():
    # fix random seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    torch.manual_seed(2019)
    torch.cuda.manual_seed(2019)
    torch.cuda.manual_seed_all(2019)
    torch.backends.cudnn.deterministic = True


def check_test_data():
    print('Check test data.')
    user = getpass.getuser()
    data_dir = f"/home/{user}/data/CheXpert-v1.0-small/"
    all_xray_df = pd.read_csv(data_dir + 'valid.csv')
    print('dev set size: ', all_xray_df.shape[0])
    pd.set_option('display.max_columns', None)
    print('print some dev samples:\n', all_xray_df.sample(3))


def prepare_data():
    print('Prepare data.')
    user = getpass.getuser()
    data_dir = f"/home/{user}/data/CheXpert-v1.0-small/"
    print(f'all dirs in {data_dir}: ', os.listdir(data_dir))
    all_xray_df = pd.read_csv(data_dir + 'train.csv')
    print('data set size: ', all_xray_df.shape[0])
    pd.set_option('display.max_columns', None)
    print('print some samples:\n', all_xray_df.sample(3))


if __name__ == "__main__":
    check_test_data()
    prepare_data()
