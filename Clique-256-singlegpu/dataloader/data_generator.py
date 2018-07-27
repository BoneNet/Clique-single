import sys
import os
from urllib import urlretrieve 
import tarfile
import zipfile
from preprocess import cifar_preprocess, svhn_preprocess, mura_preprocess
import numpy as np


def data_normalization(train_data_raw, test_data_raw, normalize_type):
    if normalize_type == 'divide-255':
        train_data = train_data_raw / 255.0
        test_data = test_data_raw / 255.0

        return train_data, test_data

    elif normalize_type == 'None':

        return train_data_raw, test_data_raw


def load_data():
    data_path = "MURA-v1.1"
    train_data_raw, train_labels, test_data_raw, test_labels = mura_preprocess(data_path)
    normalize_type = 'divide-255'
    train_data, test_data = data_normalization(train_data_raw, test_data_raw, normalize_type)
    return train_data, train_labels, test_data, test_labels
