import numpy as np
import sys
import os
from scipy.io import loadmat
import pandas as pd
import glob
import matplotlib._png as png
from scipy.misc import imresize
from imageio import imread


def mura_preprocess(train_path):
    train_path = "MURA-v1.1/"
    csv_train_filess = os.path.join("dataloader", train_path, "train_labeled_studies.csv")
    csv_valid_filess = os.path.join("dataloader", train_path, "valid_labeled_studies.csv")

    train_df = pd.read_csv(csv_train_filess, names=['img', 'label'], header=None)
    valid_df = pd.read_csv(csv_valid_filess, names=['img', 'label'], header=None)

    train_img_paths = train_df.img.values.tolist()
    valid_img_paths = valid_df.img.values.tolist()
    train_labels_patient = train_df.label.values.tolist()
    valid_labels_patient = valid_df.label.values.tolist()
    train_data_list = []
    train_labels = []
    valid_data_list = []
    valid_labels = []

    for i in range(len(train_img_paths)):
        patient_dir = os.path.join("dataloader", train_img_paths[i])
        msg = "\r Loading: %s (%d/%d)    " % (patient_dir, i + 1, len(train_img_paths))
        sys.stdout.write(msg)
        sys.stdout.flush()
        for f in glob.glob(patient_dir + "*"):
            train_data_patient = []
            train_img = png.read_png_int(f)
            train_img = imresize(train_img, (128, 128))
            train_img = train_img[:,:,np.newaxis]
            # you can replace 256 with other number but any number greater then 256 will exceed the memory limit of 12GB
            train_data_patient.append(train_img)
        train_data_list.extend(train_data_patient)
        for _ in range(len(train_data_patient)):
            lst = [0, 0]
            lst[train_labels_patient[i]] = 1
            train_labels.append(lst)
    train_data = np.asarray(train_data_list)

    for i in range(len(valid_img_paths)):
        patient_dir = os.path.join("dataloader", valid_img_paths[i])
        msg = "\r Loading: %s (%d/%d)     " % (patient_dir, i + 1, len(valid_img_paths))
        sys.stdout.write(msg)
        sys.stdout.flush()
        for f in glob.glob(patient_dir + "*"):
            valid_data_patient = []
            valid_img = png.read_png_int(f)
            valid_img = imresize(valid_img, (128, 128))
            valid_img = valid_img[:,:,np.newaxis]
            valid_data_patient.append(valid_img)
        valid_data_list.extend(valid_data_patient)
        for _ in range(len(valid_data_patient)):
            lst = [0, 0]
            lst[valid_labels_patient[i]] = 1
            valid_labels.append(lst)
    valid_data = np.asarray(valid_data_list)

    return np.array(train_data), np.array(train_labels), np.array(valid_data), np.array(valid_labels)
