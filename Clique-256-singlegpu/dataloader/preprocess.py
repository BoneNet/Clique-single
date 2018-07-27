import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
import glob
import matplotlib._png as png
from scipy.misc import imresize
from imageio import imread


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    if 'data' in dict:
        dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)

    return dict

def load_data_one_10(f):
    batch = unpickle(f)
    data = batch['data']
    labels = batch['labels']
    print "Loading %s: %d" % (f, len(data))
    return data, labels

def load_data_one_100(f):
    batch = unpickle(f)
    data = batch['data']
    labels = batch['fine_labels']
    print "Loading %s: %d" % (f, len(data))
    return data, labels

def labels_to_one_hot(labels, classes):
    #Convert 1D array of labels to one hot representation
    
    new_labels = np.zeros((labels.shape[0], classes))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

def load_data10(phase, data_path):
    if phase=='train':
        files = [ 'data_batch_%d' % d for d in xrange(1, 6) ]
    else:
        files = ['test_batch']
        
    data, labels = load_data_one_10(data_path + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one_10(data_path + '/' + f)
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([ [ float(i == label) for i in xrange(10) ] for label in labels ])   ## cifar-10
    return data, labels

def load_data100(phase, data_path):
    if phase=='train':
        files = phase
    else:
        files = 'test'
        
    data, labels = load_data_one_100(data_path + '/' + files)
    labels=np.hstack(labels)
    labels = labels_to_one_hot(labels, 100)   ## cifar-100
    return data, labels

def cifar_preprocess(cifar_type, data_path):
    if cifar_type=='cifar-10':
        train_data, train_labels=load_data10('train', data_path)
        test_data, test_labels=load_data10('test', data_path)
    elif cifar_type=='cifar-100':
        train_data, train_labels=load_data100('train', data_path)
        test_data, test_labels=load_data100('test', data_path)
    
    return train_data, train_labels, test_data, test_labels

def svhn_preprocess(train_path, extra_path, test_path):
    train_file=os.path.join(train_path, 'train_32x32.mat')
    extra_file=os.path.join(extra_path, 'extra_32x32.mat')
    test_file=os.path.join(test_path, 'test_32x32.mat')
    ##
    raw_data=loadmat(train_file)
    train_data_0=raw_data['X'].transpose(3,0,1,2)
    train_labels_0=raw_data['y'].reshape(-1)
    train_labels_0[train_labels_0==10]=0
    train_labels_0=np.array([[float(i==label) for i in xrange(10)] for label in train_labels_0])
    ##
    raw_data=loadmat(extra_file)
    extra_data=raw_data['X'].transpose(3,0,1,2)
    extra_labels=raw_data['y'].reshape(-1)
    extra_labels[extra_labels==10]=0
    extra_labels=np.array([[float(i==label) for i in xrange(10)] for label in extra_labels])
    ##
    train_data=np.concatenate((train_data_0, extra_data), axis=0)
    train_labels=np.concatenate((train_labels_0, extra_labels), axis=0)   
    ##
    raw_data=loadmat(test_file)
    test_data=raw_data['X'].transpose(3,0,1,2)
    test_labels=raw_data['y'].reshape(-1)
    test_labels[test_labels==10]=0
    test_labels=np.array([[float(i==label) for i in xrange(10)] for label in test_labels])    
    
    return train_data, train_labels, test_data, test_labels

def mura_preprocess(train_path):
    train_path="MURA-v1.1/"
    csv_train_filess=os.path.join("dataloader", train_path, "train_labeled_studies.csv")
    csv_valid_filess=os.path.join("dataloader", train_path, "valid_labeled_studies.csv")
    
    train_df=pd.read_csv(csv_train_filess, names=['img', 'label'], header=None)
    valid_df=pd.read_csv(csv_valid_filess, names=['img', 'label'], header=None)
    
    
    # train_data_list=[]
    # train_labels=[]
    # for i in range(len(train_img_paths)):
    #     patient_dir=os.path.join("dataloader", train_img_paths[i])
    #     print("Loading: %s (%d/%d)"%(patient_dir, i, len(train_img_paths)))
    #     for f in glob.glob(patient_dir + "*"):
    #         train_data_patient=[]
    #         train_img=png.read_png_int(f).tolist()
    #         train_data_patient.append(train_img)
    #     train_data_list.extend(train_data_patient)
    #     for _ in range(len(train_data_patient)):
    #         train_labels.append(train_labels_patient[i])
    
    train_img_paths=train_df.img.values.tolist()
    valid_img_paths=valid_df.img.values.tolist()
    train_labels_patient=train_df.label.values.tolist()
    valid_labels_patient=valid_df.label.values.tolist()
    train_data_list=[]
    train_labels=[]
    valid_data_list=[]
    valid_labels=[]
    
    for i in range(len(train_img_paths)):
        patient_dir=os.path.join("dataloader", train_img_paths[i])
        print("Loading: %s (%d/%d)"%(patient_dir, i, len(train_img_paths)))
        for f in glob.glob(patient_dir + "*"):
            train_data_patient=[]
            train_img=png.read_png_int(f)
            train_img=imresize(train_img, (256, 256))
            train_img = np.stack((train_img,)*3, -1)
            train_data_patient.append(train_img)
        train_data_list.extend(train_data_patient)
        for _ in range(len(train_data_patient)):
            lst = [0, 0]
            lst[train_labels_patient[i]] = 1
            train_labels.append(lst)
    train_data=np.asarray(train_data_list)
    
    for i in range(len(valid_img_paths)):
        patient_dir=os.path.join("dataloader", valid_img_paths[i])
        print("Loading: %s (%d/%d)"%(patient_dir, i, len(valid_img_paths)))
        for f in glob.glob(patient_dir + "*"):
            valid_data_patient=[]
            valid_img=png.read_png_int(f)
            valid_img=imresize(valid_img, (256, 256))
            valid_img = np.stack((valid_img,)*3, -1)
            valid_data_patient.append(valid_img)
        valid_data_list.extend(valid_data_patient)
        for _ in range(len(valid_data_patient)):
            lst = [0, 0]
            lst[valid_labels_patient[i]] = 1
            valid_labels.append(lst)
    valid_data=np.asarray(valid_data_list)
    
    
    # return train_data, train_labels, valid_data, valid_labels

    return np.array(train_data), np.array(train_labels), np.array(valid_data), np.array(valid_labels)
    
    
    
