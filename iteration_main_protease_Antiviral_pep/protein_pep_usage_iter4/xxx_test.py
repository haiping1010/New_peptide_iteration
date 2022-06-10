import tensorflow as tf
import random
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.models import load_model
from keras.optimizers import RMSprop
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers import  MaxPool2D
from keras.layers import  Softmax, Dropout, Flatten
import pandas as pd
import numpy as np
import glob
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from pandas import DataFrame
import sys
np.set_printoptions(threshold=sys.maxsize)
random.seed(1)
#*************************************jupyter_notebook*****************************************


def loadSplit(path):
    t = np.loadtxt(path,dtype=np.str)
    t1= []        
    for i in range(len(t)):
        t1.append([int(x) for x in list(t[i])])               
    output = np.array(t1)
    return output

def aucJ(true_labels, predictions):
    
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr,tpr)

    return auc

def randomShuffle(X, Y):
    idx = [t for t in range(X.shape[0])]
    random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    print()
    print('-' * 36)
    print('dimension of X after synthesis:', X.shape)
    print('dimension of Y after synthesis', Y.shape)
    print('label after shuffle:', '\n', DataFrame(Y).head())
    print('-' * 36)
    return X, Y

def synData(X_0, Y_0, X_1, Y_1, time):

    X_0_syn = X_0
    Y_0_syn = Y_0
    for i in range(time - 1):
        X_0_syn = np.vstack( (X_0_syn, X_0) )
        Y_0_syn = np.hstack( (Y_0_syn, Y_0) )

    print('dimension of generation data of X', X_0_syn.shape)
    print('dimension of generation data of Y', Y_0_syn.shape)
    print('dimension of generation data of X with label of 1', X_1.shape)
    print('dimension of generation data of Y with label of 1', Y_1.shape)

    #synthesis dataset
    X_syn = np.vstack( (X_0_syn, X_1) )
    Y_syn = np.hstack( (Y_0_syn, Y_1) )

    print()
    print('dimension of X after combination', X_syn.shape)
    print('dimension of Y after combination', Y_syn.shape)
    print(DataFrame(Y_syn).head())

    #shuffle data
    X_syn, Y_syn = randomShuffle(X_syn, Y_syn)
    
    return X_syn, Y_syn

pos_path = 'docking_complex/*_learn.txt'

col_size=40
row_size=300

pos_num = len(glob.glob(pos_path))

print("pos_num", pos_num)

pos_samples = np.zeros((pos_num,  row_size, col_size))
pos_labels = np.ones(pos_num)

pos_names=[]

print(pos_samples.shape)

index=0
for name in glob.glob(pos_path):
    print(name)
    t2=loadSplit(name)
    pos_samples[index,:,:] = t2
    index=index+1
    name=name.replace('docking_complex/', '')
    name=name.replace('_learn.txt','')
    pos_names.append(name)
    #print name


