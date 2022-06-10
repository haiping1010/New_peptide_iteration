import tensorflow as tf
import random
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
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
from keras.models import  load_model

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
################################################
#################load training data##############
train_pos_path = 'all_data/random_train/????_?_learn.txt'
train_neg_path = 'all_data/random_train/*complex*_learn.txt'

col_size=40
row_size=300
pos_num = len(glob.glob(train_pos_path))
neg_num = len(glob.glob(train_neg_path))

print("pos_num", pos_num)
print("neg_num", neg_num)

train_pos_samples = np.zeros((pos_num,  row_size, col_size))
train_pos_labels = np.ones(pos_num)
train_neg_samples = np.zeros((neg_num, row_size ,col_size))
train_neg_labels = np.zeros(neg_num)
'''
index=0
for name in glob.glob(train_pos_path):
    print(name)
    t2=loadSplit(name)
    train_pos_samples[index,:,:] = t2
    index=index+1

index = 0
for name in glob.glob(train_neg_path):
    print(name)
    t2=loadSplit(name)
    train_neg_samples[index,:,:] = t2
    index=index+1

import h5py
with h5py.File('train_pos_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset1",  data=train_pos_samples)

import h5py
with h5py.File('train_neg_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset2",  data=train_neg_samples)
'''
import h5py
with h5py.File('train_pos_l.h5', 'r') as hf:
    train_pos_samples = hf['name-of-dataset1'][:]
with h5py.File('train_neg_l.h5', 'r') as hf:
    train_neg_samples = hf['name-of-dataset2'][:]

###################################################
###################load validation data ###########
val_pos_path = 'all_data/random_val/????_?_learn.txt'
val_neg_path = 'all_data/random_val/*complex*_learn.txt'

col_size=40
row_size=300

val_pos_num = len(glob.glob(val_pos_path))
val_neg_num = len(glob.glob(val_neg_path))

print("pos_num", val_pos_num)
print("neg_num", val_neg_num)

val_pos_samples = np.zeros((val_pos_num,  row_size, col_size))
val_pos_labels = np.ones(val_pos_num)
val_neg_samples = np.zeros((val_neg_num, row_size ,col_size))
val_neg_labels = np.zeros(val_neg_num)
'''
index=0
for name in glob.glob(val_pos_path):
    print(name)
    t2=loadSplit(name)
    val_pos_samples[index,:,:] = t2
    index=index+1

index = 0
for name in glob.glob(val_neg_path):
    print(name)
    t2=loadSplit(name)
    val_neg_samples[index,:,:] = t2
    index=index+1

import h5py
with h5py.File('val_pos_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset1",  data=val_pos_samples)

import h5py
with h5py.File('val_neg_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset2",  data=val_neg_samples)
'''
import h5py
with h5py.File('val_pos_l.h5', 'r') as hf:
    val_pos_samples = hf['name-of-dataset1'][:]
with h5py.File('val_neg_l.h5', 'r') as hf:
    val_neg_samples = hf['name-of-dataset2'][:]

###################################################
###################load test data ###########
test_pos_path = 'all_data/random_test/????_?_learn.txt'
test_neg_path = 'all_data/random_test/*complex*_learn.txt'

col_size=40
row_size=300

test_pos_num = len(glob.glob(test_pos_path))
test_neg_num = len(glob.glob(test_neg_path))

print("pos_num", test_pos_num)
print("neg_num", test_neg_num)

test_pos_samples = np.zeros((test_pos_num,  row_size, col_size))
test_pos_labels = np.ones(test_pos_num)
test_neg_samples = np.zeros((test_neg_num, row_size ,col_size))
test_neg_labels = np.zeros(test_neg_num)
'''
index=0
for name in glob.glob(test_pos_path):
    print(name)
    t2=loadSplit(name)
    test_pos_samples[index,:,:] = t2
    index=index+1

index = 0
for name in glob.glob(test_neg_path):
    print(name)
    t2=loadSplit(name)
    test_neg_samples[index,:,:] = t2
    index=index+1

import h5py
with h5py.File('test_pos_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset1",  data=test_pos_samples)

import h5py
with h5py.File('test_neg_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset2",  data=test_neg_samples)
'''
import h5py
with h5py.File('test_pos_l.h5', 'r') as hf:
    test_pos_samples = hf['name-of-dataset1'][:]
with h5py.File('test_neg_l.h5', 'r') as hf:
    test_neg_samples = hf['name-of-dataset2'][:]

###################################################
###################load test data ###########





X_train, Y_train = synData(train_pos_samples, train_pos_labels, train_neg_samples, train_neg_labels, 9)


X_val, Y_val = synData(val_pos_samples, val_pos_labels, val_neg_samples, val_neg_labels, 9)


X_test = np.vstack((test_pos_samples, test_neg_samples))
Y_test = np.hstack((test_pos_labels, test_neg_labels))

print("shape of test data: ", X_test.shape, Y_test.shape)
print("shape of training data", X_train.shape, Y_train.shape)

X_train = X_train.reshape(X_train.shape[0],  row_size, col_size, 1)
X_val = X_val.reshape(X_val.shape[0],  row_size, col_size, 1)

X_test = X_test.reshape(X_test.shape[0], row_size, col_size, 1)
print("shape of training set: ", X_train.shape, '   ', Y_train.shape)
print("shape of training set: ", X_val.shape, '   ', Y_val.shape)


print("shape of test set: ", X_test.shape, '   ', Y_test.shape)

print("test proportion: ", (len(Y_test) * 1.0) / (len(Y_train) + len(Y_test)))
print("proportion of negative test samples", (np.sum(Y_test == 0) * 1.0) / len(Y_test))
print("proportion of negative train samples", (np.sum(Y_train == 0) * 1.0) / len(Y_train))
	
#*************************************jupyter_notebook*****************************************


model2 = Sequential()
##first layer
model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (300,40,1)))
model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2), strides = 2))
model2.add(Dropout(0.25))

##second layer

model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model2.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model2.add(Dropout(0.25))


##fully layer

model2.add(Flatten())
model2.add(Dense(256, activation = "relu"))
model2.add(Dropout(0.5))

###softmax
#model2.add(Dense(10, activation = "softmax"))


#model2.add(Dense(2, activation = "softmax"))
#sigmoid
model2.add(Dense(1, activation = "sigmoid"))

model2.summary()

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
###
###ategorical_crossentropy
model2.compile(optimizer = optimizer , loss = "binary_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
epochs = 60 
batch_size = 64

history = model2.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 120, validation_data = (X_val, Y_val), callbacks = [learning_rate_reduction])
model2.save("model_one_hot_PP60.h5")


#model2=load_model("model_one_hot_PP60.h5")

print (history.history.keys())
#print(history.history['val_loss'])
fww=open('acc_check.txt', 'w')

i=0
for i in range(len(history.history['acc'])):
    index=i+1
    fww.write(str(index)+'  '+str(history.history['acc'][i])+' ' + str(history.history['val_acc'][i])+     '  '+str(history.history['loss'][i])+' ' + str(history.history['val_loss'][i]) + "\n")

model2=load_model("model_one_hot_PP60.h5")

def acc(true_n, pred):
    
    return np.sum(true_n == pred) * 1.0 / len(true_n)



###############################################
#####################training dataset #############


y_pred_m = model2.predict(X_train)

y_pred_m=y_pred_m.flatten()
auc = aucJ(Y_train, y_pred_m)

print('auc: ', auc)

print('test:')
threshold = 0.5
y_pred_m[y_pred_m > threshold] = 1
y_pred_m[y_pred_m <= threshold] = 0


from sklearn.metrics import accuracy_score
##accuracy=accuracy_score(Y_train, y_pred_m)

from sklearn.metrics import matthews_corrcoef


accuracy = acc(Y_train, y_pred_m)

print('accuracy', accuracy)


pos_index = y_pred_m == 1
pos_index = pos_index.flatten()
TPR = np.sum(Y_train[pos_index] == 1) * 1.0 / np.sum(Y_train)
print("TPR: ", TPR)

precision= np.sum(Y_train[pos_index] == 1) * 1.0 / np.sum(y_pred_m)
print("precision: ", precision)

MCC=matthews_corrcoef(Y_train, y_pred_m)
print("MCC: ", MCC)

print("pos_num", train_pos_samples.shape[0])
print("neg_num", train_neg_samples.shape[0])

fw=open("output.csv",'w')
line=str(auc)+" "+str(accuracy)+" "+str(TPR)+" "+str(precision)+" "+str(MCC)+" "+str(test_pos_samples.shape[0]) +" "+str(test_neg_samples.shape[0])

fw.write(line)

fw.write("\n")
###############################################
#####################validate#################

y_pred_m = model2.predict(X_val)
y_pred_m=y_pred_m.flatten()

auc = aucJ(Y_val, y_pred_m)

print('auc: ', auc)

print('test:')
threshold = 0.5
y_pred_m[y_pred_m > threshold] = 1
y_pred_m[y_pred_m <= threshold] = 0


from sklearn.metrics import accuracy_score
##accuracy=accuracy_score(Y_val, y_pred_m)

from sklearn.metrics import matthews_corrcoef


accuracy = acc(Y_val, y_pred_m)

print('accuracy', accuracy)


pos_index = y_pred_m == 1
pos_index = pos_index.flatten()
TPR = np.sum(Y_val[pos_index] == 1) * 1.0 / np.sum(Y_val)
print("TPR: ", TPR)

precision= np.sum(Y_val[pos_index] == 1) * 1.0 / np.sum(y_pred_m)
print("precision: ", precision)

MCC=matthews_corrcoef(Y_val, y_pred_m)
print("MCC: ", MCC)

print("pos_num", val_pos_samples.shape[0])
print("neg_num", val_neg_samples.shape[0])

line=str(auc)+" "+str(accuracy)+" "+str(TPR)+" "+str(precision)+" "+str(MCC)+" "+str(test_pos_samples.shape[0]) +" "+str(test_neg_samples.shape[0])

fw.write(line)
fw.write("\n")
###############################################
#####################test dataset #############
y_pred_m = model2.predict(X_test)
y_pred_m=y_pred_m.flatten()

auc = aucJ(Y_test, y_pred_m)

print('auc: ', auc)

print('test:')
threshold = 0.5
y_pred_m[y_pred_m > threshold] = 1
y_pred_m[y_pred_m <= threshold] = 0


from sklearn.metrics import accuracy_score
##accuracy=accuracy_score(Y_test, y_pred_m)

from sklearn.metrics import matthews_corrcoef


accuracy = acc(Y_test, y_pred_m)

print('accuracy', accuracy)


pos_index = y_pred_m == 1
pos_index = pos_index.flatten()
TPR = np.sum(Y_test[pos_index] == 1) * 1.0 / np.sum(Y_test)
print("TPR: ", TPR)

precision= np.sum(Y_test[pos_index] == 1) * 1.0 / np.sum(y_pred_m)
print("precision: ", precision)

MCC=matthews_corrcoef(Y_test, y_pred_m)
print("MCC: ", MCC)

print("pos_num", test_pos_samples.shape[0])
print("neg_num", test_neg_samples.shape[0])


line=str(auc)+" "+str(accuracy)+" "+str(TPR)+" "+str(precision)+" "+str(MCC)+" "+str(test_pos_samples.shape[0]) +" "+str(test_neg_samples.shape[0])
fw.write("\n")
fw.write(line)

fw.close()
