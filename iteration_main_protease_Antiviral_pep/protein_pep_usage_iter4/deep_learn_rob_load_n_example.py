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
    #print(name)
    t2=loadSplit(name)
    pos_samples[index,:,:] = t2
    index=index+1
    name=name.replace('docking_complex/', '')
    name=name.replace('_learn.txt','')
    pos_names.append(name)
    #print name


        
print(pos_samples.shape)

import h5py
with h5py.File('pos_l.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset1",  data=pos_samples)

np.save("filename.npy",np.array(pos_names))
#b = np.load("filename.npy")

import h5py
with h5py.File('pos_l.h5', 'r') as hf:
    pos_samples = hf['name-of-dataset1'][:]


print(pos_samples.shape)

#pos_samples, pos_labels = randomShuffle(pos_samples, pos_labels)

X_test = pos_samples[:, :, :]
Y_test = pos_labels[:]

print("shape of test data: ", X_test.shape, Y_test.shape)
'''
atrain_index = np.random.randint(0,5280)
test_index = np.random.randint(0,1467)
print("train_sample:\n", X_train[train_index,:,:])
print("test_sample:\n", X_test[test_index,:,:])
'''

X_test = X_test.reshape(X_test.shape[0], row_size, col_size, 1)


print("shape of test set: ", X_test.shape, '   ', Y_test.shape)

	
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

#history = model2.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 120, validation_data = (X_val, Y_val), callbacks = [learning_rate_reduction])
#model2.save("model_one_hot_PP60.h5")

model2=load_model("model_one_hot_PP60.h5")




pred = model2.predict(X_test)

pred = pred.flatten()

print (pos_names)


outcontent=[]
for i in  range(len(pos_names)):
        outcontent.append(pos_names[i]+'  '+str(pred[i]))

outcontent=DataFrame(outcontent, columns=['all'])
new = outcontent['all'].str.split(" ", n = 1, expand = True)

outcontent2=DataFrame()

print (new)
outcontent2["name"]= new[0]
outcontent2["prediction"]= new[1].astype('float')
result=outcontent2.sort_values(by=["prediction"],ascending=False)

result.to_csv('out_list.csv', index = False)



