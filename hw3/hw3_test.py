import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import csv 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.models import load_model
input_file = sys.argv[1]
output_file = sys.argv[2]
# seafood = pd.read_csv('seafood.csv', sep=',', header=0)
# okok1 = pd.read_csv('okok1.csv', sep=',', header=0)
# ZZ = pd.read_csv('ZZ.csv', sep=',', header=0)
# output = pd.read_csv('output.csv', sep=',', header=0)
# voteanswer = pd.read_csv('voteanswer.csv', sep=',', header=0)

with open(input_file, 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y_test = []
    X_test = []
    for y in range(0,len(data),48*48+1):   
        X_test.append(data[y+1:y+48*48+1].reshape(48,48,1)) 
#X = np.array(X).astype(float)/255
X_test = np.array(X_test).astype(float)/255



cl_1 = load_model('model-1.h5')
cl_1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
predict = cl_1.predict(X_test)
predict = np.argmax(predict, axis=1)
voteanswer = predict
cl_2 = load_model('model-2.h5')

fieldnames = ['id','label']
pre_2 = np.argmax(cl_2.predict(X_test),axis=1)
cl_3 = load_model('model-3.h5')
voteanswer = np.c_[voteanswer,pre_2]
pre_3 = np.argmax(cl_3 .predict(X_test),axis=1)
voteanswer = np.c_[voteanswer,pre_3]


#for line in df_new:
#    print(np.argmax(np.bincount(line)))




fieldnames = ['id','label']
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writerow({'id': 'id', 'label': 'label'})
    for index,line in enumerate(voteanswer):
        a = int(np.argmax(np.bincount(line)))
        writer.writerow({'id': index, 'label':a })
#with open(output_path, 'w') as f:
# f.write('id,label\n')
# for i, v in  enumerate(predictions):
#   f.write('%d,%d\n' %(i+1, v))

#pd.merge(seafood, okok1, ZZ, output, voteanswer, on='id')

#print(df_new)

#result = pd.concat([df1, s1], axis=1)
