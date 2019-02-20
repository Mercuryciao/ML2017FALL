import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import csv as CSV_LIB

output_file = 'vote_result.csv'
# seafood = pd.read_csv('seafood.csv', sep=',', header=0)
# okok1 = pd.read_csv('okok1.csv', sep=',', header=0)
# ZZ = pd.read_csv('ZZ.csv', sep=',', header=0)
# output = pd.read_csv('output.csv', sep=',', header=0)
# voteanswer = pd.read_csv('voteanswer.csv', sep=',', header=0)

cl_1 = load_model('model-1')
cl_1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
predictions = cl_1.predict(X)
predictions = np.argmax(predictions, axis=1)
voteanswer = predictions
cl_2 = load_model('model-2')

fieldnames = ['id','label']
pre_2 = np.argmax(cl_2.predict(X),axis=1)
cl_3 = load_model('model-3')
voteanswer = np.c_[voteanswer,pre_2]
pre_3 = np.argmax(cl_3 .predict(X),axis=1)
voteanswer = np.c_[voteanswer,pre_3]


#for line in df_new:
#    print(np.argmax(np.bincount(line)))




fieldnames = ['id','label']
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writerow({'id': 'id', 'label': 'label'})
    for index,line in enumerate(df_new):
        a = int(np.argmax(np.bincount(line)))
        writer.writerow({'id': index, 'label':a })
#with open(output_path, 'w') as f:
#	f.write('id,label\n')
#	for i, v in  enumerate(predictions):
#		f.write('%d,%d\n' %(i+1, v))

#pd.merge(seafood, okok1, ZZ, output, voteanswer, on='id')

#print(df_new)

#result = pd.concat([df1, s1], axis=1)
