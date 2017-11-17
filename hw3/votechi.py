import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd
import csv as CSV_LIB
output_file = 'fuckyou.csv'
seafood = pd.read_csv('seafood.csv', sep=',', header=0)
okok1 = pd.read_csv('okok1.csv', sep=',', header=0)
ZZ = pd.read_csv('ZZ.csv', sep=',', header=0)
output = pd.read_csv('output.csv', sep=',', header=0)
voteanswer = pd.read_csv('voteanswer.csv', sep=',', header=0)

df_new = pd.concat([seafood.label, okok1.label, ZZ.label, output.label, voteanswer.label], axis = 1)
df_new = df_new.values

#for line in df_new:
#    print(np.argmax(np.bincount(line)))




fieldnames = ['id','label']
with open(output_file, 'w') as csvfile:
    writer = CSV_LIB.DictWriter(csvfile, fieldnames = fieldnames)
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