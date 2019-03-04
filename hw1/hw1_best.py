import csv
import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
import math
import sys
import matplotlib.pyplot as plt

input_filename = sys.argv[1]
output_filename = sys.argv[2]
w = [1.7145319126569958,-0.002860669430699625,-0.05345213453658307,0.21457186005484863,-0.23216594213253788,-0.059665332726959744,0.517424753657096,-0.4909913432322578,-0.030206252005096135,1.0581867263421896,-0.0004975551361940476,0.00042195579405514294,8.431843139169353e-05,-0.00010397880007682794,0.00011052482899671597,0.00018145861490316377,-0.0013218353239959504,0.0005191855370287159,0.000580532258674874]

test = pd.read_csv(input_filename,header=None)
test = test[test.iloc[:,1]=='PM2.5']
name = test[0]

test = test.iloc[0:,2:]
test = test.astype(float)
test = np.array(test)
test = np.concatenate((test,test**2), axis=1)
row = np.ones(240)
test = np.c_[row, test]
test = test.astype(float)

result = np.dot(test, w)
result_t = result.transpose()
np_data = np.c_[name, result_t]

filename = output_filename
text = open(filename, "w+")
s = csv.writer(text, delimiter=',')
s.writerow(["id", "value"])
for i in range(len(np_data)):
    s.writerow(np_data[i]) 
text.close()
