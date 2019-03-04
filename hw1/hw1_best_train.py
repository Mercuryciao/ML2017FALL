import csv
import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
import math
import sys
import matplotlib.pyplot as plt

data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0
text = open('C:/Users/Vicky/Desktop/ML/hw1/train.csv', 'r')
#text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

x = []
y = []
lamda = 0.01
#train_x=[]
#train_x[w]=x.extend(data[lista[w]][j+(480*i):j+(480*i)+9])
#lista=[3,6,7,8,9]
for i in range(12):
#    for w in range(5):
    for j in range(471): 
        x.append(data[9][j+(480*i):j+(480*i)+9])
        y.append(data[9][j+(480*i)+9])

x = np.array(x)
y = np.array(y)
#x.extend(data[9][j+(480*i):j+(480*i)+9],data[3][j+(480*i):j+(480*i)+9])
# add square term
x = np.concatenate((x,x**2), axis=1)
# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 1
repeat = 10000


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2) / len(x)+lamda*sum(w**2)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)+lamda*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
#    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)), x.transpose()), y)

test = pd.read_csv('C:/Users/Vicky/Desktop/ML/hw1/test.csv',header = None)

test = test[test.iloc[:,1]=='PM2.5']
name = test[0]
test = test.iloc[0:,2:]
test = test.astype(float)
test = np.array(test)
test = np.concatenate((test, test**2), axis=1)
row = np.ones(240)
test = np.c_[row, test]
test = test.astype(float)

result = np.dot(test, w)
result_t = result.transpose()
np_data = np.c_[name, result_t]

filename = "C:/Users/Vicky/Desktop/ML/hw1/predict9.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter = ',')
s.writerow(["id","value"])
for i in range(len(np_data)):
    s.writerow(np_data[i]) 
text.close()

##作圖
#X = np.linspace(-np.pi, np.pi, 200)
#C, S = np.cos(X), np.sin(X)
#plt.plot(X,C, label='sin')
#plt.plot(X,S, label='cos')
#plt.xlim(X.min(), X.max())
#plt.yticks([-1, 0, +1])
#plt.legend(loc='upper left')
#plt.show()
