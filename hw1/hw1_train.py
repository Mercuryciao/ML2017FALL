import csv
import pandas as pd
import numpy as np
import math

data = pd.read_csv('C:/Users/Vicky/Desktop/ML/hw1/train.csv',encoding = 'gb18030')
data = data.rename(columns = {'代兜': 'item'})

#pm25 = data[data['item']=='PM2.5'].drop(['item','data'])
pm25 = data[data['item']=='PM2.5']
pm25f9 = pm25.iloc[:,4:13].replace('NR', 0)
row = np.ones(240)
temp = np.c_[row, pm25f9]
theta = np.ones(10)
theta = theta.transpose()
temp = temp.astype(float)

for i in range(10000):
    hypothesis = np.dot(temp, theta)
    y = pm25['10']
    y = y.astype(float)
    loss = hypothesis - y
    m = 240
    cost = np.sum(loss ** 2) / m
    x = temp
    xTrans = x.transpose()
    gradient = np.dot(xTrans, loss) / m
    #print(math.sqrt(cost))
    alpha = 0.00005
    theta = theta - alpha * gradient

w = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)), x.transpose()), y)

test = pd.read_csv('C:/Users/Vicky/Desktop/ML/hw1/test.csv',header=None)
test = test[test.iloc[:,1]=='PM2.5']
name = test[0]
test = test.iloc[0:,2:]
row = np.ones(240)
test = np.c_[row, test]
test = test.astype(float)
result = np.dot(test, w)
np_data = np.c_[name, result]
print(len(np_data))
filename = "C:/Users/Vicky/Desktop/ML/hw1/handcraft.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter = ',')
s.writerow(["id","value"])
for i in range(len(np_data)):
    s.writerow(np_data[i]) 
text.close()


