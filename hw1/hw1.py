import csv
import pandas as pd
import numpy as np
import math
import sys
input_filename = sys.argv[1]
output_filename = sys.argv[2]
test = pd.read_csv(input_filename,header=None)
w=[2.2618693,-0.01079356,0.0422497,0.14858045,-0.29331687,0.02119222, 0.64541901,-0.51883644,-0.26141243,1.2171603]

test = test[test.iloc[:,1]=='PM2.5']
name=test[0]
test = test.iloc[0:,2:]
row=np.ones(240)
test=np.c_[row,test]
test = test.astype(float)
result=np.dot(test, w)
aa=np.c_[name,result]
filename = output_filename
text = open(filename, "w+")
s = csv.writer(text,delimiter=',')
s.writerow(["id","value"])
for i in range(len(aa)):
    s.writerow(aa[i]) 
text.close()

