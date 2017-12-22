import keras
import numpy as np
import pandas as pd
from keras.models import load_model
import csv
import sys

model_name = 'higher.h5'
answer_file = sys.argv[2]
test_file = sys.argv[1]

train_m = 3.58171208604

test = pd.read_csv(test_file)
model = load_model(model_name)
ans = model.predict([test['UserID'],test['MovieID']])
test['Rating'] = pd.DataFrame(ans+train_m)
test.to_csv(answer_file,index=False,columns=['TestDataID','Rating'])
