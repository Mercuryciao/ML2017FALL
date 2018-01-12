import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.cluster import KMeans, MiniBatchKMeans
from keras.layers import Input, Dense
from keras.models import Model
import math
import h5py
import keras
import sys

label_file_path = sys.argv[1]
test_file_path = sys.argv[2]
ans_file = sys.argv[3]
test = pd.read_csv(test_file_path)
train = np.load(label_file_path)
train = train / 255

encoder = keras.models.load_model('chi_encoder.h5')
feature = encoder.predict(train)
kmeans = KMeans(n_clusters=2,random_state=0).fit(feature)

ans = []

for index, row in test.iterrows():
    ans.append(int(kmeans.labels_[row['image1_index']] == kmeans.labels_[row['image2_index']]))
    if(index%200000==0):
        print(index)
test['Ans'] = pd.DataFrame(ans)
test.to_csv(ans_file,index=False,columns=['ID','Ans'])
