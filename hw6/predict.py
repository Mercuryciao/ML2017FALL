import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import cluster
from sklearn.cluster import KMeans, MiniBatchKMeans
import pandas as pd
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

def build_model():
    encoding_dim = 64
    input_img = Input(shape=(784,))
    encoded = Dense(512, activation='selu')(input_img)
    encoded = Dense(128, activation='selu')(encoded)

    encoded = Dense(encoding_dim, activation='linear')(encoded)
    decoded = Dense(64, activation='selu')(encoded)


    decoded = Dense(256, activation='selu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)


    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(train, train, nb_epoch=200, batch_size=256, shuffle=True)

    encoder = Model(input = input_img, output = encoded)
    return encoder
#encoder = build_model()
#encoder.save('chi_encoder.h5')
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

