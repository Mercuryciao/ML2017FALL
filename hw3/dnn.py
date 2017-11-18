import sys
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
output_file = 'ggg.csv'

with open('./data/train.csv', 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y = []
    X = []
    for y in range(0,len(data),48*48+1):
        Y.append(data[y])
        X.append(data[y+1:y+48*48+1])

with open('./data/test.csv', 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y_test = []
    X_test = []
    for y in range(0,len(data),48*48+1):   
        X_test.append(data[y+1:y+48*48+1])

X = np.array(X).astype(float)/255
X_test = np.array(X_test).astype(float)/255
Y = np.array(Y).astype(int)
X_train, X_valid = X[:-3000], X[-3000:]
Y_train, Y_valid = Y[:-3000], Y[-3000:]
Y_train = np_utils.to_categorical(Y_train)
Y_valid = np_utils.to_categorical(Y_valid)

# create model
model = Sequential()
model.add(Dense(1000, input_dim=48*48, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(7, activation='softmax'))
# Compile model
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), epochs=30, batch_size=256, verbose=2)

predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
predictions = predictions.tolist()
# accuracy
with open('returns.csv', 'w') as f:
     writer = csv.writer(f)
     for index in range(len(train_history.history['acc'])):
         writer.writerow([train_history.history['acc'][index],train_history.history['val_acc'][index]])


fieldnames = ['id','label']
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writerow({'id': 'id', 'label': 'label'})
    for index,each_ans in enumerate(predictions):
        writer.writerow({'id': index, 'label': each_ans})
