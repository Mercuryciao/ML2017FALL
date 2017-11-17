#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import csv
output_file = 'seafood.csv'
with open('train.csv', 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y = []
    X = []
    for y in range(0,len(data),48*48+1):
        Y.append(data[y])
        X.append(data[y+1:y+48*48+1].reshape(48,48,1))
    #X = np.delete(data
with open('test.csv', 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y_test = []
    X_test = []
    for y in range(0,len(data),48*48+1):   
        X_test.append(data[y+1:y+48*48+1].reshape(48,48,1)) 
X = np.array(X).astype(float)/255
X_test = np.array(X_test).astype(float)/255

Y = np.array(Y).astype(int)
X_train, X_valid = X[:-3000], X[-3000:]
Y_train, Y_valid = Y[:-3000], Y[-3000:]
Y_train = np_utils.to_categorical(Y_train)
Y_valid = np_utils.to_categorical(Y_valid)
print(X.shape)
model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same', input_shape=(48,48 , 1), activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(filters=200, kernel_size=(3, 3), padding='same', activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(filters=500, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding = 'same'))
model.add(Dropout(0.45))


model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), epochs=30, batch_size=300, verbose=2)
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
predictions = predictions.tolist()
fieldnames = ['id','label']
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writerow({'id': 'id', 'label': 'label'})
    for index,each_ans in enumerate(predictions):
        writer.writerow({'id': index, 'label': each_ans})

#my_data = genfromtxt('train.csv', delimiter=',',names=True)