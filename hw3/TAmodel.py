import sys
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard  
import numpy as np
#from keras.utils import to_categorical
from keras.utils import np_utils

#filepath="1.weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"  
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')  
#tensorboard = TensorBoard(log_dir='logs', histogram_freq=0)  
#callbacks_list = [checkpoint, tensorboard]

with open('./data/train.csv', 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y = []
    X = []
    for y in range(0,len(data),48*48+1):
        Y.append(data[y])
        X.append(data[y+1:y+48*48+1].reshape(48,48,1))

with open('./data/test.csv', 'r') as f:
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
batch_size = 128
epochs = 200
print(Y_valid)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',input_shape=(48,48 , 1), activation='relu'))
model.add(ZeroPadding2D(padding=(2, 2), data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D(padding=(1, 1), data_format='channels_last'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))
opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#train_history=model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), epochs=30, batch_size=300, verbose=2)
model.summary()


#datagen = ImageDataGenerator(
#    rotation_range=30,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    zoom_range=[0.8, 1.2],
#    shear_range=0.2,
#    horizontal_flip=True)
#datagen.fit(X_train)
#callbacks = []
#callbacks.append(ModelCheckpoint('ckpt/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))


#model.fit_generator(
#        datagen.flow(X_train, Y_train, batch_size=batch_size), 
#        steps_per_epoch=5*len(X_train)//batch_size,
#        epochs=epochs,
#        validation_data=(X_valid, Y_valid),
#        callbacks=callbacks,
#        verbose = 1
#)


