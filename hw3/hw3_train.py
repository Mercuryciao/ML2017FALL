import sys
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
import csv
import sys

input_file = sys.argv[1]
output_file = 'gg.csv'
with open(input_file, 'r') as f:
    data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
    data = np.array(data)
    Y = []
    X = []
    for y in range(0,len(data),48*48+1):
        Y.append(data[y])
        X.append(data[y+1:y+48*48+1].reshape(48,48,1))




# with open('./data/test.csv', 'r') as f:
#     data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
#     data = np.array(data)
#     Y_test = []
#     X_test = []
#     for y in range(0,len(data),48*48+1):   
#         X_test.append(data[y+1:y+48*48+1].reshape(48,48,1)) 
# X = np.array(X).astype(float)/255
# X_test = np.array(X_test).astype(float)/255

X = np.array(X).astype(float)/255
Y = np.array(Y).astype(int)
X_train, X_valid = X[:-1500], X[-1500:]
Y_train, Y_valid = Y[:-1500], Y[-1500:]
Y_train = np_utils.to_categorical(Y_train)
Y_valid = np_utils.to_categorical(Y_valid)

batch_size = 256
epochs = 100

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(48,48 , 1), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.1))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_json = model.to_json()
# with open("model-3.json", "w") as json_file:
#     json_file.write(model_json)
model.summary()

datagen = ImageDataGenerator(
   rotation_range=10,
   width_shift_range=0.2,
   height_shift_range=0.2,
   horizontal_flip=True)

datagen.fit(X_train)
callbacks = []
callbacks.append(ModelCheckpoint('ckpt/model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))


train_history = model.fit_generator(
       datagen.flow(X_train, Y_train, batch_size=batch_size), 
       steps_per_epoch=len(X_train)//batch_size,
       epochs=epochs,
       validation_data=(X_valid, Y_valid),
       callbacks=callbacks,
       verbose = 1
       )
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)
predictions = predictions.tolist()
