import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import csv 
from keras import regularizers
from keras.models import Model,load_model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from utils import DataManager

test_path = sys.argv[1]
output_file = sys.argv[2]
load_path = 'model/GREAT/'
path = ''
dm = DataManager()

dm.add_data('test_data', test_path, False)
dm.load_tokenizer(os.path.join(load_path,'token.pk'))
dm.to_sequence(40)


'''
inputs = Input(shape=(args.max_length,))
embed = gensim.models.Word2Vec.load('GREATmodel')
#print( embed[embed.wv.vocab])
embedding_matrix = zeros((5225 + 1, 100))
print(len(dm.tokenizer.word_index))
for word, i in dm.tokenizer.word_index.items():
    if word in embed.wv.vocab:    
            embedding_vector = embed.wv[word]
            embedding_matrix[i] = embedding_vector
    # Embedding layer
embedding_inputs = Embedding(args.vocab_size+1, 
args.embedding_dim, 
weights=[embedding_matrix],
trainable=True)(inputs)
# RNN 
return_sequence = False
dropout_rate = 0.4
RNN_cell = LSTM(48, 
    return_sequences=return_sequence, 
    dropout=dropout_rate)       
conv_outputs = Conv1D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu')(embedding_inputs)
conv_outputs = Dropout(0.4)(conv_outputs)
outputs = RNN_cell(conv_outputs)
outputs = Dense(1, activation='sigmoid')(outputs)
model =  Model(inputs=inputs,outputs=outputs)
# optimizer
'''






model = load_model(os.path.join(load_path,'model.h5'))
testX = dm.get_data('test_data')[0][1:]
predictions = model.predict(testX)
#raise Exception ('Implement your testing function')
print("predict over!")
fieldnames = ['id','label']
all_ans = np.round_(predictions)
with open(output_file, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
    writer.writerow({'id': 'id', 'label': 'label'})
#ddd
    for index, each_ans in enumerate(all_ans):
        writer.writerow({'id': index, 'label':int(each_ans[0]) })
print('done!')
