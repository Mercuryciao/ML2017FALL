import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
import csv
import gensim

from numpy import zeros
from keras import regularizers
from keras.models import Model,Sequential
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Conv1D
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from utils import DataManager

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('model')
parser.add_argument('action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=30, type=int)
parser.add_argument('--val_ratio', default=0.06, type=float)
parser.add_argument('--gpu_fraction', default=2, type=float)
parser.add_argument('--vocab_size', default=5230, type=int)
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU','BOW'])
parser.add_argument('-emb_dim', '--embedding_dim', default=100, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=48, type=int)
parser.add_argument('--dropout_rate', default=0.40, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.05,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
args = parser.parse_args()

train_path = 'data/training_label.txt'
test_path = 'data/testing_data.txt'
semi_path = 'data/training_nolabel.txt'
output_file = args.model+'vec.csv'


# build model
def simpleRNN(args,dm):
    inputs = Input(shape=(args.max_length,))
    embed = gensim.models.Word2Vec.load(args.model + 'model')
    #print( embed[embed.wv.vocab])
    embedding_matrix = zeros((args.vocab_size + 1, 100))
    k = 0
    print(len(dm.tokenizer.word_index))
    for word, i in dm.tokenizer.word_index.items():
        if word in embed.wv.vocab:    
            k=k+1
            print(k)
            embedding_vector = embed.wv[word]
            embedding_matrix[i] = embedding_vector
    # Embedding layer
    embedding_inputs = Embedding(args.vocab_size+1, 
    args.embedding_dim, 
    weights=[embedding_matrix],
    trainable=True)(inputs)
    # RNN 
    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
         RNN_cell = GRU(args.hidden_size, 
         return_sequences=return_sequence, 
         dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size, 
        return_sequences=return_sequence, 
        dropout=dropout_rate)       
    conv_outputs = Conv1D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu')(embedding_inputs)
    conv_outputs = Dropout(0.4)(conv_outputs)

    outputs = RNN_cell(conv_outputs)
       # DNN layer
    
#    outputs = Dense(args.hidden_size//2, 
#                   activation='relu',
#                   kernel_regularizer=regularizers.l2(0.01))(outputs)
#    outputs = Dropout(dropout_rate)(outputs)
#    outputs = Dense(args.hidden_size*2,
#                   activation='relu',
#                   kernel_regularizer=regularizers.l2(0.01))(outputs)
#    outputs = Dense(args.hidden_size*2,
#                   activation='relu',
#                   kernel_regularizer=regularizers.l2(0.01))(outputs)

#    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)
                                                                                                                                                                  
    model =  Model(inputs=inputs,outputs=outputs)
    # optimizer
    adam = Adadelta(lr=1.5)#Adam()
    print ('compile model...')
    # compile model
    model.compile( loss=args.loss_function, optimizer=adam, metrics=[ 'accuracy',])                                                                                                                                                                     
    return model
def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
    K.set_session(get_session(args.gpu_fraction))
                          
    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
         load_path = os.path.join(args.save_dir,args.load_model)
    #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('semi_data', semi_path, False)
        #dm.add_data('test_data',test_path, False)
        dm.add_data('train_data', train_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    else:
        dm.add_data('test_data',test_path,False)
        #raise Exception ('Implement your testing parser')
    # prepare tokenizer
    print ('cleaning...')
    dm.clean()
    #if args.action =='train': 
    #    dm.wordvec()
    if args.cell == 'BOW':
        print('TFIDF ing')
        dm.tokenize(args.vocab_size)
        dm.to_bow()
    else:
        print ('get Tokenizer...')
        if args.load_model is not None:
            # read exist tokenizer
            dm.load_tokenizer(os.path.join(load_path,'token.pk'))
        else:
            # create tokenizer on new data
            dm.tokenize(args.vocab_size)
                                                                    
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path,'token.pk')):
            dm.save_tokenizer(os.path.join(save_path,'token.pk'))

    # convert to sequences
        dm.to_sequence(args.max_length)
    # initial model
    print ('initial model...')
    if args.cell == 'BOW':
        n_words = dm.get_data('train_data')[0].shape[1]
        model = Sequential()
        model.add(Dense(256, input_shape=(n_words,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
# compile network
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif not args.action == 'test':
        #dm.wordvec()
        model = simpleRNN(args,dm)
        
    #print (model.summary())
    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model = load_model(path)
            #model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')
    # training
    if args.action == 'train':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        #print(X[100000:100005])
        earlystopping = EarlyStopping(monitor='val_acc', patience = 6, verbose=1, mode='max')
        model_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=model_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        history = model.fit(X, Y, 
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch, 
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )
        hist = history.history
        save_path = os.path.join(save_path,'history.csv')
        thefile = open(save_path,'w')
        wr = csv.writer(thefile)
        for index, item in enumerate(hist['acc']):
            wr.writerow( [hist['acc'][index],hist['loss'][index],hist['val_acc'][index],hist['val_loss'][index]])
    # testing
    elif args.action == 'test' :
        testX = dm.get_data('test_data')[0][1:] 
        predictions = model.predict(testX)
        #raise Exception ('Implement your testing function')
        print("predict over!")
        fieldnames = ['id','label']
        all_answer = np.round_(predictions)
        with open(output_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writerow({'id': 'id', 'label': 'label'})
            for index,each_ans in enumerate(all_answer):
                 writer.writerow({'id': index, 'label':int(each_ans[0]) })


    # semi-supervised training
    elif args.action == 'semi':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        [semi_all_X] = dm.get_data('semi_data')
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
        save_path = os.path.join(save_path,'model.h5')
        checkpoint = ModelCheckpoint(filepath=save_path, 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
        # repeat 10 times
        for i in range(3):
            # label the semi-data
            semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
            semi_X, semi_Y = dm.get_semi_data('semi_data', semi_pred, args.threshold, args.loss_function)
            semi_X = np.concatenate((semi_X, X))
            semi_Y = np.concatenate((semi_Y, Y))
            print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
            # train
            history = model.fit(semi_X, semi_Y, 
                                validation_data=(X_val, Y_val),
                                epochs=2, 
                                batch_size=args.batch_size,
                                callbacks=[checkpoint, earlystopping] )
            if os.path.exists(save_path):
                print ('load model from %s' % save_path)
                model.load_weights(save_path)
            else:
                raise ValueError("Can't find the file %s" %path)
if __name__ == '__main__':
    main()
