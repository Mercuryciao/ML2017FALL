import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# If you wish to get the same shuffle result
# np.random.seed(2401)

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    X_train = np.array(X_train.values)
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = np.array(X_test.values)

    return (X_train, Y_train, X_test)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def train(X_all, Y_all, save_dir, X_test, output_dir):

    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.01
    seed = 4
    X_train, x_validation, Y_train, y_validation = train_test_split(X_all, Y_all, test_size=valid_set_percentage,random_state = seed)
    model = XGBClassifier(max_depth = 5)
    model.fit(X_train, Y_train)
    # make predictions for test data
    y_pred = model.predict(x_validation)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_validation, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    output_path = os.path.join(output_dir)
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(predictions):
            f.write('%d,%d\n' %(i+1, v))

    return

def main(opts):
    # Load feature and label
    X_all, Y_all, X_test = load_data(opts.train_data_path, opts.train_label_path, opts.test_data_path)
    # Normalization
    X_all, X_test = normalize(X_all, X_test)

    # To train or to infer
    if opts.train:
        train(X_all, Y_all, opts.save_dir, X_test, opts.output_dir)
    elif opts.infer:
        infer(X_test, opts.save_dir, opts.output_dir)
    else:
        print("Error: Argument --train or --infer not found")
    return

def infer(X_test, save_dir, output_dir):
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    print(predictions)


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str,
                        default='feature/X_train', dest='train_data_path',
                        help='Path to training data')
    parser.add_argument('--train_label_path', type=str,
                        default='feature/Y_train', dest='train_label_path',
                        help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str,
                        default='feature/X_test', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--save_dir', type=str,
                        default='logistic_params/', dest='save_dir',
                        help='Path to save the model parameters')
    parser.add_argument('--output_dir', type=str,
                        default='xgbo_output/', dest='output_dir',
                        help='Path to save the model parameters')
    opts = parser.parse_args()
    main(opts)

