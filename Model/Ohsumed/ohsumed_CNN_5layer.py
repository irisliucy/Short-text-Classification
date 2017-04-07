from __future__ import print_function
import os
import numpy as np
import time
np.random.seed(1337)

import theano
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.layers import Input, Dropout
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
import sys
import csv

MAX_SEQUENCE_LENGTH = 200 # pad zero for length longer than Max_sequence length
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
CONVOLUTION_FEATURE = 256
DENSE_FEATURE = 1024
DROP_OUT = 0.3
Nb_EPOCH = 30
BATCH_SIZE = 30
Classes = 5
K_FOLD = 5

GLOVE_DIR = './glove.6B/'
FILENAME = 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'
TEXT_DATA_DIR = './ohsumed_' + str(Classes) 
weights_path = "Ohsumed_LSTM_model_" + str(MAX_SEQUENCE_LENGTH)+ "D_" + str(Classes) +".h5" 

def embedding_index(GLOVE_DIR, FILENAME):
    global embeddings_index 
    embeddings_index = {}
    fname = os.path.join(GLOVE_DIR, FILENAME)
    f = open(fname)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

def load_data(TEXT_DATA_DIR):
    global labels_index
    print('Processing text dataset')
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    print('Found %s texts.' % len(texts))
    
    global word_index, tokenizer

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    return (data, labels, labels_index)

def embeddingMatrix():
    global nb_words, embedding_matrix
    print('Preparing embedding matrix.')
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def create_model():
    global labels_index
    print('Number of class: ||%d||' % (Classes))
    model = Sequential()

    model.add(Embedding(                          # Layer 0, Start
        input_dim=nb_words + 1,                   # Size to dictionary, has to be input + 1
        output_dim=EMBEDDING_DIM,                 # Dimensions to generate
        weights=[embedding_matrix],               # Initialize word weights
        input_length=MAX_SEQUENCE_LENGTH))        # Define length to input sequences in the first layer

    model.add(Convolution1D(                      # Layer 1,   Features: 256, Kernel Size: 3
        nb_filter=CONVOLUTION_FEATURE,            # Number of kernels or number of filters to generate
        filter_length=3,                          # Size of kernels
        border_mode='valid',                      # Border = 'valid', cause kernel to reduce dimensions
        activation='relu'))                       # Activation function to use

    model.add(MaxPooling1D(                       # Layer 1a,  Max Pooling: 3
        pool_length=3))                           # Size of kernels
    """
    model.add(Convolution1D(                      # Layer 2,   Features: 256, Kernel Size: 3
        nb_filter=CONVOLUTION_FEATURE,            # Number of kernels or number of filters to generate
        filter_length=3,                          # Size of kernels
        border_mode='valid',                      # Border = 'valid', cause kernel to reduce dimensions
        activation='relu'))                       # Activation function to use

    model.add(MaxPooling1D(                       # Layer 2a,  Max Pooling: 3
        pool_length=3))                           # Size of kernels
    
    model.add(Convolution1D(                      # Layer 3,   Features: 256, Kernel Size: 3
        nb_filter=CONVOLUTION_FEATURE,            # Number of kernels or number of filters to generate
        filter_length=3,                          # Size of kernels
        border_mode='valid',                      # Border = 'valid', cause kernel to reduce dimensions
        activation='relu'))                       # Activation function to use
    
    model.add(Convolution1D(                      # Layer 4,   Features: 256, Kernel Size: 3
        nb_filter=CONVOLUTION_FEATURE,            # Number of kernels or number of filters to generate
        filter_length=3,                          # Size of kernels
        border_mode='valid',                      # Border = 'valid', cause kernel to reduce dimensions
        activation='relu'))                       # Activation function to use
    
    model.add(Convolution1D(                      # Layer 5,   Features: 256, Kernel Size: 3
        nb_filter=CONVOLUTION_FEATURE,            # Number of kernels or number of filters to generate
        filter_length=3,                          # Size of kernels
        border_mode='valid',                      # Border = 'valid', cause kernel to reduce dimensions
        activation='relu'))                       # Activation function to use
    
    model.add(Convolution1D(                      # Layer 6,   Features: 256, Kernel Size: 3
        nb_filter=CONVOLUTION_FEATURE,            # Number of kernels or number of filters to generate
        filter_length=5,                          # Size of kernels
        border_mode='valid',                      # Border = 'valid', cause kernel to reduce dimensions
        activation='relu'))                       # Activation function to use
    
    model.add(MaxPooling1D(                       # Layer 6a,  Max Pooling: 3
        pool_length=3))                           # Size of kernels
    """    
    model.add(Flatten())                          # Layer 7
    
    model.add(Dense(                              # Layer 7a,  Output Size: 1024
        output_dim=DENSE_FEATURE,                 # Output dimension
        activation='relu'))                       # Activation function to use
    
    model.add(Dropout(DROP_OUT))
    
    model.add(Dense(                              # Layer 8,   Output Size: 1024
        output_dim=DENSE_FEATURE,                 # Output dimension
        activation='relu'))                       # Activation function to use
    
    model.add(Dropout(DROP_OUT))

    model.add(Dense(                              # Layer 9,  Output Size: Size Unique Labels, Final
        output_dim=len(labels_index),             # Output dimension
        activation='softmax')) 

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



def train_and_evaluate_model(model, train_X, train_Y, test_X, test_Y):
    global best_weight_path
    start = time.time()

    best_weight_path="CNN-weights-improvement.hdf5"
    checkpoint = ModelCheckpoint(best_weight_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint] 
    print("Saved model weights to disk")   

    history = model.fit(train_X, train_Y, validation_split=VALIDATION_SPLIT, nb_epoch=Nb_EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    
    model.load_weights(best_weight_path)
   
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.save(weights_path)
    print("Saved CNN model to disk")    
    
    trainTime = time.time() - start
    print ("Training Time : ", trainTime)

    training_loss, training_accuracy = model.evaluate(train_X, train_Y)
    print ("Training Loss: ", training_loss)
    print ("Training Accuracy: ", training_accuracy)

    eval_loss, eval_accuracy = model.evaluate(test_X, test_Y)
    print ("Testing Loss: ", eval_loss)
    print ("Testing Accuracy: ", eval_accuracy)

    model_history = history.history

    return (model_history, training_loss, training_accuracy, eval_loss, eval_accuracy, trainTime) 

def main():
    global embeddings_index

    embedding_index(GLOVE_DIR, FILENAME) 
    data, labels, labels_index = load_data(TEXT_DATA_DIR)
    embedding_matrix = embeddingMatrix()

    skf = KFold(n_splits=K_FOLD, shuffle=True, random_state=1337)
    cvscores_train = []
    cvscores_test = []
    global_time = time.time()
    i= 1 
    for train, test in skf.split(data):
        print ("*****Running fold", i, "/", K_FOLD, " *****")
        model = None 
        model = create_model()
        trainX, trainY, testX, testY = data[train], labels[train], data[test], labels[test]
        history, training_loss, training_accuracy, eval_loss, eval_accuracy, trainTime = train_and_evaluate_model(model, trainX, trainY, testX, testY) 
        print("Training Acc: %.5f , Testing Acc: %.5f" % (training_accuracy, eval_accuracy))
        cvscores_train.append(training_accuracy)
        cvscores_test.append(eval_accuracy)
        i=i+1    
    total_time = time.time()-global_time
    print ("*********Finished compiling ", i-1 , "fold*********")
    print("Average Training Accuracy: %s , Average Testing Accuracy: %s" % (np.mean(cvscores_train), np.mean(cvscores_test)))
    print("Average Training Accuracy: %.5f%% , Average Testing Accuracy: %.5f%%" % ((np.mean(cvscores_train)*100), (np.mean(cvscores_test)*100)))
    print ("Total Running Time : ", total_time)
    
main()

