# coding: utf-8
from __future__ import print_function
import os
import numpy as np
import time

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.layers import Input, Dropout
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential
from sklearn.model_selection import train_test_split, KFold
import sys
import csv

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
DROP_OUT = 0.3
Nb_EPOCH = 25
BATCH_SIZE = 10
Classes = 2
K_FOLD = 10
write_Trigger = 0 


BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.twitter.27B/'
FILENAME = 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroups/'

def write_csv_result(fname, train_accuracy, valid_accuracy, test_accuracy, time):
    global header, items
    header = [['Cross Validation','Classes', 'Dropout', 'Iterations', 'Batch Size','Embedding Dimension',
              'Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Time']]
    
    items = [K_FOLD, Classes, DROP_OUT, Nb_EPOCH, BATCH_SIZE, EMBEDDING_DIM, 
             train_accuracy, valid_accuracy, test_accuracy, time]
    header.append(items)
    
    
    if write_Trigger == 1:
    	f = open(fname, 'wb')
    	writer = csv.writer(f)
    	writer.writerows(header)
    	f.close()

def embeddings_index(GLOVE_DIR, FILENAME):
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
    
    return embeddings_index

print('Processing text dataset')

def load_data(TEXT_DATA_DIR):
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

# split the data into a training set and a validation set
def train_Val_Split(data, labels):
    trainX, valX, trainY, valY = train_test_split(data, labels, test_size=VALIDATION_SPLIT)      
    return (trainX, trainY, valX, valY)

# prepare embedding matrix
print('Preparing embedding matrix.')
def embeddingMatrix():
    global nb_words, embedding_matrix
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
    print('Number of class: ||%d||' % (Classes))
    model = Sequential()
    model.add(Embedding(                          # Layer 0, Start
        input_dim=nb_words + 1,                   # Size to dictionary, has to be input + 1
        output_dim= EMBEDDING_DIM,                # Dimensions to generate
        weights=[embedding_matrix],               # Initialize word weights
        input_length=MAX_SEQUENCE_LENGTH,
	trainable=False))       		  # Define length to input sequences in the first layer
    model.add(LSTM(128, dropout_W=DROP_OUT, dropout_U=DROP_OUT))  
    model.add(Dense(Classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(model, trainX, trainY, testX, testY):
    start = time.time()
    history = model.fit(trainX, trainY, nb_epoch=Nb_EPOCH, batch_size=BATCH_SIZE)    
    trainTime = time.time() - start
    print ("Training Time : ", trainTime)
    
    last_epoch_training_accuracy = history.history['acc'][Nb_EPOCH-1]
    last_epoch_training_loss = history.history['loss'][Nb_EPOCH-1]
    print ("Training Loss: ", last_epoch_training_loss)
    print ("Training Accuracy: ", last_epoch_training_accuracy)

    eval_loss, eval_accuracy = model.evaluate(testX, testY, verbose=0)
    print ("Testing Loss: ", eval_loss)
    print ("Testing Accuracy: ", eval_accuracy)

    

    return (last_epoch_training_loss, last_epoch_training_accuracy, eval_loss, eval_accuracy, trainTime) 

def main(embeddings_index):
    #n_splits = 10
    embeddings_index = embeddings_index(GLOVE_DIR, FILENAME) 
    data, labels, labels_index = load_data(TEXT_DATA_DIR)
    embedding_matrix = embeddingMatrix()

    skf = KFold(n_splits=K_FOLD, shuffle=True, random_state=1337)
    i=1
    final_training_acc = 0
    final_testing_acc = 0
    
    global_time = time.time()
    for (train, test) in skf.split(data):
	    print ("*****Running fold", i, "/", K_FOLD, " *****") 
	    #print (data[train])
	    #print (labels[train])
	    model = None 
            model = create_model()
            last_epoch_training_loss, last_epoch_training_accuracy, eval_loss, eval_accuracy, trainTime = train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test])  
	    final_training_acc = final_training_acc + last_epoch_training_accuracy
            final_testing_acc = final_testing_acc + eval_accuracy
	    i=i+1    
    total_time = time.time()-global_time
    print ("*********Finished compiling ", i-1 , "fold*********")

    avg_training_accuracy = final_training_acc/K_FOLD
    avg_testing_accuracy = final_testing_acc/K_FOLD
    print ("Average Training Accuracy : ", avg_training_accuracy)
    print ("Average Testing Accuracy : ", avg_testing_accuracy)
    print ("Total Running Time : ", total_time)
    
    write_Trigger = 1 
    write_csv_result("20news_LSTM_CV_2classes.csv", avg_training_accuracy, 0 , avg_testing_accuracy, total_time)

#    import SQLdb as db
#    db.updateLSTM(classes = Classes, dropouts = DROP_OUT, iterations = Nb_EPOCH, accuracy = avg_testing_accuracy, remark = total_time)
