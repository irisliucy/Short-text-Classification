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
import csv
import sys

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
DROP_OUT = 0.3
Nb_EPOCH = 25
BATCH_SIZE = 10
Classes = 2

parameters = {
"classes" : [11],
#"batches" : [10, 20, 50, 100],
#"epochs": [1, 10, 25, 50, 100], 
#"dropout_rate" : [0.0, 0.1, 0.2, 0.3, 0.4],
#"embedding_dimension" : [25, 50, 100, 200]
}

def add_csv_result(train_accuracy, valid_accuracy, test_accuracy, time):
    global header, items
    global MAX_SEQUENCE_LENGTH, MAX_NB_WORDS , EMBEDDING_DIM, VALIDATION_SPLIT, DROP_OUT, Nb_EPOCH, BATCH_SIZE, Classes 
    header = [['Classes', 'Dropout', 'Iterations', 'Batch Size','Embedding Dimension',
              'Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Time']]
    
    items = [Classes, DROP_OUT, Nb_EPOCH, BATCH_SIZE, EMBEDDING_DIM, 
             train_accuracy, valid_accuracy, test_accuracy, time]
    header.append(items)
    
def write_csv_result(fname): 
    f = open(fname, 'wb')
    writer = csv.writer(f)
    writer.writerows(header)
    header = [] # reset header after each loop 
    f.close()


def reset_parameter():
    global MAX_SEQUENCE_LENGTH, MAX_NB_WORDS , EMBEDDING_DIM, VALIDATION_SPLIT, DROP_OUT, Nb_EPOCH, BATCH_SIZE, Classes 

    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 200
    VALIDATION_SPLIT = 0.1
    DROP_OUT = 0.2
    Nb_EPOCH = 30
    BATCH_SIZE = 10 
    Classes = 2

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


def train_Test_Split(data, labels):
    train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=VALIDATION_SPLIT)      
    return (train_X, train_Y, test_X, test_Y)


# split the data into a training set and a validation set
def train_Val_Split(train_X, train_Y):
    trainX, valX, trainY, valY = train_test_split(train_X, train_Y, test_size=VALIDATION_SPLIT)      
    return (trainX, trainY, valX, valY)

# prepare embedding matrix

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

def train_and_evaluate_model(model, train_X, train_Y, test_X, test_Y):
    start = time.time()
    history = model.fit(train_X, train_Y, validation_split=VALIDATION_SPLIT, nb_epoch=Nb_EPOCH, batch_size=BATCH_SIZE)
    trainTime = time.time() - start
    print ("Training Time : ", trainTime)
    
    last_epoch_training_accuracy = history.history['acc'][Nb_EPOCH-1]
    last_epoch_training_loss = history.history['loss'][Nb_EPOCH-1]
    print ("Training Loss: ", last_epoch_training_loss)
    print ("Training Accuracy: ", last_epoch_training_accuracy)

    last_epoch_validation_accuracy = history.history['val_acc'][Nb_EPOCH-1]
    last_epoch_validation_loss = history.history['val_loss'][Nb_EPOCH-1]
    print ("validation Loss: ", last_epoch_validation_loss)
    print ("Validation Accuracy: ", last_epoch_validation_accuracy)

    eval_loss, eval_accuracy = model.evaluate(test_X, test_Y, verbose=0)
    print ("Testing Loss: ", eval_loss)
    print ("Testing Accuracy: ", eval_accuracy)

    model_history = history.history

    return (model_history, last_epoch_training_loss, last_epoch_training_accuracy, eval_loss, eval_accuracy, trainTime) 

def main():
    global Classes, DROP_OUT, EMBEDDING_DIM, Nb_EPOCH, FILENAME, TEXT_DATA_DIR
    for key in parameters:
	reset_parameter()
	for indx, val in enumerate(parameters[key]):
           
	    if key == 'classes':
		Classes = val
            elif key == 'dropout_rate': 
		DROP_OUT = val
	    elif key == 'embedding_dimension':
		EMBEDDING_DIM = val
	    elif key == 'epochs':
                Nb_EPOCH = val
            elif key == 'batches':
		BATCH_SIZE = val
	    
            GLOVE_DIR = './glove.twitter.27B/'
            FILENAME = 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'
            TEXT_DATA_DIR = './20_newsgroups_' + str(Classes)  

            print ("******************Loop Test******************")
            print ("*****Parameters: ", key, ", Value: ", val, "*****")
	    embedding_index(GLOVE_DIR, FILENAME)
            data, labels, labels_index = load_data(TEXT_DATA_DIR)      #load datasets
	       #create embedding index with GloVe embeddings_index(GLOVE_DIR, FILENAME)   #create embedding index with GloVe
            embedding_matrix = embeddingMatrix()                       #make embedding matrix as inp
	    train_X, train_Y, test_X, test_Y = train_Test_Split(data, labels) #split test set
	    model = create_model()  
	    global_time = time.time()
	    model_history, last_epoch_training_loss, last_epoch_training_accuracy, eval_loss, eval_accuracy, trainTime = train_and_evaluate_model(model, train_X, train_Y, test_X, test_Y)
	    total_time = time.time()-global_time
	    print ("Total Training Time : ", total_time)
	    add_csv_result(last_epoch_training_accuracy, 0 , eval_accuracy, total_time)
	write_csv_result(key+"_20news_LSTM_CV.csv")


main()
#write_csv_result("20news_LSTM_CV_2classes.csv")

	   
#    import SQLdb as db
#    db.updateLSTM(classes = Classes, dropouts = DROP_OUT, iterations = Nb_EPOCH, accuracy = eval_accuracy, remark = total_time)
