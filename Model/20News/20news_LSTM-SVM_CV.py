

from __future__ import print_function
import os
import sys
import numpy as np
import time
from IPython import get_ipython

np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Activation
from keras.layers import Convolution1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model, load_model
from keras.layers import Input, Dropout
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn import *
from sklearn.model_selection import train_test_split, KFold
import theano
import csv
import h5py

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
DROP_OUT = 0.3
Nb_EPOCH = 30
BATCH_SIZE = 50
Classes = 5


GLOVE_DIR = './glove.twitter.27B/'
FILENAME = 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'
TEXT_DATA_DIR = './20_newsgroups_' + str(Classes) 
weights_path = "20news_LSTM_model_" + str(Classes) +".h5" 


def reset_parameter():
    global MAX_SEQUENCE_LENGTH, MAX_NB_WORDS , EMBEDDING_DIM, VALIDATION_SPLIT, DROP_OUT, Nb_EPOCH, BATCH_SIZE, Classes 

    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.1
    DROP_OUT = 0.3
    Nb_EPOCH = 30
    BATCH_SIZE = 50 
    Classes = 2
    
def write_csv_result(fname, train_accuracy, valid_accuracy, test_accuracy, time): 
    global header, items
    global MAX_SEQUENCE_LENGTH, MAX_NB_WORDS , EMBEDDING_DIM, VALIDATION_SPLIT, DROP_OUT, Nb_EPOCH, BATCH_SIZE, Classes 
    header = [['Classes', 'Dropout', 'Iterations', 'Batch Size','Embedding Dimension',
              'Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Time']]
    
    items = [Classes, DROP_OUT, Nb_EPOCH, BATCH_SIZE, EMBEDDING_DIM, 
             train_accuracy, valid_accuracy, test_accuracy, time]
    header.append(items)

    f = open(fname, 'wb')
    writer = csv.writer(f)
    writer.writerows(header)
    header = [] # reset header after each loop 
    f.close()

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


# def train_Test_Split(data, labels):
#     train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size=VALIDATION_SPLIT)      
#     return (train_X, train_Y, test_X, test_Y)



def create_model():
    print('Number of class: ||%d||' % (Classes))
    model = Sequential()
    model.add(Embedding(                          # Layer 0, Start
        input_dim=nb_words + 1,                   # Size to dictionary, has to be input + 1
        output_dim= EMBEDDING_DIM,                # Dimensions to generate
        weights=[embedding_matrix],               # Initialize word weights
        input_length=MAX_SEQUENCE_LENGTH,
        name = "embedding_layer",
        trainable=False))
    model.add(LSTM(128, dropout_W=DROP_OUT, dropout_U=DROP_OUT, name = "lstm_layer"))  
    model.add(Dense(Classes, activation = 'sigmoid', name = "dense_one"))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(model, train_X, train_Y, test_X, test_Y, data):
    global best_weight_path
    start = time.time()

    best_weight_path="weights-improvement.hdf5"
    checkpoint = ModelCheckpoint(best_weight_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
    print("Saved best weights to disk") 
    callbacks_list = [checkpoint]   

    history = model.fit(train_X, train_Y, validation_split=VALIDATION_SPLIT, nb_epoch=Nb_EPOCH, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    
    model.load_weights(best_weight_path)

    first = Sequential()
    first.add(Embedding(                          # Layer 0, Start
        input_dim=nb_words + 1,                   # Size to dictionary, has to be input + 1
        output_dim= EMBEDDING_DIM,                # Dimensions to generate
        weights=[embedding_matrix],               # Initialize word weights
        input_length=MAX_SEQUENCE_LENGTH,
        name = "embedding_layer",
        trainable=False))

    first.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
   
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.save(weights_path)

    training_loss, training_accuracy = model.evaluate(train_X, train_Y)
    print ("Training Loss: ", training_loss)
    print ("Training Accuracy: ", training_accuracy)

    eval_loss, eval_accuracy = model.evaluate(test_X, test_Y)
    print ("Testing Loss: ", eval_loss)
    print ("Testing Accuracy: ", eval_accuracy)

    model_history = history.history

    model.pop()
   
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
	
    print("Saved model to disk")
    
    intermediate_output_train = model.predict(train_X)
    print ("Intermediate training output shape : ", intermediate_output_train.shape)
    np.savetxt('20news_' + str(MAX_SEQUENCE_LENGTH) + ' D_' + str(Classes)+ '_train_output.txt', intermediate_output_train, fmt = '%s')


    intermediate_output_test = model.predict(test_X)
    print ("Intermediate testing output shape : ", intermediate_output_test.shape)
    np.savetxt('20news_' + str(MAX_SEQUENCE_LENGTH) + ' D_' + str(Classes)+ '_test_output.txt', intermediate_output_test, fmt = '%s')

    trainTime = time.time() - start
    print ("Training Time : ", trainTime)
    return (model, training_accuracy, eval_accuracy)

 

def transform_labels(train_Y, test_Y, labels):
    trainY = [] 
    testY = []
    Y = []
    for label in train_Y:
        for idx, bit in enumerate(label):
            if bit ==1:
                trainY.append(idx)
    for label in test_Y:
        for idx, bit in enumerate(label):
            if bit ==1:
                testY.append(idx)
    trainY = np.array(trainY)
    testY = np.array(testY)
    
    print (trainY.shape , testY.shape)
    np.savetxt('20news_' + str(MAX_SEQUENCE_LENGTH) + ' D_' + str(Classes)+ '_train_output_labels.txt', trainY, fmt = '%s')
    np.savetxt('20news_' + str(MAX_SEQUENCE_LENGTH) + ' D_' + str(Classes)+ '_test_output_labels.txt', testY, fmt = '%s')
    return trainY, testY


def evaluate_with_SVM(data, labels, train_X, train_Y,test_X, test_Y):
    print ("Starting SVM")
    clf = svm.SVC(kernel='linear')
    clf.fit(train_X, train_Y)
    predict_Y = clf.predict(test_X) 
    s=metrics.accuracy_score(test_Y, predict_Y) 
    print ("SVM Testing Acc: ", s)
    return s


def evaluate_with_KNN(data, labels, train_X, train_Y,test_X, test_Y):
    print ("Starting KNN")
    kNN = neighbors.KNeighborsClassifier()
    kNN.fit(train_X, train_Y)
    predict_Y = kNN.predict(test_X)  
    s=metrics.accuracy_score(test_Y, predict_Y) 
    print ("KNN Testing Acc: ", s)
    return s

def main():
    global Classes, DROP_OUT, EMBEDDING_DIM, Nb_EPOCH, FILENAME, TEXT_DATA_DIR, weights_path
    
    embedding_index(GLOVE_DIR, FILENAME)
    data, labels, labels_index = load_data(TEXT_DATA_DIR)
    embedding_matrix = embeddingMatrix()

    
    K_FOLD = 5
    i=1
    skf = KFold(n_splits=K_FOLD, shuffle=True, random_state=1337)
    cvscores_train = []
    cvscores_test = []
    svm_score = []
    knn_score = []
    global_time = time.time()
    for train, test in skf.split(data):
        print ("*****Running fold", i, "/", K_FOLD, " *****")
        model = None 
        model = create_model()
        trainX, trainY, testX, testY = data[train], labels[train], data[test], labels[test]
        model, training_accuracy, eval_accuracy = train_and_evaluate_model(model, trainX, trainY, testX, testY, data)  
        print("Training Acc: %.5f , Testing Acc: %.5f" % (training_accuracy, eval_accuracy))
        cvscores_train.append(training_accuracy)
        cvscores_test.append(eval_accuracy)  
        trainY, testY = transform_labels(trainY, testY, labels)
        svm_test_accuracy = evaluate_with_SVM(data, labels, model.predict(trainX), trainY, model.predict(testX), testY)
        knn_test_accuracy = evaluate_with_KNN(data, labels, model.predict(trainX), trainY, model.predict(testX), testY)
   
        svm_score.append(svm_test_accuracy)
        knn_score.append(knn_test_accuracy)
        i=i+1   
        
    total_time = time.time()-global_time
    print ("+++++++++++++++++++++++++++Results after ", i-1, " Fold(s)+++++++++++++++++++++++++++")
    print("Average Training Accuracy: %f , Average Testing Accuracy: %f" % (np.mean(cvscores_train), np.mean(cvscores_test)))
    print ("SVM Testing Accuracy: %f " % (np.mean(svm_score)))
    print ("KNN Testing Accuracy: %f " % (np.mean(knn_score)))
    print ("Total Running Time : ", total_time)
    write_csv_result("_20news_LSTM_CV.csv", np.mean(cvscores_train), 0 , np.mean(cvscores_test), total_time)

main() 


