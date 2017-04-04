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
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold
from theano import function
import sys
import h5py

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 10000 #20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1
DROP_OUT = 0.3
Nb_EPOCH = 1
BATCH_SIZE = 10
Classes = 2

GLOVE_DIR = './glove.twitter.27B/'
FILENAME = 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt'
TEXT_DATA_DIR = './20_newsgroups/'
weights_path = "weights.best.h5" # Save weight of model as checkpoints



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
    
    return embeddings_index

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

# prepare embedding matrix 
def embeddingMatrix():
    print('Preparing embedding matrix.')
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
        output_dim= EMBEDDING_DIM,                # Dimensions to generate: 100
        weights=[embedding_matrix],               # Initialize word weights
        input_length=MAX_SEQUENCE_LENGTH,         # Define length to input sequences in the first layer
       trainable=False))                       
    model.add(LSTM(32, dropout_W=DROP_OUT, dropout_U=DROP_OUT))  # Output dimension: 128
    model.add(Dense(Classes))
    model.add(Activation('sigmoid'))

    #model.load_weights('./' + weights_path)
    #print ("Loaded model weights from disk")
       
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def train_and_evaluate_model(model, train_X, train_Y, test_X, test_Y):
    start = time.time()
    
    history = model.fit(train_X, train_Y, validation_split = VALIDATION_SPLIT, nb_epoch = Nb_EPOCH, batch_size=BATCH_SIZE)
    np.savetxt("model_history.txt", history, fmt = "%s" )
    
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

    model.save_weights(weights_path)
    print("Saved model weights to disk")

    return (last_epoch_training_loss, last_epoch_training_accuracy, last_epoch_validation_loss, last_epoch_validation_accuracy, trainTime) 


def get_bottleNeck_layer(model, train_X, train_Y):
    assert os.path.weights(exists_path),
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    print ('train_X: ', train_X.shape)
    print ('train_Y: ', train_Y.shape)

    bottleNeck_output = model.predict(train_X)
    print ('output_size: ',bottleNeck_output.shape)
    
    return bottleNeck_output
    
def get_output_intermediate_layer(model, train_X):
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[2].output])
    layer_output = get_layer_output([train_X, 0])[0]
    print (layer_output.shape)
    layer_output_size = len(layer_output)
    layer_output = np.reshape(layer_output_size, -1)
    layer_output = np.reshape(layer_output, (1, layer_output.shape[0], layer_output.shape[1]*layer_output.shape[2]))
    print ("Training output: ", layer_output)
    print ("Training output shape 0 : ", layer_output.shape[0])
    print ("Training output shape 1 : ", layer_output.shape[1])
    print ("Training output shape 2 : ", layer_output.shape[2])

    return (layer_output)
       
def evaluate_with_SVM(layer_output, train_X, train_Y, test_X, test_Y ):
    print ("Starting SVM")
    clf = svm.SVC()
    print ("Training SVM")
    train_X_SVM = clf.fit(layer_output, train_Y)
    print ("SVM Train Labels: ", train_Y)
    clf.predict(test_X) 
       

def main():
    embeddings_index = embedding_index(GLOVE_DIR, FILENAME)          # create embedding index with GloVe
    data, labels, labels_index = load_data(TEXT_DATA_DIR)             # load datasets
    embedding_matrix = embeddingMatrix()                              # make embedding matrix as input
    train_X, train_Y, test_X, test_Y = train_Test_Split(data, labels) # split trian and test sets
    model = create_model()  
    global_time = time.time()
    last_epoch_training_loss, last_epoch_training_accuracy, last_epoch_validation_loss, last_epoch_validation_accuracy, trainTime = train_and_evaluate_model(model, train_X, train_Y, test_X, test_Y)
    
    #layer_output = get_output_intermediate_layer(model, train_X)
    #layer_output = get_bottleNeck_layer(model, train_X, train_Y)
    #evaluate_with_SVM(layer_output, train_X, train_Y, test_X, test_Y )
    total_time = time.time()-global_time
    print ("Total Training Time : ", total_time)

    

main()
