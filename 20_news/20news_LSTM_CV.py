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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
import sys

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.twitter.27B/'

TEXT_DATA_DIR = BASE_DIR + '/20_newsgroups/'

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 25  #25, 50, 100, 200
VALIDATION_SPLIT = 0.2
DENSE_FEATURE = 1024
DROP_OUT = 0.3

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')
print('Embedding Dimesions: %s' % (str(EMBEDDING_DIM)))

embeddings_index = {}
fname = os.path.join(GLOVE_DIR, 'glove.twitter.27B.' + str(EMBEDDING_DIM) + 'd.txt')
f = open(fname)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
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

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(nb_words + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)

print('Training model.')
def create_model(optimizer='sgd', dropout_rate= 0.2):
	start = time.time()
	model = Sequential()

	model.add(Embedding(                          # Layer 0, Start
    		input_dim=nb_words + 1,                   # Size to dictionary, has to be input + 1
    		output_dim=EMBEDDING_DIM,                 # Dimensions to generate
    		weights=[embedding_matrix],               # Initialize word weights
	    	input_length=MAX_SEQUENCE_LENGTH))        # Define length to input sequences in the first layer

	model.add(LSTM(128, dropout_W=dropout_rate, dropout_U=dropout_rate))  # try using a GRU instead, for fun
	model.add(Dense(5))
	model.add(Activation('sigmoid'))

	model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

	return model

model = KerasClassifier(build_fn=create_model,verbose=0)
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(batch_size=batch_size, nb_epoch=epochs, optimizer=optimizers, 
		dropout_rate=dropout_rate)

start = time.time()

lstm = GridSearchCV(estimator=model, param_grid=param_grid, cv=10) # Cross Validation for the best hyperparameters

grid_result = lstm.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

print ("Fitting Time : ", time.time() - start)


print("Done compiling.")
