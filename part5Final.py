#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary libraries
get_ipython().system('pip install gensim')
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from matplotlib import pyplot as plt

from nltk.corpus import brown
from nltk.corpus import treebank
from nltk.corpus import conll2000

import seaborn as sns

from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[27]:


import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report
import ast


# In[3]:


def load_file(train_file):
    # Split train file into word and state lists
    word_ls = []
    st_ls = []
    f = open(train_file, encoding="utf8")
    for line in f:
        line_2 = str(line.strip('\n'))
        if line_2 != '':
            inter_ls = line_2.split()
            word_ls.append(inter_ls[0])
            st_ls.append(inter_ls[1])
        elif line_2 == '':
            word_ls.append(line_2)
            st_ls.append('')
    
    compiled_word_ls = []
    compiled_state_ls = []
    big_word_ls = []
    big_state_ls = []
    for i in range(len(word_ls)):
        if word_ls[i] != '':
            compiled_word_ls.append(word_ls[i])
            compiled_state_ls.append(st_ls[i])
        elif word_ls[i] == '':
            big_word_ls.append(compiled_word_ls)
            big_state_ls.append(compiled_state_ls)
            compiled_word_ls = []
            compiled_state_ls = []
    return big_word_ls, big_state_ls


# In[6]:


# Retrieve Files and split them into respective lists
train_x_ls, train_y_ls = load_file('train')
test_x_ls, test_y_ls = load_file('dev.out')


# In[8]:


## Count your unique words and tags
num_words = len(set([word.lower() for sentence in train_x_ls for word in sentence]))
num_tags   = len(set([word.lower() for sentence in train_y_ls for word in sentence]))
print("Total number of tagged sentences: {}".format(len(train_x_ls)))
print("Vocabulary size: {}".format(num_words))
print("Total number of tags: {}".format(num_tags))


# In[36]:


# encode X

word_tokenizer = Tokenizer()                      # instantiate tokeniser
word_tokenizer.fit_on_texts(train_x_ls)                    # fit tokeniser on data
X_train_encoded = word_tokenizer.texts_to_sequences(train_x_ls)  # use the tokeniser to encode input sequence
X_test_encoded = word_tokenizer.texts_to_sequences(test_x_ls)


# In[37]:


# encode Y

tag_tokenizer = Tokenizer()
tag_tokenizer.fit_on_texts(train_y_ls)
Y_train_encoded = tag_tokenizer.texts_to_sequences(train_y_ls)
Y_test_encoded = tag_tokenizer.texts_to_sequences(test_y_ls)


# In[38]:


tokenizer_dic = tag_tokenizer.get_config()


# In[39]:


# Retrieve your state dictionary for mapping
state_dic = tokenizer_dic['index_word']
state_dic = ast.literal_eval(state_dic)
state_dic = {k:v.upper() for k,v in state_dic.items()}


# In[40]:


state_dic


# In[41]:


# Pad each sequence to MAX_SEQ_LENGTH using KERAS' pad_sequences() function. 
# Sentences longer than MAX_SEQ_LENGTH are truncated.
# Sentences shorter than MAX_SEQ_LENGTH are padded with zeroes.

# Truncation and padding can either be 'pre' or 'post'. 
# For padding we are using 'pre' padding type, that is, add zeroes on the left side.
# For truncation, we are using 'post', that is, truncate a sentence from right side.

MAX_SEQ_LENGTH = 100  # sequences greater than 100 in length will be truncated

X_train_padded = pad_sequences(X_train_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
X_test_padded = pad_sequences(X_test_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

Y_train_padded = pad_sequences(Y_train_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
Y_test_padded = pad_sequences(Y_test_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")


# In[42]:


# Word2Vec
# load word2vec using the following function present in the gensim library
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[43]:


# assign word vectors from word2vec model

EMBEDDING_SIZE  = 300  # each word in word2vec model is represented using a 300 dimensional vector
VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1

# create an empty embedding matix
embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))

# create a word to index dictionary mapping
word2id = word_tokenizer.word_index

# copy vectors from word2vec model to the words present in corpus
for word, index in word2id.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass


# In[44]:


# assign padded sequences to X and Y
X, Y = X_train_padded, Y_train_padded


# In[62]:


# use Keras' to_categorical function to one-hot encode Y
Y_oh = to_categorical(Y)
Y_oh.shape


# In[63]:


# use Keras' to_categorical function to one-hot encode Y
Y_test_oh = to_categorical(Y_test_padded, 22)
Y_test_oh.shape


# In[64]:


# split entire data into training and testing sets
TEST_SIZE = 0.2
X_train, X_val, Y_train, Y_val = train_test_split(X, Y_oh, test_size=TEST_SIZE, random_state=4)


# In[66]:


# total number of tags
NUM_CLASSES = Y_oh.shape[2]

# create architecture


bidirect_model = Sequential()
bidirect_model.add(Embedding(input_dim     = VOCABULARY_SIZE,
                             output_dim    = EMBEDDING_SIZE,
                             input_length  = MAX_SEQ_LENGTH,
                             weights       = [embedding_weights],
                             trainable     = True
))
bidirect_model.add(Bidirectional(LSTM(64, return_sequences=True)))
bidirect_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
bidirect_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
# check summary of model
bidirect_model.summary()


# In[67]:


bidirect_training = bidirect_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_val, Y_val))


# In[102]:


y_pred = bidirect_model.predict(X_test_padded)
y_pred.argmax(axis=1)


# In[94]:


Y_test_padded.argmax(axis=1)


# In[99]:


Y_test_oh.argmax(axis=1)


# In[97]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(Y_test_padded.argmax(axis=1), y_pred.argmax(axis=1))
matrix


# In[81]:


print(classification_report(Y_test_padded.argmax(axis=1), y_pred.argmax(axis=1)))


# In[85]:


loss, accuracy = bidirect_model.evaluate(X_test_padded, Y_test_oh, verbose = 1)
print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))


# In[ ]:




