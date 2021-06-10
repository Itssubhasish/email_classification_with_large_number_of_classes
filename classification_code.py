# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:02:50 2021

@author: Subhasish Roy
"""
import pandas as pd; import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from stop_words import get_stop_words
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings(action = 'ignore')
#import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from tensorflow.keras.callbacks import EarlyStopping


## Reading data in local file
metadata = pd.read_csv('email_classification_data.csv')
data = metadata[['Description', 'Category', 'Sub Category']]

#print(str(len(np.array(data.Category.unique())))+ ' Categories are - ')
#print(np.array(data.Category.unique()))
#print(str(len(np.array(data['Sub Category'].unique())))+' Sub-categories are - ')
#print(np.array(data['Sub Category'].unique()))

data = data.dropna(subset  = ['Description'])
data.Category.value_counts()

def clearData(x):
    return " ".join(x.strip().split())
data.Description = data.Description.apply(lambda x: re.sub(r'<[^<]+?>|\\[A-Za-z]', ' ', str(x)))
data.Description = data.Description.apply(lambda x: clearData(str(x)))


#%%


stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english'))   #About 150 stopwords
stop_words.extend(nltk_words)
    
data['modified_description'] = [w for w in data.Description if not w in stop_words]

#%%
data = data.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
#STOPWORDS = set(stopwords.words('english'))
#stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english')) #About 150 stopwords
#stop_words.extend(nltk_words)
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in nltk_words) # remove stopwors from text
    return text
data['Description'] = data['Description'].apply(clean_text)
data['Description'] = data['Description'].str.replace('\d+', '')



#%%

MAX_NB_WORDS = 25000
MAX_SEQUENCE_LENGTH = 310
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['Description'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#%%

X = tokenizer.texts_to_sequences(data['Description'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
#%%
Y = pd.get_dummies(data['Category']).values
print('Shape of label tensor:', Y.shape)

#%%
#%%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
#%%
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
#model.add(LSTM(200, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.4))
model.add(Dense(14, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64
#%%
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.20,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
#%%

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

#%%


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#%%

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
#%%
def getFreq(x):
    return len(x)
#%%
data['length_words'] = data.Description.apply(lambda x: getFreq(x))
#%%
# Plot the training graph
def plot_training(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    
    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')
    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='best')

    axes[1].plot(epochs, loss, 'r-', label='Training Loss')
    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='best')
    
    plt.show()
    
plot_training(history.history)


#%%

new_complaint = ['I am not going to say sorry!']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['Payments', 'Inquiry', 'Disputes', 'Documents', 'Invoicing',
       'Business Related', 'End of Lease', 'Delinquency', 'Dealer',
       'Portal', 'In-House', 'Managed Print', 'Change Request',
       'Attorney']
print(pred, labels[np.argmax(pred)])


#%%
### Running for sub-categories(There are 88 sub catagories).
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 450
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['Description'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
Yt = pd.get_dummies(data['Sub Category']).values
print('Shape of label tensor:', Yt.shape)
X = tokenizer.texts_to_sequences(data['Description'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


Xt_train, Xt_test, Yt_train, Yt_test = train_test_split(X,Yt, test_size = 0.20)
print(Xt_train.shape,Yt_train.shape)
print(Xt_test.shape,Yt_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
#model.add(LSTM(200, dropout=0.3, recurrent_dropout=0.2))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.4))
model.add(Dense(100))
model.add(Dense(88, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

epochs = 20
batch_size = 32

history = model.fit(Xt_train, Yt_train, epochs=epochs, batch_size=batch_size,validation_split=0.20,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])