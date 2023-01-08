#%%
import pandas as pd


#%%
# 1) Data loading
url = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
df = pd.read_csv(url)

#%%
# 2) Data inspection
print(df.info())
df.head()

# to check duplicated data
df.duplicated().sum()

print(df['review'][100])

#%%
# 3) Data cleaning

# 1) remove numbers (settled)
# 2) remove HTML tags (settled)
# 3) remove punctuations (settled)
# 4) change all to lowercase

import re
# temp = df['review'][0]
# print(re.sub('<.*?>', '', temp))
for index, data in enumerate(df['review']):
    df['review'][index] = re.sub('<.*?>', '', data)
    df['review'][index] = re.sub('[^a-zA-Z]', ' ', df['review'][index]).lower()




#%%
# 4) Feature selection

review = df['review']
sentiment = df['sentiment']

#%%
# 5) Data preprocessing

# step 1) tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

num_words = 5000 # num_words must use the total number of unique words in all the sentences
oov_token = '<oov>' # oov is out of vocabulary

# instantiate Tokenizer()
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

# to train the tokenizer --> same as mms.fit
tokenizer.fit_on_texts(review)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# to transform the text using tokenizer --> mms.transform
review = tokenizer.texts_to_sequences(review)

# step 2) padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_review = pad_sequences(review, maxlen=200, padding='post', truncating='post')

# step 3) One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(sentiment[::,None])

# step 4) train test split
# expand dimension before feeding to train test split
import numpy as np
padded_review = np.expand_dims(padded_review, axis=-1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded_review, sentiment, test_size=0.2, random_state=64)

# %%
# Model development
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras import Sequential

embedding_layer = 64

model = Sequential()
model.add(LSTM(64,input_shape=(x_train.shape[1:]), return_sequences=True)) # input follows the x_train
model.add(Dropout(0.3)) # dropout value follows 
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax')) # output follows the y_train. and activation uses softmax bcs it gives probability (this is a classification problem)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
"""
- categorical crossentropy can be used for multiple classes
- binary crossentropy can only be used for 2 classes but same as categorical crossentropy
- sparse categorical crossentropy is used if one hot encoding isnot used
"""

#%%
# iMPROVING MODEL USING ENBEDDING LAYER

embedding_layer = 64

model = Sequential()
model.add(Embedding(num_words, embedding_layer))
model.add(LSTM(embedding_layer, return_sequences=True)) # input follows the x_train
model.add(Dropout(0.3)) # dropout value follows 
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax')) # output follows the y_train. and activation uses softmax bcs it gives probability (this is a classification problem)

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#%%
# Model training
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=5)

#%%
history.history.keys()

# %%
# Model analysis
import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.show()

y_pred = model.predict(x_test)

#%%
from sklearn.metrics import classification_report, confusion_matrix

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# y_test is y actual and y_pred is predicted data
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# %%
# Model saving
import pickle

model.save('model.h5')

#to save one hot encoder model
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# to save tokenizer
import json
token_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    json.dump(token_json, f)