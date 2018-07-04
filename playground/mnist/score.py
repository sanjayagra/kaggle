
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import keras


# In[2]:

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, BatchNormalization, Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


# In[7]:

X_train = np.load('../data/X_train.npy').reshape(36000,28,28,1)
X_valid = np.load('../data/X_valid.npy').reshape(6000,28,28,1)
X_score = np.load('../data/X_score.npy').reshape(28000,28,28,1)


# ### Model

# In[4]:

model = Sequential()
model.add(BatchNormalization(input_shape=(28,28,1), axis=-1))
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# ### Best Model

# In[5]:

model.load_weights('../model/model_3.h5')


# In[10]:

np.save('../data/scores/cnn1_train.npy', model.predict_proba(X_train))
np.save('../data/scores/cnn1_valid.npy', model.predict_proba(X_valid))
np.save('../data/scores/cnn1_score.npy', model.predict_proba(X_score))


# ### Second Best Model

# In[11]:

model.load_weights('../model/model_2.h5')


# In[12]:

np.save('../data/scores/cnn2_train.npy', model.predict_proba(X_train))
np.save('../data/scores/cnn2_valid.npy', model.predict_proba(X_valid))
np.save('../data/scores/cnn2_score.npy', model.predict_proba(X_score))


# ### Worst Model

# In[13]:

model.load_weights('../model/model_1.h5')


# In[14]:

np.save('../data/scores/cnn3_train.npy', model.predict_proba(X_train))
np.save('../data/scores/cnn3_valid.npy', model.predict_proba(X_valid))
np.save('../data/scores/cnn3_score.npy', model.predict_proba(X_score))


# In[ ]:



