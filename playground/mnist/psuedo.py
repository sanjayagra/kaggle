
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import keras


# In[2]:

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, BatchNormalization, Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


# ### Use Pseudo Labels

# In[3]:

X_train = np.load('../data/X_train.npy').reshape(36000,28,28,1)
X_valid = np.load('../data/X_valid.npy').reshape(6000,28,28,1)
X_score = np.load('../data/X_score.npy').reshape(28000,28,28,1)
y_train = np.load('../data/y_train.npy')
y_valid = np.load('../data/y_valid.npy')
y_score = pd.read_csv('../data/submit_v2.csv')
y_score = np.array(pd.get_dummies(y_score['Label']))
print(X_train.shape, y_train.shape)
print(X_score.shape, y_score.shape)
print(X_valid.shape, y_valid.shape)


# ### Data Augmentation

# In[4]:

transform = {}
transform['width_shift_range'] = 0.075
transform['height_shift_range'] = 0.075
transform['rotation_range'] = 5
transform['shear_range'] = 0.3
transform['zoom_range'] = 0.075

generator = ImageDataGenerator(**transform)


# In[5]:

train_batch = generator.flow(X_train, y_train, batch_size=20)
score_batch = generator.flow(X_score, y_score, batch_size=10)


# ### Model Architecture

# In[6]:

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


# In[7]:

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:

model.summary()


# ### Train Model

# In[9]:

def mix_iterator(iterators):
    while True:
        nexts = [next(iter_tuple) for iter_tuple in iterators]
        X = np.concatenate([n[0] for n in nexts])
        y = np.concatenate([n[1] for n in nexts])
        yield (X, y)


# In[10]:

params = {}
params['epochs'] = 1
params['steps_per_epoch'] = 1800 # 36000 / 20
params['validation_data'] = (X_valid, y_valid)


# In[11]:

model.optimizer.lr = 0.001
params['epochs'] = 3
model.fit_generator(mix_iterator([train_batch, score_batch]), **params)


# In[12]:

model.optimizer.lr = 0.01
params['epochs'] = 5
model.fit_generator(generator.flow(X_train, y_train, batch_size=30), **params)


# In[13]:

model.optimizer.lr = 0.007
params['epochs'] = 5
model.fit_generator(generator.flow(X_train, y_train, batch_size=30), **params)


# In[14]:

model.optimizer.lr = 0.005
params['epochs'] = 5
model.fit_generator(generator.flow(X_train, y_train, batch_size=30), **params)


# In[15]:

model.save_weights('../model/model_3.h5')


# ### Score Model

# In[16]:

X_score = np.load('../data/X_score.npy').reshape(28000,28,28,1)


# In[17]:

predict = model.predict_classes(X_score)


# In[18]:

predict = pd.DataFrame(predict, columns=['Label'])
predict = predict.reset_index()
predict.columns = ['ImageId','Label']
predict['ImageId'] = predict['ImageId'] + 1


# In[19]:

predict.to_csv('../data/submit_v3.csv', index=False)


# In[ ]:

### Submission scored 0.99442 on LB

