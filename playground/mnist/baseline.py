import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, BatchNormalization, Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

X_train = np.load('../data/X_train.npy')
X_valid = np.load('../data/X_valid.npy')
y_train = np.load('../data/y_train.npy')
y_valid = np.load('../data/y_valid.npy')
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

mean_px = X_train.mean()
std_px = X_train.std()

def normalize(x):
    return (x - mean_px) / std_px

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

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

params = {}
params['x'] = X_train.reshape(36000,28,28,1)
params['y'] = y_train
params['validation_data'] = (X_valid.reshape(6000,28,28,1), y_valid)

model.optimizer.lr = 0.001
params['epochs'] = 1
model.fit(**params)

model.optimizer.lr = 0.01
params['epochs'] = 3
model.fit(**params)

model.optimizer.lr = 0.005
params['epochs'] = 39
model.fit(**params)

model.save_weights('../model/model_1.h5')

X_score = np.load('../data/X_score.npy').reshape(28000,28,28,1)

predict = model.predict_classes(X_score)
predict = pd.DataFrame(predict, columns=['Label'])
predict = predict.reset_index()
predict.columns = ['ImageId','Label']
predict['ImageId'] = predict['ImageId'] + 1
predict.to_csv('../data/submit_v1.csv', index=False)