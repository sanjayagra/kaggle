import pandas as pd
import numpy as np

data = pd.read_csv('../data/train.csv')
score = pd.read_csv('../data/test.csv')
print(data.shape)
print(score.shape)

train = data.iloc[:36000,:]
valid = data.iloc[36000:,:]

valid_dist = valid['label'].value_counts() / valid['label'].shape[0]
train_dist = train['label'].value_counts() / train['label'].shape[0]
valid_dist = valid_dist.reset_index()
train_dist = train_dist.reset_index()

distribution = train_dist.merge(valid_dist, on='index')
distribution.round(3)

X_train = np.array(train.drop('label',axis=1))
X_valid = np.array(valid.drop('label',axis=1))
X_score = np.array(score)
print(X_train.shape, X_valid.shape, X_score.shape)

np.save('../data/X_train.npy', X_train)
np.save('../data/X_valid.npy', X_valid)
np.save('../data/X_score.npy', X_score)

y_train = np.array(pd.get_dummies(train['label']))
y_valid = np.array(pd.get_dummies(valid['label']))
print(y_train.shape, y_valid.shape)

np.save('../data/y_train.npy', y_train)
np.save('../data/y_valid.npy', y_valid)