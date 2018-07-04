
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ### Data

# In[2]:

X_train = np.load('../data/X_train.npy')
X_valid = np.load('../data/X_valid.npy')
X_score = np.load('../data/X_score.npy')
y_train = np.load('../data/y_train.npy')
y_valid = np.load('../data/y_valid.npy')


# ### Model

# In[3]:

model = KNeighborsClassifier(n_jobs=15, n_neighbors=5)
model.fit(X_train, y_train)


# ### Validation

# In[4]:

predict = model.predict(X_valid)
valid_pred = np.argmax(predict,axis=1)
valid_actual = np.argmax(y_valid, axis=1)
print(accuracy_score(valid_actual, valid_pred))


# ### Score

# In[5]:

def scores(data):
    data = [x[:,1] for x in data]
    data = pd.DataFrame(data).T
    return np.array(data)


# In[6]:

np.save('../data/scores/knn_train.npy',scores(model.predict_proba(X_train)))
np.save('../data/scores/knn_valid.npy',scores(model.predict_proba(X_valid)))
np.save('../data/scores/knn_score.npy',scores(model.predict_proba(X_score)))


# In[ ]:



