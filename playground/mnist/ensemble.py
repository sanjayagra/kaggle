import pandas as pd
import numpy as np
import xgboost as xgb

y_train = np.load('../data/y_train.npy').argmax(axis=1)
y_valid = np.load('../data/y_valid.npy').argmax(axis=1)
print(y_train.shape, y_valid.shape)

sum_train = np.atleast_2d(np.load('../data/X_train.npy').mean(axis=1)).T
sum_valid = np.atleast_2d(np.load('../data/X_valid.npy').mean(axis=1)).T
sum_score = np.atleast_2d(np.load('../data/X_score.npy').mean(axis=1)).T
print(sum_train.shape, sum_valid.shape, sum_score.shape)

max_train = np.atleast_2d(np.count_nonzero(np.load('../data/X_train.npy'),axis=1)).T
max_valid = np.atleast_2d(np.count_nonzero(np.load('../data/X_valid.npy'),axis=1)).T
max_score = np.atleast_2d(np.count_nonzero(np.load('../data/X_score.npy'),axis=1)).T
print(max_train.shape, max_valid.shape, max_score.shape)

knn_train = np.load('../data/scores/knn_train.npy')
knn_valid = np.load('../data/scores/knn_valid.npy')
knn_score = np.load('../data/scores/knn_score.npy')
print(knn_train.shape, knn_valid.shape, knn_score.shape)

cnn1_train = np.load('../data/scores/cnn1_train.npy')
cnn1_valid = np.load('../data/scores/cnn1_valid.npy')
cnn1_score = np.load('../data/scores/cnn1_score.npy')
print(cnn1_train.shape, cnn1_valid.shape, cnn1_score.shape)

cnn2_train = np.load('../data/scores/cnn2_train.npy')
cnn2_valid = np.load('../data/scores/cnn2_valid.npy')
cnn2_score = np.load('../data/scores/cnn2_score.npy')
print(cnn2_train.shape, cnn2_valid.shape, cnn2_score.shape)

cnn3_train = np.load('../data/scores/cnn3_train.npy')
cnn3_valid = np.load('../data/scores/cnn3_valid.npy')
cnn3_score = np.load('../data/scores/cnn3_score.npy')
print(cnn3_train.shape, cnn3_valid.shape, cnn3_score.shape)

X_train = np.concatenate([sum_train,max_train,knn_train,cnn1_train,cnn2_train,cnn3_train], axis=1)
X_valid = np.concatenate([sum_valid,max_valid,knn_valid,cnn1_valid,cnn2_valid,cnn3_valid], axis=1)
X_score = np.concatenate([sum_score,max_score,knn_score,cnn1_score,cnn2_score,cnn3_score], axis=1)
print(X_train.shape, X_valid.shape, X_score.shape)

train_matrix = xgb.DMatrix(data=X_valid[:3000,:], label=y_valid[:3000])
valid_matrix = xgb.DMatrix(data=X_valid[3000:,:], label=y_valid[3000:])
score_matrix = xgb.DMatrix(data=X_score)

booster = {}
booster['booster'] = 'gbtree'
booster['nthread'] = 15
booster['max_depth'] = 6
booster['min_child_weight'] = 1
booster['subsample'] = 0.8
booster['colsample_bytree'] = 1.0
booster['colsample_bylevel'] = 0.9
booster['lambda'] = 1.0
booster['alpha'] = 0.0
booster['objective'] = 'multi:softprob'
booster['num_class'] = 10
booster['eval_metric'] = ['merror']
booster['seed'] = 108

params = {}
params['params'] = booster
params['dtrain'] = train_matrix
params['num_boost_round'] = 2000
params['evals'] = [(train_matrix,'train_matrix'),(valid_matrix,'valid_matrix')]
params['early_stopping_rounds'] = 10
params['verbose_eval'] = 1
params['callbacks'] = [xgb.callback.reset_learning_rate([0.001] * 2000)]

model = xgb.train(**params)

predict = model.predict(score_matrix).argmax(axis=1)
predict = pd.DataFrame(predict, columns=['Label'])
predict = predict.reset_index()
predict.columns = ['ImageId','Label']
predict['ImageId'] = predict['ImageId'] + 1

predict.to_csv('../data/submit_v4.csv', index=False)