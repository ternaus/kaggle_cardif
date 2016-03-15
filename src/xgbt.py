from __future__ import division
__author__ = 'Vladimir Iglovikov'


import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss
import numpy as np
import math
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import StratifiedKFold

joined = pd.read_csv('../data/joined_le1.csv', low_memory=False)

obj_columns = ['v107', 'v110', 'v112',
#                'v113',
#                'v125',
               'v22', 'v24', 'v3', 'v30',
'v31',
'v47',
'v52',
# 'v56',
'v66',
'v71',
'v74',
'v75',
'v79',
'v91']

print 'filling missing categorcial'
for column in obj_columns:
    le = LabelEncoder()
    joined[column].fillna(-10000, inplace=True)
    joined[column] = le.fit_transform(joined[column])

train = joined[joined['target'] != -1]
test = joined[joined['target'] == -1]

y = train['target']
features = [
    # 'ID',
 # 'target',
 'v1',
 'v10',
 'v100',
 'v101',
 'v102',
 'v103',
 'v104',
 # 'v105',
 'v106',
 # 'v108',
 # 'v109',
 'v11',
 # 'v110',
 'v111',
 'v112',
     'v113_0',
 'v113_1',
 'v113_10',
 'v113_11',
 'v113_12',
 'v113_13',
 'v113_14',
 'v113_15',
 'v113_16',
 'v113_17',
 'v113_18',
 'v113_19',
 'v113_2',
 'v113_20',
 'v113_21',
 'v113_22',
 'v113_23',
 'v113_24',
 'v113_25',
 'v113_3',
 'v113_4',
 'v113_5',
 'v113_6',
 'v113_7',
 'v113_8',
 'v113_9',
 'v114',
 'v115',
 # 'v116',
 # 'v117',
 # 'v118',
 # 'v119',
 'v12',
 'v120',
 'v121',
 'v122',
 # 'v123',
 # 'v124',
    'v125_0',
 'v125_1',
 'v125_10',
 'v125_11',
 'v125_12',
 'v125_13',
 'v125_14',
 'v125_15',
 'v125_16',
 'v125_17',
 'v125_18',
 'v125_19',
 'v125_2',
 'v125_20',
 'v125_21',
 'v125_22',
 'v125_23',
 'v125_24',
 'v125_25',
 'v125_3',
 'v125_4',
 'v125_5',
 'v125_6',
 'v125_7',
 'v125_8',
 'v125_9',
 'v126',
 'v127',
 # 'v128',
 'v129',
 'v13',
 'v130',
 'v131',
 'v14',
 'v15',
 'v16',
 'v17',
 'v18',
 'v19',
 'v2',
 'v20',
 'v21',
 'v22',
 # 'v23',
 'v24',
 # 'v25',
 'v26',
 'v27',
 'v28',
 'v29',
 'v3',
 'v30',
 # 'v31',
 'v32',
 'v33',
 'v34',
 'v35',
 # 'v36',
 # 'v37',
 'v38',
 'v39',
 'v4',
 'v40',
 'v41',
 'v42',
 'v43',
 'v44',
 'v45',
 # 'v46',
 'v48',
 'v49',
 'v5',
 'v50',
 # 'v51',
 'v52',
 # 'v53',
 # 'v54',
 'v55',
    'v56_0',
 'v56_1',
 'v56_10',
 'v56_11',
 'v56_12',
 'v56_13',
 'v56_14',
 'v56_15',
 'v56_16',
 'v56_17',
 'v56_18',
 'v56_19',
 'v56_2',
 'v56_20',
 'v56_21',
 'v56_22',
 'v56_23',
 'v56_24',
 'v56_25',
 'v56_3',
 'v56_4',
 'v56_5',
 'v56_6',
 'v56_7',
 'v56_8',
 'v56_9',
 'v57',
 'v58',
 'v59',
 'v6',
 'v60',
 'v61',
 'v62',
 # 'v63',
 'v64',
 'v65',
 'v66',
 'v67',
 'v68',
 'v69',
 'v7',
 'v70',
 'v71',
 # 'v73',
 'v74',
 # 'v75',
 'v76',
 'v77',
 'v78',
 # 'v79',
 # 'v8',
 'v80',
 # 'v81',
 # 'v82',
 'v83',
 'v84',
 'v85',
 'v86',
 'v87',
 'v88',
 # 'v89',
 'v9',
 'v90',
 'v91',
 # 'v92',
 'v93',
 'v94',
 # 'v95',
 'v96',
 'v97',
 'v98',
 'v99',
'countNA',
'pattern1_NAcount',
'pattern2_NAcount']

X = train[features]
X_test = test[features]

from sklearn.preprocessing import LabelEncoder
obj_columns = ['v107', 'v110', 'v112',
#                'v113',
#                'v125',
               'v22', 'v24', 'v3', 'v30',
'v31',
'v47',
'v52',
# 'v56',
'v66',
'v71',
'v74',
'v75',
'v79',
'v91']

for column in obj_columns:
    le = LabelEncoder()
    joined[column].fillna(-10000, inplace=True)
    joined[column] = le.fit_transform(joined[column])




print 'filling missing values'
for column in X.columns:  
  # a = X[column].mean()
  X[column] = X[column].fillna(-999)
  X_test[column] = X_test[column].fillna(-999)

params = {
  # "objective": "multi:softmax",
  'objective': "binary:logistic",
  # 'objective': 'count:poisson',
  # 'eta': 0.005,
  # 'min_child_weight': 6,
  # 'subsample': 0.7,
  # 'colsabsample_bytree': 0.7,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'eval_metric': "logloss",
  # 'max_depth': 9
}

num_rounds = 5000
random_state = 42
offset = int(0.2 * X.shape[0])

X = X.values
X_test = X_test.values
print
print X.shape



ind = 1
if ind == 1:
  n_iter = 10
  # rs = KFold(len(y), n_folds=5)
  rs = StratifiedKFold(y.values, n_folds=n_iter)

  # rs = ShuffleSplit(len(y), n_iter=n_iter, test_size=0.2, random_state=random_state)

  result = []
  
  for min_child_weight in [5]:
    for eta in [0.01, 0.05, 0.1]:
      for colsample_bytree in [0.9]:
        for max_depth in [9]:
          for subsample in [0.9]:
            for gamma in [1]:
              fName = open('results', 'a')
              params['min_child_weight'] = min_child_weight
              params['eta'] = eta
              params['colsample_bytree'] = colsample_bytree
              params['max_depth'] = max_depth
              params['subsample'] = subsample
              params['gamma'] = gamma
              
              params_new = list(params.items())
              score = []
              
              for train_index, test_index in rs:
                # X_train = X.values[train_index]
                # X_test = X.values[test_index]
                # y_train = y.values[train_index]
                # y_test = y.values[test_index]

                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                xgtest = xgb.DMatrix(X_test)

                xgtrain = xgb.DMatrix(X_train[offset:, :], label=y_train[offset:])
                xgval = xgb.DMatrix(X_train[:offset, :], label=y_train[:offset])

                watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                model = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=50)
                # xgb.get_fscore(fmap='xgb.fmap')

                preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

                # X_train = X_train[::-1, :]
                # labels = y_train[::-1]

                # xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
                # xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

                # watchlist = [(xgtrain, 'train'), (xgval, 'val')]

                # model = xgb.train(params_new, xgtrain, num_rounds, watchlist, feval=rmpse_xg, early_stopping_rounds=50)

                # preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

                # preds = model.predict(xgval, ntree_limit=model.best_iteration)

                # preds = 0.5 * preds1 + 0.5 * preds2
                preds = preds1

                tp = log_loss(y_test, preds)
                              
                score += [tp]              
                               
                print tp

              sc = math.ceil(10000 * np.mean(score)) / 10000
              sc_std = math.ceil(10000 * np.std(score)) / 10000
              result += [(sc, sc_std, min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter, params['objective'])]
              print >> fName, (sc, sc_std, min_child_weight, eta, colsample_bytree, max_depth, subsample, gamma, n_iter, params['objective'])
              fName.close()
  result.sort()
  
  print
  print 'result'
  print result  

elif ind == 2:
  xgtrain = xgb.DMatrix(X[offset:, :], label=y.values[offset:])
  xgval = xgb.DMatrix(X[:offset, :], label=y.values[:offset])
  xgtest = xgb.DMatrix(X_test)

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  params = {
  # 'objective': 'reg:linear',
    # 'objective': 'count:poisson',
    #   'objective': "binary:logistic",
      'eval_metric': "logloss",
  'eta': 0.005,
  'min_child_weight': 3,
  'subsample': 0.9,
  'colsample_bytree': 0.9,
  # 'scal_pos_weight': 1,
  'silent': 1,
  'max_depth': 9,
  # 'eval_metric': "auc",
  'gamma': 1
  }    
  params_new = list(params.items())
  model1 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
  prediction_test_1 = model1.predict(xgtest, ntree_limit=model1.best_iteration)

  X_train = X[::-1, :]
  labels = y.values[::-1]

  xgtrain = xgb.DMatrix(X_train[offset:, :], label=labels[offset:])
  xgval = xgb.DMatrix(X_train[:offset, :], label=labels[:offset])

  watchlist = [(xgtrain, 'train'), (xgval, 'val')]

  model2 = xgb.train(params_new, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)

  prediction_test_2 = model2.predict(xgtest, ntree_limit=model2.best_iteration)

  prediction_test = 0.5 * prediction_test_1 + 0.5 * prediction_test_2
  submission = pd.DataFrame()
  submission['ID'] = test['ID']
  # submission['Sales'] = np.expm1(prediction_test)
  submission['PredictedProb'] = prediction_test

  submission.to_csv("predictions/xgbt.csv", index=False)
