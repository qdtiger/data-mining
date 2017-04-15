# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:25:34 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from tools import*

Product = pd.read_csv('../data/JData_Product.csv')
dataset1 = pd.read_csv('../data/train/dataset2016-04-02_2016-04-16.csv')
dataset2 = pd.read_csv('../data/train/dataset2016-04-07_2016-04-16.csv')

dataset1 = dataset1.drop('time',1)
dataset1.drop_duplicates(inplace=True)

dataset2 = dataset2.drop('time',1)
dataset2.drop_duplicates(inplace=True)

dataset1_negative = data_negative(dataset1,10)



X_train, X_test, y_train, y_test = train_test_split(dataset1.drop(['user_id','sku_id','label'],axis=1).values, dataset1['label'].values, test_size=0.2, random_state=0)
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
    'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
    'scale_pos_weight': 1, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 300
param['nthread'] = 4
#param['eval_metric'] = "auc"
plst = param.items()
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst, dtrain, num_round, evallist)

dataset1_x = dataset1.drop(['user_id','sku_id','label'],axis=1)
dataset1_y = dataset1.label
dataset2_x = dataset2.drop(['user_id','sku_id','label'],axis=1)
dataset2_y = dataset2.label
dataset2_preds = dataset2[['user_id','sku_id']]

del dataset1,dataset2
data1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
del dataset1_x,dataset1_y
data2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
del dataset2_x,dataset2_y

params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.05,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
     
watchlist = [(data1,'train')]
model = xgb.train(params,data1,num_boost_round=300,evals=watchlist)

dataset2_preds['label'] = model.predict(data2)
dataset2_preds.label = MinMaxScaler().fit_transform(dataset2_preds.label)
dataset2_preds.drop_duplicates(inplace=True)

a = dataset2_preds.groupby(['user_id'])['label'].agg(lambda x: max(x)).reset_index()
b = pd.merge(a,dataset2_preds,on=['user_id','label'])
b = pd.merge(b,Product,on='sku_id')
c = b[b.label>0.678]
c = c[['user_id','sku_id']]
#---------------4.11-4.15预测集------------------
#predict test set
dataset3_preds['label'] = model.predict(dataset3)
dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)
dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
print dataset3_preds.describe()
    
#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


test = test[['user_id','sku_id']]
test.drop_duplicates(inplace = True)
test1 = test[['user_id']]
test1['buy'] = 1
test1 = test1.groupby(['user_id']).agg('sum').reset_index()
test1 = test1[test1['buy']==1]
test1 = test1[['user_id']]
test = pd.merge(test,test1,on='user_id')
test = test[['user_id','sku_id']]
test = pd.merge(test,Product[['sku_id']],how='inner',on='sku_id')

evaluation(test,c)