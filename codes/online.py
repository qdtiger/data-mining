# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 15:58:54 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from tools import*

Product = pd.read_csv('../data/JData_Product.csv')
dataset1 = pd.read_csv('../data/train/dataset2016-04-02_2016-04-16.csv')
dataset2 = pd.read_csv('../data/train/dataset2016-04-07_2016-04-16.csv')

dataset1 = dataset1.drop(['time','time1'],1)
dataset1.drop_duplicates(inplace=True)
dataset1.index = [i for i in range(len(dataset1))]

a = dataset1.drop(['user_id','sku_id','label'],axis=1)
a.index = [i for i in range(len(a))]
a = a.fillna(0)
a = a.replace(np.inf,-1)

lr = LogisticRegression(C=1., solver='lbfgs')
lr.fit(a.values,dataset1['label'].values)
y_pred = lr.predict_proba(a.values)
y_pred = y_pred[:,1]
ind = y_pred.argsort()[np.int(0.1*len(a)):-1]

lab = dataset1[['label']]
lab = lab.loc[ind,:]


dataset1_negative = data_negative(dataset1,10)
dataset1_negative.drop_duplicates(inplace = True)
dataset1 = pd.concat([dataset1[dataset1.label==1],dataset1_negative],axis=0)

dataset2 = dataset2.drop(['time','time1','cate_y','brand_y'],1)
dataset2.drop_duplicates(inplace=True)

dataset1_negative = data_negative(dataset1,10)

dataset1.info()

X_train, X_test, y_train, y_test = train_test_split(dataset1.drop(['user_id','sku_id','label'],axis=1).values, dataset1['label'].values, test_size=0.2, random_state=0)
dtrain=xgb.DMatrix(X_train, label=y_train)
dtest=xgb.DMatrix(X_test, label=y_test)
param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
    'min_child_weight': 4, 'gamma': 0, 'subsample': 1, 'colsample_bytree': 0.8,
    'scale_pos_weight': 1, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 3000
param['nthread'] = 4
#param['eval_metric'] = "auc"
plst = param.items()
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst, dtrain, num_round, evallist)



dataset2_x = dataset2.drop(['user_id','sku_id'],axis=1)
dataset2_x = dataset2_x.fillna(0)
dataset2_x = dataset2_x.replace(np.inf,-1)

data1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
data2 = xgb.DMatrix(dataset2_x.values)
dataset2_preds = dataset2[['user_id','sku_id']]

y = bst.predict(data2)

dataset2_preds['label'] = y
pred = dataset2_preds[dataset2_preds['label'] >= 0.8]
pred = pred[['user_id', 'sku_id']]
pred = pred.groupby('user_id').first().reset_index()
pred = pred.astype(int)
pred = pd.merge(pred,Product,on='sku_id')
pred = pred[['user_id', 'sku_id']]

pred.to_csv('../results/result_04_22_20.csv', index=False, index_label=False)


feature_score = bst.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('../results/xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)




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
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }
     
watchlist = [(data1,'train')]
model = xgb.train(params,data1,num_boost_round=500,evals=watchlist)

dataset2_preds['label'] = model.predict(data2)
dataset2_preds.label = MinMaxScaler().fit_transform(dataset2_preds.label)
dataset2_preds.drop_duplicates(inplace=True)

a = dataset2_preds.groupby(['user_id'])['label'].agg(lambda x: max(x)).reset_index()
b = pd.merge(a,dataset2_preds,on=['user_id','label'])
b = pd.merge(b,Product,on='sku_id')
c = b[b.label>0.678]
c = c[['user_id','sku_id']]
c = c.astype('int')

c.to_csv('../results/result_04_09_14.csv',index=False)