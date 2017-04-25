# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:25:34 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from sklearn import cross_validation, metrics   
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

from tools import*

Product = pd.read_csv('../data/JData_Product.csv')
dataset1 = pd.read_csv('../data/train/dataset2016-03-02_2016-03-16.csv')
dataset2 = pd.read_csv('../data/train/dataset2016-04-02_2016-04-16.csv')

dataset1 = dataset1.drop(['time','time1'],1)
dataset1.drop_duplicates(inplace=True)
dataset1.index = [i for i in range(len(dataset1))]
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
    'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
    'scale_pos_weight': 1, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 300
param['nthread'] = 4
#param['eval_metric'] = "auc"
plst = param.items()
plst += [('eval_metric', 'logloss')]
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst, dtrain, num_round, evallist)

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


del dataset1

dataset2_x = dataset2.drop(['user_id','sku_id','label'],axis=1).values
data2 = xgb.DMatrix(dataset2_x)
dataset2_preds = dataset2[['user_id','sku_id']]

y = bst.predict(data2)

dataset2_preds['label'] = y
pred = dataset2_preds[dataset2_preds['label'] >= 0.075]
pred = pred[['user_id', 'sku_id']]
pred = pred.groupby('user_id').first().reset_index()
pred['user_id'] = pred['user_id'].astype(int)
pred = pd.merge(pred,Product,on='sku_id')
pred = pred[['user_id', 'sku_id']]

label = dataset2[['user_id','sku_id','label']]
label = pd.merge(label,Product[['sku_id']],on='sku_id')
label = label[label.label==1]
label = label[['user_id','sku_id']]
label.drop_duplicates(inplace=True)

evaluation(label,pred)
report(pred,label)

#==================cv=====================
target = 'label'
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='logloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
    
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

predictors = [x for x in dataset1.columns if x not in ['user_id','sku_id','label']]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, dataset1, predictors)
#==============================================================================
# dataset1_x = dataset1.drop(['user_id','sku_id','label'],axis=1)
# dataset1_y = dataset1.label
# dataset2_x = dataset2.drop(['user_id','sku_id','label'],axis=1)
# dataset2_y = dataset2.label
# dataset2_preds = dataset2[['user_id','sku_id']]
# 
# del dataset1,dataset2
# data1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
# del dataset1_x,dataset1_y
# data2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
# del dataset2_x,dataset2_y
# 
# params={'booster':'gbtree',
# 	    'objective': 'rank:pairwise',
# 	    'eval_metric':'auc',
# 	    'gamma':0.1,
# 	    'min_child_weight':1.1,
# 	    'max_depth':5,
# 	    'lambda':10,
# 	    'subsample':0.7,
# 	    'colsample_bytree':0.7,
# 	    'colsample_bylevel':0.7,
# 	    'eta': 0.05,
# 	    'tree_method':'exact',
# 	    'seed':0,
# 	    'nthread':12
# 	    }
#      
# watchlist = [(data1,'train')]
# model = xgb.train(params,data1,num_boost_round=300,evals=watchlist)
# 
# dataset2_preds['label'] = model.predict(data2)
# dataset2_preds.label = MinMaxScaler().fit_transform(dataset2_preds.label)
# dataset2_preds.drop_duplicates(inplace=True)
# 
# a = dataset2_preds.groupby(['user_id'])['label'].agg(lambda x: max(x)).reset_index()
# b = pd.merge(a,dataset2_preds,on=['user_id','label'])
# b = pd.merge(b,Product,on='sku_id')
# c = b[b.label>0.678]
# c = c[['user_id','sku_id']]
# #---------------4.11-4.15预测集------------------
# #predict test set
# dataset3_preds['label'] = model.predict(dataset3)
# dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label)
# dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
# dataset3_preds.to_csv("xgb_preds.csv",index=None,header=None)
# print dataset3_preds.describe()
#     
# #save feature score
# feature_score = model.get_fscore()
# feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
# fs = []
# for (key,value) in feature_score:
#     fs.append("{0},{1}\n".format(key,value))
#     
# with open('xgb_feature_score.csv','w') as f:
#     f.writelines("feature,score\n")
#     f.writelines(fs)
# 
# 
# test = test[['user_id','sku_id']]
# test.drop_duplicates(inplace = True)
# test1 = test[['user_id']]
# test1['buy'] = 1
# test1 = test1.groupby(['user_id']).agg('sum').reset_index()
# test1 = test1[test1['buy']==1]
# test1 = test1[['user_id']]
# test = pd.merge(test,test1,on='user_id')
# test = test[['user_id','sku_id']]
# test = pd.merge(test,Product[['sku_id']],how='inner',on='sku_id')
# 
# evaluation(test,c)
#==============================================================================
