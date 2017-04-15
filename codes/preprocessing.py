# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:49:28 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

Action_4 = pd.read_csv('../data/splitdata/Action4.csv')
Action_4 = Action_4.iloc[:,1:]
Product = pd.read_csv('../data/JData_Product.csv')

train = Action_4[Action_4['day']==10]

test = Action_4[Action_4['day']<=15]
test = test[test['day']>=11]
test = test[test.type == 4]
test1 = test[['user_id']]
test1['buy'] = 1
test1 = test1.groupby(['user_id']).agg('sum').reset_index()
test1 = test1[test1['buy']==1]
test1 = test1[['user_id']]
test = pd.merge(test,test1,on='user_id')
test = pd.merge(test,Product[['sku_id']],how='inner',on='sku_id')

us_buy = pd.merge(train,test,on=['user_id','sku_id'])
us_buy[['user_id','sku_id']].drop_duplicates()
us_buy[us_buy.user_id==286471]
test[test.user_id==286471]
a=Action_4[Action_4.user_id == 286471]
a[a.type==2]
a[a.sku_id==81708]

b = train[train.user_id==286471]
b[b.type==3]
b[b.sku_id==151327]
