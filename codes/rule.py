# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 18:42:59 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd

Action_4 = pd.read_csv('../data/splitdata/Action4.csv')
Product = pd.read_csv('../data/JData_Product.csv')

train = Action_4[Action_4['day']==1]
train = train[train['day']>=2]

#------------------test---------------------
test = Action_4[Action_4['day']<=15]
test = test[test['day']>=11]
test = test[test.type == 4]
test = test[['user_id','sku_id']]
test1 = test[['user_id']]
test1['buy'] = 1
test1 = test1.groupby(['user_id']).agg('sum').reset_index()
test1 = test1[test1['buy']==1]
test1 = test1[['user_id']]
test = pd.merge(test,test1,on='user_id')
test = test[['user_id','sku_id']]
test = pd.merge(test,Product[['sku_id']],how='inner',on='sku_id')

t2 = train[['user_id','type']]

user_buy = set()
for i in range(t2.shape[0]):
    if t2.iloc[i,1] == 4:
        user_buy.add(t2.iloc[i,0])
user_all = set(t2.user_id.unique())
user_true = user_all^user_buy
user_true = pd.DataFrame({'user_id':list(user_true)})
train = pd.merge(train,user_true,on='user_id')

train.time = train.time.apply(lambda x:x.replace(' ',''))
train['mtime'] = train.time.apply(lambda x:x[8:].replace(':',''))
u1 = train[['user_id','sku_id','type','mtime']]
u1 = u1[u1.type==2]
u1 = u1.groupby(['user_id','sku_id'])['mtime'].agg(lambda x:':'.join(x)).reset_index()
u1['max_time_buy'] = u1.mtime.apply(lambda s:max([int(d) for d in s.split(':')]))

u2 = train[['user_id','sku_id','type','mtime']]
u2 = u2[u2.type==3]
u2 = u2.groupby(['user_id','sku_id'])['mtime'].agg(lambda x:':'.join(x)).reset_index()
u2['max_time_delete'] = u2.mtime.apply(lambda s:max([int(d) for d in s.split(':')]))

u3 = pd.merge(u1[['user_id','sku_id','max_time_buy']],u2[['user_id','sku_id','max_time_delete']],how='left',on=['user_id','sku_id'])
u3.max_time_delete = u3.max_time_delete.fillna(0)
u3['judge'] = (u3.max_time_buy>u3.max_time_delete)
u3 = u3[u3.judge==True]
#---------------挑出删除购物车的人-----------
a=train[train.type==3]
a=a[['user_id','sku_id']]
a['delete']=1
a = a.groupby(['user_id','sku_id']).agg('sum').reset_index()
b=a[a.delete>1]
b=b[['user_id','sku_id']]
b.drop_duplicates(inplace='True')
train_delete = pd.merge(train[train.type==3],b,on=['user_id','sku_id'])

t1 = train[train['type']==1]
t1 = t1[['user_id','sku_id']]
t1['browse'] = 1
t1 = t1.groupby(['user_id','sku_id']).agg('sum').reset_index()
cc = pd.merge(u3,t1,how='left',on=['user_id','sku_id'])
cc.browse = cc.browse.fillna(0)
#==============================================================================
# train = pd.merge(train,t1,on=['user_id','sku_id'])
# train = train[train.type != 4]
# train = train[train.type == 2]
# train = train[train.hour > 8]
#==============================================================================
ui = cc.iloc[:,[0,1,-1]]
ui.drop_duplicates(inplace=True)

a = ui.groupby(['user_id'])['browse'].agg(lambda x: max(x)).reset_index()
b = pd.merge(a,ui,on=['user_id','browse'])
a=[]
for i in b['user_id'].unique():
    if b[b['user_id']==i].shape[0]==1:
        a.append(list(b[b['user_id']==i].iloc[0,:].values))
    else:
        a.append(list(b[b['user_id']==i].iloc[0,:].values))

a = np.array(a)
submission = pd.DataFrame({
        "user_id": a[:,0],
        "sku_id": a[:,2]
    })
submission = pd.merge(submission,Product,on='sku_id')
submission = submission[['user_id','sku_id']]
submission.user_id = submission.user_id.astype('int64')
submission.sku_id = submission.sku_id.astype('int64')
submission.to_csv('../results/result_03_31_13.csv',index=False)

#==============================================================================
# test = Action_4[Action_4['day']<=15]
# test = test[test['day']>=11]
# 
# test = test[test.type == 4]
# test = test[['user_id','sku_id']]
# test1 = test[['user_id']]
# test1['buy'] = 1
# test1 = test1.groupby(['user_id']).agg('sum').reset_index()
# test1 = test1[test1['buy']==1]
# test1 = test1[['user_id']]
# test = pd.merge(test,test1,on='user_id')
# test = test[['user_id','sku_id']]
# test = pd.merge(test,Product[['sku_id']],how='inner',on='sku_id')
# 
# 
# train = Action_4[Action_4['day']==10]
# t2 = train[['user_id','type']]
# t2 = t2[t2['type'] != 4]
# user_buy = set()
# for i in range(t2.shape[0]):
#     if t2.iloc[i,1] == 4:
#         user_buy.add(t2.iloc[i,0])
# user_all = set(t2.user_id.unique())
# user_true = user_all^user_buy
# user_true = pd.DataFrame({'user_id':list(user_true)})
# train = pd.merge(train,user_true,on='user_id')
# 
# train['mtime']=train.time.apply(lambda x:x[11:].replace(':',''))
# u1 = train[['user_id','sku_id','type','mtime']]
# u1 = u1[u1.type==2]
# u1 = u1.groupby(['user_id','sku_id'])['mtime'].agg(lambda x:':'.join(x)).reset_index()
# u1['max_time_buy'] = u1.mtime.apply(lambda s:max([int(d) for d in s.split(':')]))
# 
# u2 = train[['user_id','sku_id','type','mtime']]
# u2 = u2[u2.type==3]
# u2 = u2.groupby(['user_id','sku_id'])['mtime'].agg(lambda x:':'.join(x)).reset_index()
# u2['max_time_delete'] = u2.mtime.apply(lambda s:max([int(d) for d in s.split(':')]))
# 
# u3 = pd.merge(u1[['user_id','sku_id','max_time_buy']],u2[['user_id','sku_id','max_time_delete']],how='left',on=['user_id','sku_id'])
# u3.max_time_delete = u3.max_time_delete.fillna(0)
# u3['judge'] = (u3.max_time_buy>u3.max_time_delete)
# u3 = u3[u3.judge==True]
# 
# t1 = train[train['type']==1]
# t1 = t1[['user_id','sku_id']]
# t1['browse'] = 1
# #t1 = train[['user_id','sku_id']]
# #t1['browse'] = 1
# t1 = t1.groupby(['user_id','sku_id']).agg('sum').reset_index()
# t2 = train[['user_id','sku_id']]
# t2['click']  = 1
# t2 = t2.groupby(['user_id','sku_id']).agg('sum').reset_index()
# train = pd.merge(train,t1,on=['user_id','sku_id'])
# #train.sort_values('browse',ascending=False)
# train = train[train.type != 4]
# train = train[train.type == 2]
# train = train[train.hour > 4]
# ui = train.iloc[:,[0,1,9]]
# ui.drop_duplicates(inplace=True)
# 
# a = ui.groupby(['user_id'])['browse'].agg(lambda x: max(x)).reset_index()
# b = pd.merge(a,ui,on=['user_id','browse'])
# a=[]
# for i in b['user_id'].unique():
#     if b[b['user_id']==i].shape[0]==1:
#         a.append(list(b[b['user_id']==i].iloc[0,:].values))
#     else:
#         a.append(list(b[b['user_id']==i].iloc[0,:].values))
# 
# a = np.array(a)
# submission = pd.DataFrame({
#         "user_id": a[:,0],
#         "sku_id": a[:,2]
#     })
# submission = pd.merge(submission,Product,on='sku_id')
# submission = submission[['user_id','sku_id']]
# 
# evaluation(test,hh)
#==============================================================================
