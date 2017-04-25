# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:10:44 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd
from datetime import date
from datetime import datetime
from datetime import timedelta

action_1_path = "../data/JData_Action_201602.csv"
action_2_path = "../data/JData_Action_201603.csv"
action_3_path = "../data/JData_Action_201604.csv"
user_path = "../data/JData_User.csv"
product_path = "../data/JData_Product.csv"
comment_path = "../data/JData_Comment.csv"

comment_date = ["2016-02-01", "2016-02-08", "2016-02-15", "2016-02-22", "2016-02-29", "2016-03-07", "2016-03-14",
                "2016-03-21", "2016-03-28",
                "2016-04-04", "2016-04-11", "2016-04-15"]


def convert_age(age_str):
    if age_str == u'-1':
        return 0
    elif age_str == u'15岁以下':
        return 1
    elif age_str == u'16-25岁':
        return 2
    elif age_str == u'26-35岁':
        return 3
    elif age_str == u'36-45岁':
        return 4
    elif age_str == u'46-55岁':
        return 5
    elif age_str == u'56岁以上':
        return 6
    else:
        return -1
        
def convert_regtime(df):
    if df < 10 and df>-1:
        return 0
    elif df >= 10 and df < 30:
        return 1
    elif df >= 30 and df < 60:
        return 2
    elif df >= 60 and df < 120:
        return 3
    elif df >= 120 and df < 360:
        return 4
    elif df >= 360:
        return 5
    else:
        return -1

    
def convert_last_comments(time):
    comment_date_end = time[0:10]
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break        
    return comment_date_begin

def convert_datetime(x):
    return datetime.strptime(x,"%Y-%m-%d-%H-%M-%S")    

def convert_time_to_day(x):
    x = x.replace(' ','-')
    x = x.replace(':','-')
    x = x[0:10]
    return x.replace('-','')

def get_actions(start_date, end_date):
    action_1 = pd.read_csv(action_1_path)
    action_2 = pd.read_csv(action_2_path)
    action_3 = pd.read_csv(action_3_path)
    actions = pd.concat([action_1, action_2, action_3]) # type: pd.DataFrame
    actions = actions[(actions.time >= start_date) & (actions.time < end_date)]
    return actions

def get_user_feature(data,start_date,end_date):
    User_feacture_C = [(lambda x:('UC'+ '_'+ start_date[5:]+ '_'+end_date[5:]+ '_' + str(x).zfill(2))) (x)  for x in range(11)]
    data = data[(data.time >= start_date) & (data.time < end_date)]
    df = pd.get_dummies(data['type'], prefix='user_action')
    t1 = pd.concat([data['user_id'], df], axis=1)
    t1 = t1.groupby(['user_id'], as_index=False).sum()
    t1['user_action_1_ratio'] = t1['user_action_4'] / t1['user_action_1']
    t1['user_action_2_ratio'] = t1['user_action_4'] / t1['user_action_2']
    t1['user_action_3_ratio'] = t1['user_action_4'] / t1['user_action_3']
    t1['user_action_5_ratio'] = t1['user_action_4'] / t1['user_action_5']
    t1['user_action_6_ratio'] = t1['user_action_4'] / t1['user_action_6']
    t1 = t1.fillna(-1)  
    t1.columns = ['user_id'] + User_feacture_C
    return t1

def get_sku_feature(data,start_date,end_date):
    Sku_feacture_C = [(lambda x:('Sk'+ '_'+ start_date[5:]+ '_'+end_date[5:]+ '_' + str(x).zfill(2))) (x)  for x in range(11)]
    data = data[(data.time >= start_date) & (data.time < end_date)]
    df = pd.get_dummies(data['type'], prefix='sku_action')
    t1 = pd.concat([data['sku_id'], df], axis=1)
    t1 = t1.groupby(['sku_id'], as_index=False).sum()
    t1['product_action_1_ratio'] = t1['sku_action_4'] / t1['sku_action_1']
    t1['product_action_2_ratio'] = t1['sku_action_4'] / t1['sku_action_2']
    t1['product_action_3_ratio'] = t1['sku_action_4'] / t1['sku_action_3']
    t1['product_action_5_ratio'] = t1['sku_action_4'] / t1['sku_action_5']
    t1['product_action_6_ratio'] = t1['sku_action_4'] / t1['sku_action_6']    
    t1 = t1.fillna(-1)     
    t1.columns = ['sku_id'] + Sku_feacture_C
    return t1
    
def get_sc_feature(data,start_date,end_date):    
    Sc_feacture_C = [(lambda x:('SC'+ '_'+ start_date[5:]+ '_'+end_date[5:]+ '_' + str(x).zfill(2))) (x)  for x in range(11)]
    data = data[(data.time >= start_date) & (data.time < end_date)]
    df = pd.get_dummies(data['type'], prefix='us_action')
    t = pd.concat([data[['user_id','sku_id']], df], axis=1)
    t = t.groupby(['user_id','sku_id'], as_index=False).sum()    
    t['us_action_1_ratio'] = t['us_action_4'] / t['us_action_1']
    t['us_action_2_ratio'] = t['us_action_4'] / t['us_action_2']
    t['us_action_3_ratio'] = t['us_action_4'] / t['us_action_3']
    t['us_action_5_ratio'] = t['us_action_4'] / t['us_action_5']
    t['us_action_6_ratio'] = t['us_action_4'] / t['us_action_6'] 
    t = t.fillna(-1)     
    t.columns = ['user_id','sku_id'] + Sc_feacture_C
    return t

def get_uc_feature(data,start_date,end_date):    
    Uc_feacture_C = [(lambda x:('SC'+ '_'+ start_date[5:]+ '_'+end_date[5:]+ '_' + str(x).zfill(2))) (x)  for x in range(11)]
    data = data[(data.time >= start_date) & (data.time < end_date)]
    df = pd.get_dummies(data['type'], prefix='uc_action')
    t = pd.concat([data[['user_id','cate']], df], axis=1)
    t = t.groupby(['user_id','cate'], as_index=False).sum()    
    t['uc_action_1_ratio'] = t['uc_action_4'] / t['uc_action_1']
    t['uc_action_2_ratio'] = t['uc_action_4'] / t['uc_action_2']
    t['uc_action_3_ratio'] = t['uc_action_4'] / t['uc_action_3']
    t['uc_action_5_ratio'] = t['uc_action_4'] / t['uc_action_5']
    t['uc_action_6_ratio'] = t['uc_action_4'] / t['uc_action_6'] 
    t = t.fillna(-1)     
    t.columns = ['user_id','cate'] + Uc_feacture_C
    return t
  
def get_comments_product_feat(start_date, end_date):
    comments = pd.read_csv(comment_path)
    comment_date_end = end_date
    comment_date_begin = comment_date[0]
    for date in reversed(comment_date):
        if date < comment_date_end:
            comment_date_begin = date
            break
    comments = comments[(comments.dt >= comment_date_begin) & (comments.dt < comment_date_end)]
    df = pd.get_dummies(comments['comment_num'], prefix='comment_num')
    comments = pd.concat([comments, df], axis=1) # type: pd.DataFrame
    #del comments['dt']
    #del comments['comment_num']
    comments = comments[['sku_id', 'has_bad_comment', 'bad_comment_rate', 'comment_num_1', 'comment_num_2', 'comment_num_3', 'comment_num_4']]
    return comments
    
    
def get_feature(file,train_start_date,train_end_date,test_start_date,test_end_date):
    User = pd.read_csv(user_path,encoding='gbk')
    train = get_actions(train_start_date,train_end_date)
    train.drop_duplicates(inplace='True')
    
    end = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=1)
    end = end.strftime('%Y-%m-%d')
    #-------------------user_feature----------------
    t = train[['user_id']]
    t = pd.merge(t,User,how='left',on='user_id')
    #年龄、性别、等级
    t['age'] = t['age'].map(convert_age)
    age_df = pd.get_dummies(t["age"], prefix="age")
    sex_df = pd.get_dummies(t["sex"], prefix="sex")
    user_lv_df = pd.get_dummies(t["user_lv_cd"], prefix="user_lv_cd")
    t = pd.concat([t[['user_id','user_reg_tm']], age_df, sex_df, user_lv_df], axis=1)
    #注册时间
    t['user_reg_tm'] = pd.to_datetime(t['user_reg_tm'])
    reg_dis = pd.to_datetime(train_end_date) - t['user_reg_tm']
    reg_dis = reg_dis.fillna(-1)
    t['user_reg_tm'] = reg_dis.map(lambda x: x.days)
    t['user_reg_tm'] = t['user_reg_tm'].map(convert_regtime)
    reg_time = pd.get_dummies(t['user_reg_tm'], prefix="reg_time")
    t.drop(['user_reg_tm'],axis=1,inplace=True)
    t = pd.concat([t,reg_time],axis=1)
    
    t.drop_duplicates(inplace='True')

    
    actions = None
    for i in (('00','02'),('00','06'),('00','12')):
        start_days = end + ' ' + i[0]
        end_time = end + ' '+ i[1]
        if actions is None:
            actions = get_user_feature(train,start_days,end_time)
            print(i)
        else:
            actions = pd.merge(actions, get_user_feature(train,start_days,end_time), how='right',
                               on=['user_id'])
            print(i)    
    actions = actions.fillna(0)
        
   
    for i in (1, 3, 5, 7, 10):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_user_feature(train,start_days,train_end_date)
            print(i)
        else:
            actions = pd.merge(actions, get_user_feature(train,start_days,train_end_date), how='right',
                               on=['user_id'])
            print(i)    
    actions = actions.fillna(0)
    
    user_feature = pd.merge(t,actions,how='left',on='user_id')
    user_feature = user_feature.fillna(0)
    user_feature.drop_duplicates(inplace='True')
    user_feature.to_csv('../data/feature/'+'user_feature'+str(train_start_date)+'_'+str(train_end_date)+'.csv',index=None)
    
    print ("--------------user_feature finished-------------")
    #-------------------sku_feature----------------
    t = train[['sku_id','cate','brand']]
    product = pd.read_csv(product_path)
    attr1_df = pd.get_dummies(product["a1"], prefix="a1")
    attr2_df = pd.get_dummies(product["a2"], prefix="a2")
    attr3_df = pd.get_dummies(product["a3"], prefix="a3")
    product = pd.concat([product[['sku_id']], attr1_df, attr2_df, attr3_df], axis=1)
    t = pd.merge(t,product,how='left',on='sku_id')
    t.drop_duplicates(inplace='True')
    
    actions = None
    for i in (('00','02'),('00','06'),('00','12')):
        start_days = end + ' '+ i[0]
        end_time = end + ' '+ i[1]
        if actions is None:
            actions = get_sku_feature(train,start_days,end_time)
            print(i)
        else:
            actions = pd.merge(actions, get_sku_feature(train,start_days,end_time), how='right',
                               on=['sku_id'])
            print(i)    
    actions = actions.fillna(0)
        
   
    for i in (1, 3, 5, 7, 10):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_sku_feature(train,start_days,train_end_date)
            print(i)
        else:
            actions = pd.merge(actions, get_sku_feature(train,start_days,train_end_date), how='right',
                               on=['sku_id'])
            print(i)    
    actions = actions.fillna(0)

    t2 = train[['sku_id','user_id','type']]
    t2 = t2[t2.type==4]
    t2 = t2[['sku_id','user_id']]
    t2['rebought_num'] = 1
    t2 = t2.groupby(['sku_id','user_id']).agg('sum').reset_index()
    t3 = t2[['sku_id','rebought_num']]
    t2 = t2[t2.rebought_num>1]    
    t4 = t2[['sku_id','rebought_num']]
    t2 =t2[['sku_id']]
    t2['rebought_guest_num'] = 1
    t2 = t2.groupby(['sku_id']).agg('sum').reset_index()
    
    t3 = t3.groupby(['sku_id']).agg('sum').reset_index()
    t3.rename(columns={'rebought_num':'rebought_num1'},inplace=True)
    
    t4 = t4.groupby(['sku_id']).agg('sum').reset_index()
    t4 = pd.merge(t2,t4,how='left',on='sku_id')
    t4 = pd.merge(t4,t3,how='left',on='sku_id')
    t4['rebought_ratio'] = t4['rebought_guest_num']/t4['rebought_num1']
    t4['avg_rebought_guest_num'] =  t4['rebought_num']/t4['rebought_guest_num']
    t4 = t4[['sku_id','rebought_guest_num','avg_rebought_guest_num','rebought_ratio']] 
    
    
    
    sku_feature = pd.merge(t,actions,how='left',on='sku_id')
    sku_feature = pd.merge(sku_feature,t4,how='left',on='sku_id')
    sku_feature = sku_feature.fillna(0)
    sku_feature.drop_duplicates(inplace='True')    
    sku_feature.to_csv('../data/feature/'+'sku_feature'+str(train_start_date)+'_'+str(train_end_date)+'.csv',index=None)
    
    print ("--------------sku_feature finished-------------")    
    #-------------------user-sku_feature----------    
    actions = None
    for i in (('00','02'),('00','06'),('00','12')):
        start_days = end + ' '+ i[0]
        end_time = end + ' '+ i[1]
        if actions is None:
            actions = get_sc_feature(train,start_days,end_time)
            print(i)
        else:
            actions = pd.merge(actions, get_sc_feature(train,start_days,end_time), how='right',
                               on=['user_id','sku_id'])
            print(i)    
    actions = actions.fillna(0)
        
   
    for i in (1, 3, 5, 7, 10):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_sc_feature(train,start_days,train_end_date)
            print(i)
        else:
            actions = pd.merge(actions, get_sc_feature(train,start_days,train_end_date), how='right',
                               on=['user_id','sku_id'])
            print(i)    
    actions = actions.fillna(0)
    
    us_feature = actions
    us_feature.to_csv('../data/feature/'+'us_feature'+str(train_start_date)+'_'+str(train_end_date)+'.csv',index=None)
    
    print ("--------------us_feature finished-------------")
    #-------------------user-category_feature---------- 
    actions = None
    for i in (('00','02'),('00','06'),('00','12')):
        start_days = end + ' '+ i[0]
        end_time = end + ' '+ i[1]
        if actions is None:
            actions = get_uc_feature(train,start_days,end_time)
            print(i)
        else:
            actions = pd.merge(actions, get_uc_feature(train,start_days,end_time), how='right',
                               on=['user_id','cate'])
            print(i)    
    actions = actions.fillna(0)
        
   
    for i in (1, 3, 5, 7, 10):
        start_days = datetime.strptime(train_end_date, '%Y-%m-%d') - timedelta(days=i)
        start_days = start_days.strftime('%Y-%m-%d')
        if actions is None:
            actions = get_uc_feature(train,start_days,train_end_date)
            print(i)
        else:
            actions = pd.merge(actions, get_uc_feature(train,start_days,train_end_date), how='right',
                               on=['user_id','cate'])
            print(i)    
    actions = actions.fillna(0)
    
    uc_featrue = actions
    uc_featrue.cate = uc_featrue.cate.astype('int64')
    uc_featrue.drop_duplicates(inplace='True')    

    uc_featrue.to_csv('../data/feature/'+'uc_featrue'+str(train_start_date)+'_'+str(train_end_date)+'.csv',index=None)
    
    print ("--------------uc_feature finished-------------")   
    #-------------------other feature----------------
#==============================================================================
#     other_feature = train[['time']]
#     other_feature.drop_duplicates(inplace = True)
#     other_feature['time1'] = other_feature.time.apply(lambda x: x.replace(' ','-'))
#     other_feature['time1'] = other_feature.time1.apply(lambda x: x.replace(':','-'))
#     other_feature['time1'] = other_feature.time1.map(convert_datetime)
#     other_feature['day_of_week'] =  [i.weekday()+1 for i in other_feature.time1]
#     weekday_dummies = pd.get_dummies(other_feature.day_of_week, prefix='weekday') 
#     other_feature['is_weekend'] = other_feature.day_of_week.apply(lambda x:1 if x in (6,7) else 0)    
#     other_feature = pd.concat([other_feature[['time','is_weekend']], weekday_dummies], axis=1)
#     other_feature.drop_duplicates(inplace='True')
#==============================================================================
    #-------------------sc feature----------------
#==============================================================================
#     t = train[['sku_id','time']]
#     t.drop_duplicates(inplace = True)
#     comments = pd.read_csv(comment_path)
#     t['dt'] = t['time'].map(convert_last_comments)    
#     t = pd.merge(t,comments,how='left',on=['sku_id','dt'])    
#     df = pd.get_dummies(t['comment_num'], prefix='comment_num')
#     t = pd.concat([t, df], axis=1) # type: pd.DataFrame
#     sc_feature = t[['sku_id','time', 'has_bad_comment', 'bad_comment_rate', 'comment_num_0.0','comment_num_1.0', 'comment_num_2.0', 'comment_num_3.0', 'comment_num_4.0']]    
#     sc_feature.drop_duplicates(inplace='True')
#==============================================================================
    comment_acc = get_comments_product_feat(train_start_date, train_end_date)
    
    #-------------------ut feature----------------    
#==============================================================================
#     t = train[['user_id','time']]
#     t['time1'] = t.time.map(convert_time_to_day)
#     t.drop_duplicates(inplace = True)
#     ut_feature = t
#     t = t.groupby(['user_id'])['time1'].agg(lambda x:':'.join(x)).reset_index()
#     t['action_number'] = t.time1.apply(lambda s:len(s.split(':')))
#     t = t[t.action_number>1]
#     t['max_time_action'] = t.time1.apply(lambda s:max([int(d) for d in s.split(':')]))
#     t['min_time_action'] = t.time1.apply(lambda s:min([int(d) for d in s.split(':')]))
#     t = t[['user_id','max_time_action','min_time_action']]    
# 
#     ut_feature = pd.merge(ut_feature,t,on=['user_id'],how='left')
#     ut_feature.time1 = ut_feature.time1.astype('int')
#     ut_feature['distance_lasttime'] = ut_feature.max_time_action - ut_feature.time1
#     ut_feature['distance_firsttime'] = ut_feature.time1 - ut_feature.min_time_action
#     ut_feature = ut_feature[['user_id','time1','distance_lasttime','distance_firsttime']]
#     ut_feature.time1 = ut_feature.time1.astype('str')    
#     ut_feature.drop_duplicates(inplace='True') 
#==============================================================================
    print ("--------------other_feature finished-------------")
    #-------------------dataset----------------  
    #del User,t,t1,t2,t3,t4,df
    #train['time1'] = train.time.map(convert_time_to_day)
    dataset1 = pd.merge(us_feature,user_feature,how='left',on='user_id')
    #del user_feature    
    #dataset1 = pd.merge(dataset1,ut_feature,how='left',on=['user_id','time1'])
   
    dataset1 = pd.merge(dataset1,sku_feature,how='left',on='sku_id')
    #dataset1 = pd.merge(dataset1,us_feature,how='left',on=['user_id','sku_id'])    
    #dataset1 = pd.merge(dataset1,other_feature,how='left',on=['time'])
    #dataset1 = pd.merge(dataset1,sc_feature,how='left',on=['sku_id‘])
    dataset1 = pd.merge(dataset1, comment_acc, how='left', on='sku_id')
    dataset1 = pd.merge(dataset1, uc_featrue, how='left', on=['user_id','cate'])

    dataset1.drop_duplicates(inplace=True)

    print ("--------------dataset merge finished-------------")
    #-------------------label----------------
    test = get_actions(test_start_date, test_end_date)
    test = test[test.type == 4]
    test = test[['user_id','sku_id']]
    test['label'] = 1
    test.drop_duplicates(inplace=True)
    
    dataset1 = pd.merge(dataset1,test,how='left',on=['user_id','sku_id'])
    dataset1.label = dataset1.label.fillna(0)
    dataset1.to_csv('../data/train/'+'dataset'+str(train_start_date)+'_'+str(train_end_date)+'.csv',index=None)

if __name__=='__main__':
    train_start_date = '2016-04-07'
    train_end_date = '2016-04-16'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'
    get_feature('../data/splitdata/Action4.csv',\
                  train_start_date,train_end_date,test_start_date,test_end_date)

    #==============================================================================
    # temp = user_feature[['user_id','age']]
    # temp = User[['user_id','age']]
    # temp.drop_duplicates(inplace='true')
    # for i in range(temp.shape[0]):
    #     if temp.iloc[i,1][0:2]=='56':
    #         temp.iloc[i,1] = 5
    #         continue
    #     if temp.iloc[i,1][0:2]=='46':
    #         temp.iloc[i,1] = 4
    #         continue
    #     if temp.iloc[i,1][0:2]=='36':
    #         temp.iloc[i,1] = 3
    #         continue 
    #     if temp.iloc[i,1][0:2]=='26':
    #         temp.iloc[i,1] = 2
    #         continue 
    #     if temp.iloc[i,1][0:2]=='16':   
    #         temp.iloc[i,1] = 1
    #         continue
    #     if temp.iloc[i,1][0:2]=='15': 
    #         temp.iloc[i,1] = 0
    #         continue
    #==============================================================================
#==============================================================================
#     t = pd.merge(t,temp,how='left',on='user_id')
#     t.age_x = t.age_y      
#     t = t.drop('age_y',1)
#     t.rename(columns={'age_x':'age'},inplace='True')
#     t = t.drop('user_reg_tm',1)
#     
#     t1 = train[train['type']==1]
#     t1 = t1[['user_id']]
#     t1['user_total_browse'] = 1
#     t1 = t1.groupby(['user_id']).agg('sum').reset_index()
#     
#     t2 = train[train['type']==2]
#     t2 = t2[['user_id']]
#     t2['user_total_basket'] = 1
#     t2 = t2.groupby(['user_id']).agg('sum').reset_index()
#     
#     t3 = train[train['type']==3]
#     t3 = t3[['user_id']]
#     t3['user_total_delete'] = 1
#     t3 = t3.groupby(['user_id']).agg('sum').reset_index()
#     
#     t4 = train[train['type']==4]
#     t4 = t4[['user_id']]
#     t4['user_total_buy'] = 1
#     t4 = t4.groupby(['user_id']).agg('sum').reset_index()
#     
#     t5 = train[train['type']==5]
#     t5 = t5[['user_id']]
#     t5['user_total_attention'] = 1
#     t5 = t5.groupby(['user_id']).agg('sum').reset_index()
#     
#     t6 = train[train['type']==6]
#     t6 = t6[['user_id']]
#     t6['user_total_click'] = 1
#     t6 = t6.groupby(['user_id']).agg('sum').reset_index()
#     
#     t7 = train[['user_id']]
#     t7['user_total_action'] = 1
#     t7 = t7.groupby(['user_id']).agg('sum').reset_index()
#==============================================================================
    
#==============================================================================
#     t1 = train[train['type']==1]
#     t1 = t1[['sku_id']]
#     t1['sku_total_browse'] = 1
#     t1 = t1.groupby(['sku_id']).agg('sum').reset_index()
#     
#     t2 = train[train['type']==2]
#     t2 = t2[['sku_id']]
#     t2['sku_total_basket'] = 1
#     t2 = t2.groupby(['sku_id']).agg('sum').reset_index()
#     
#     t3 = train[train['type']==3]
#     t3 = t3[['sku_id']]
#     t3['sku_total_delete'] = 1
#     t3 = t3.groupby(['sku_id']).agg('sum').reset_index()
#     
#     t4 = train[train['type']==4]
#     t4 = t4[['sku_id']]
#     t4['sku_total_buy'] = 1
#     t4 = t4.groupby(['sku_id']).agg('sum').reset_index()
#     
#     t5 = train[train['type']==5]
#     t5 = t5[['sku_id']]
#     t5['sku_total_attention'] = 1
#     t5 = t5.groupby(['sku_id']).agg('sum').reset_index()
#     
#     t6 = train[train['type']==6]
#     t6 = t6[['sku_id']]
#     t6['sku_total_click'] = 1
#     t6 = t6.groupby(['sku_id']).agg('sum').reset_index()
#     
#     t7 = train[['sku_id']]
#     t7['sku_total_action'] = 1
#     t7 = t7.groupby(['sku_id']).agg('sum').reset_index()
#==============================================================================
#==============================================================================
#     
#     t1 = train[train.type==1]
#     t1 = t1[['user_id','sku_id']]
#     t1['us_total_browse'] = 1
#     t1 = t1.groupby(['user_id','sku_id']).agg('sum').reset_index()
#     
#     t2 = train[train.type==2]
#     t2 = t2[['user_id','sku_id']]
#     t2['us_total_basket'] = 1
#     t2 = t2.groupby(['user_id','sku_id']).agg('sum').reset_index()
#     
#     t3 = train[train.type==3]
#     t3 = t3[['user_id','sku_id']]
#     t3['us_total_delete'] = 1
#     t3 = t3.groupby(['user_id','sku_id']).agg('sum').reset_index()
#     
#     t4 = train[train.type==4]
#     t4 = t4[['user_id','sku_id']]
#     t4['us_total_buy'] = 1
#     t4 = t4.groupby(['user_id','sku_id']).agg('sum').reset_index()
#     
#     t5 = train[train.type==5]
#     t5 = t5[['user_id','sku_id']]
#     t5['us_total_attention'] = 1
#     t5 = t5.groupby(['user_id','sku_id']).agg('sum').reset_index()
#     
#     t6 = train[train.type==6]
#     t6 = t6[['user_id','sku_id']]
#     t6['us_total_click'] = 1
#     t6 = t6.groupby(['user_id','sku_id']).agg('sum').reset_index()
#     
#     t7 = train[['user_id','sku_id']]
#     t7['us_total_action'] = 1
#     t7 = t7.groupby(['user_id','sku_id']).agg('sum').reset_index()
#==============================================================================
