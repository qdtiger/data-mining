# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:03:10 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd
from datetime import datetime

User = pd.read_csv('../data/JData_user.csv')
Product = pd.read_csv('../data/JData_Product.csv')
Comment = pd.read_csv('../data/JData_Comment(modified).csv')
Action_2 = pd.read_csv('../data/JData_Action_201602.csv')
Action_3 = pd.read_csv('../data/JData_Action_201603.csv')
Action_3_extra = pd.read_csv('../data/JData_Action_201603_extra.csv')
Action_4 = pd.read_csv('../data/JData_Action_201604.csv')
#--------Action_4------------
Action_4['time'] = Action_4.time.apply(lambda x: x.replace(' ','-'))
Action_4['time'] = Action_4.time.apply(lambda x: x.replace(':','-'))
Action_4['time'] = Action_4.time.apply(lambda x: datetime.strptime(x,"%Y-%m-%d-%H-%M-%S"))
Action_4['month'] = [i.month for i in Action_4.time]
Action_4['day'] = [i.day for i in Action_4.time]
Action_4['hour'] = [i.hour for i in Action_4.time]
Action_4.to_csv('../data/splitdata/Action4.csv',index=None)
#---------S----------
sku2 = Action_2.sku_id.unique()
sku3 = Action_3.sku_id.unique()
sku3_extra = Action_3_extra.sku_id.unique()
sku4 = Action_4.sku_id.unique()
c=list(set(sku4).union(set(sku2)))
c=list(set(c).union(set(sku3)))
c=list(set(c).union(set(sku3_extra)))
All_product = pd.DataFrame({'sku_id':c})
All_product.to_csv('../data/splitdata/all_product.csv')


