# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 11:18:39 2017

@author: Thinkpad
"""
import numpy as np
import pandas as pd

dataset1_negative = dataset1[dataset1.label==0]
ix = [i for i in range(dataset1_negative.shape[0])]
dataset1_negative.index = ix

idx = [i for i in range(dataset1_negative.shape[0]) if i%10 ==0]
dataset1_negative = dataset1_negative.ix[idx]

dataset1_negative.to_csv('../data/train/dataset1_negative.csv',index=None)

