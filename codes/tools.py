# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 20:58:20 2017

@author: Thinkpad
"""

import numpy as np
import pandas as pd

def evaluation(answer,pre):
    a = (pd.merge(answer,pre,how='inner',on='user_id'))
    n11 = a.user_id.unique().shape[0]
    n12 = len(pd.merge(answer,pre,how='inner',on=['user_id','sku_id']))
    nda = answer.shape[0]
    npred = pre.shape[0]
    p11 = float(n11) / npred
    r11 = float(n11) / nda
    f11 = 6 * r11 * p11 / (5*r11 + p11)
    p12 = float(n12) / npred
    r12 = float(n12) / nda
    f12 = 5 * r12 * p12/(2*r12 + 3*p12)
    print("P11:",p11," R11:",r11," F11:",f11)
    print("P12:",p12," R12:",r12," F12:",f12)
    print("score:", 0.4*f11 + 0.6*f12)
    
def data_negative(data,sample_rate):
    dataset_negative = data[data.label==0]
    ix = [i for i in range(dataset_negative.shape[0])]
    dataset_negative.index = ix    
    idx = [i for i in range(dataset_negative.shape[0]) if i%sample_rate ==0]
    dataset_negative = dataset_negative.ix[idx]
    return dataset_negative
    
    
