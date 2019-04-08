# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

df = pd.read_csv('/Users/Noa/Documents/train.csv')
df.keys()
df['question_text'].head(10)

df[df['target']==1].shape[0]

####imbalanced data!!!

#import pandas_ml as pdml
import random
from pandas import Series,DataFrame
 
d_class0 = df[df['target'] == 0]
d_class1 = df[df['target'] == 1]
    
numRows_class0 = len(d_class0.index)
numRows_class1 = len(d_class1.index)
    
# downsample the class 0
    
d_class0_downsampled = d_class0.sample(n=numRows_class1,replace=False, random_state=42)
    
# new output data frame containing 1:1 class ratios
    
data_set = DataFrame()
data_set = data_set.append(d_class0_downsampled)
data_set = data_set.append(d_class1)
    
# shuffle the rows
    
numRows_data_set =len(data_set.index)
data_set = data_set.sample(n=numRows_data_set, replace= False)
