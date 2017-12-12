# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:55:19 2017

@author: wolf
"""

import numpy as np
import pandas as pd

df=pd.read_csv("email_spam.csv")

data=df.values

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

y=np.asarray(data[:,1],dtype="int64")

x=data[:,2:]

x[:,0]=lb.fit_transform(x[:,0])