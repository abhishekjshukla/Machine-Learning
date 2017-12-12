# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:57:36 2017

@author: wolf
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
data=pd.read_csv("data_MR.csv").values

x=data[:,:2]
y=data[:,2:3]

plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y,color="r")
plt.show()
x_trans=np.transpose(x)
w=np.linalg.solve(x_trans.dot(x),x_trans.dot(y))

Yhat=x.dot(w)

d1 = y - Yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)