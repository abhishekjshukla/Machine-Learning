# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:55:19 2017

@author: wolf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("train_titanic.csv")

# Preprossing data
from sklearn.preprocessing import LabelEncoder, Imputer,StandardScaler

lb=LabelEncoder()

imp=Imputer(axis=0)

y=np.asarray(df.iloc[:,1],dtype="int64")

x=df.loc[:,["Pclass","Sex","Age","SibSp","Parch"]]

x.iloc[:,1]=lb.fit_transform(x.iloc[:,1])
imp.fit(x.iloc[:,2:3])
x.iloc[:,2:3]=(imp.transform(x.iloc[:,2:3]))
x=x.values
#Scaling
sc=StandardScaler()
x=sc.fit_transform(x)

ones=np.ones((len(x),1))
x=np.concatenate((ones,x),axis=1)

def error(yhat,y):
	e=-(y.T.dot(np.log(yhat))+(1-y).T.dot(np.log(1-yhat)))
	return e
def sig(x):
	final=[]
	for i in range(len(x)):
		final.append(1/(1+np.exp(-x[i])))
	return np.array(final)
def predict(y):
	pred=[]
	for i in range(len(y)):
		if(y[i]>0.5):
			pred.append(1)
		else:
			pred.append(0)
	return np.array(pred)
def step_gradient(x,y,yhat,w,rate):
	w=w-rate*(((x.T).dot(yhat-y))/len(y))
	return w
def grad(x,y,epocs,rate):
	w=np.random.rand(6,)
	yhat=np.random.rand(891,)
	for i in range(epocs):
		w=step_gradient(x,y,yhat,w,rate)
		yhat=x.dot(w)
		yhat=sig(yhat)
#		if(i%1000==0):
#			print(error(yhat,y))
	return w
weight=grad(x,y,100000,.0001)

yhat=sig(x.dot(weight))

final=predict(yhat)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y,final)

#testdf=pd.read_csv("test_titanic.csv")
#
#
#x_test=testdf.loc[:,["Pclass","Sex","Age","SibSp","Parch"]]
#
#x_test.iloc[:,1]=lb.fit_transform(x_test.iloc[:,1])
#imp.fit(x_test.iloc[:,2:3])
#x_test.iloc[:,2:3]=(imp.transform(x_test.iloc[:,2:3]))
#x_test=x_test.values
##Scaling
#sc=StandardScaler()
#x_test=sc.fit_transform(x_test)
#ones=np.ones((len(x_test),1))
#x_test=np.concatenate((ones,x_test),axis=1)
#fd=open("titanic_sub.csv","w")
#
#yhat_test=sig(x_test.dot(weight))
#
#final_test=predict(yhat_test)
#fd.write("PassengerId,Survived\n")
#j=892
#for i in final_test:
#	fd.write(str(j)+","+str(i)+"\n")
#	j+=1
#fd.close()
#   