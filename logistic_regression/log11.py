import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("ex2data1.csv")
x=data.iloc[:,:2].values
y=data.iloc[:,2].values

from sklearn.preprocessing import LabelEncoder,StandardScaler
label=LabelEncoder()
x[:,0]=label.fit_transform(x[:,0])

one=np.ones((99,1),dtype="int64")
x=np.array(x,dtype="int64")
#x=np.concatenate((one,x),axis=1)

scale=StandardScaler()
x=scale.fit_transform(x)

def error(yhat,y):
	e=-(y.T.dot(np.log(yhat))+(1-y).T.dot(np.log(1-yhat)))
	return e
def sig(x):
	final=[]
	for i in range(len(x)):
		final.append(1/(1+math.exp(-x[i])))
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
	w=np.random.randn(2,)
	yhat=np.random.randn(99,)
	for i in range(epocs):
		w=step_gradient(x,y,yhat,w,rate)
		yhat=x.dot(w)
		yhat=sig(yhat)
#		if(i%1000==0):
#			print(error(yhat,y))
	return w

weight=grad(x,y,100000,0.0001)
yhat=sig(x.dot(weight))
final=predict(yhat)

from sklearn.metrics import accuracy_score
acc_scr_model=accuracy_score( final,y)
	
# SK learn Model
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(x, y)
coef=clf.coef_
inter=clf.intercept_
skl_pred=clf.predict(x)

acc_sklearn=accuracy_score(skl_pred,y)
print(acc_sklearn,acc_scr_model)
print(coef,inter,weight)
