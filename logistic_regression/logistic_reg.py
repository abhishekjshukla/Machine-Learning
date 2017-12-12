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



scale=StandardScaler()
x=scale.fit_transform(x)

one=np.ones((99,1))
x=np.concatenate((one,x),axis=1)

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
	w=np.zeros(3,)
	yhat=np.zeros(99,)
	for i in range(epocs):
		w=step_gradient(x,y,yhat,w,rate)
		#print(w)
		yhat=x.dot(w)
		yhat=sig(yhat)
#		if(i%1000==0):
#			print(error(yhat,y))
	return w

weight=grad(x,y,50000,0.001)
yhat=sig(x.dot(weight))
final=predict(yhat)

from sklearn.metrics import accuracy_score
acc_scr_model=accuracy_score( final,y)

 #SK learn Model
from sklearn import linear_model
clf = linear_model.LogisticRegression(solver="lbfgs")
clf.fit(x, y)
coef=clf.coef_
inter=clf.intercept_
skl_pred=clf.predict(x)

acc_sklearn=accuracy_score(skl_pred,y)

from sklearn import linear_model
clf2 = linear_model.SGDClassifier(max_iter=50000,alpha=.0001)
clf2.fit(x,y)
sgd_pred=clf2.predict(x)
acc_sgd=accuracy_score(sgd_pred,y)

