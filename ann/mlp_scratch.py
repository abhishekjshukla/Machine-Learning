import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("ex2data1.csv")
x=data.iloc[:,:2].values
y=data.iloc[:,2:3].values
#fig = plt.figure(figsize=(4,4))
#plt.scatter(x[:,0],x[:,1], s=20, c=y, cmap=plt.cm.spring)
#plt.show()

def relu(z):
    z=np.maximum(0,z)
    return z
def sig(z):
	return 1/(1+np.exp(-z))
def der_sig(z):
	return z*(1-z)

def error(yhat,y):
	return 1/2*np.sum(((y-yhat)**2))
from sklearn.preprocessing import LabelEncoder,StandardScaler


np.random.seed(200)
w1=np.random.randn(x[0].shape[0],3)
scale=StandardScaler()
x=scale.fit_transform(x)
w2=np.random.randn(3,1)
yhat=0
z_hiddenLayer=0
def forward(x,w1,w2):
	global yhat,z_hiddenLayer
	z_hiddenLayer=sig(x.dot(w1))
	yhat=sig(z_hiddenLayer.dot(w2))
	return yhat
forward(x,w1,w2)
#backpropagation

for epocs in range(10000):
	w2upd=w1upd=0
	for i in range(x.shape[0]):
		#for layer 2
		delta_oj=(yhat[i]-y[i])[0]  #d(e)/d(oj)
		delta_oi=der_sig(yhat[i]) #d(oj)/d(oi)
		delta_wij=np.reshape(z_hiddenLayer[i],(3,1))
		w2upd+=(delta_oj*delta_oi)*delta_wij
		
		#for layer 2
		z_hidden=np.reshape(z_hiddenLayer[i],(1,3))
		delta_hij=z_hidden
		delta_hij=np.concatenate((z_hidden,delta_hij))
		delta_der=der_sig(z_hidden)
		delta_der=np.concatenate((der_sig(z_hidden),delta_der))
		x_temp=np.reshape(x[i],(2,1))
		delta_w1ij=np.zeros((2,3))
		for i in range(3):
			delta_w1ij[:,i:i+1]=x_temp
		w1upd+=delta_oj*delta_oi*delta_hij*delta_der*delta_w1ij
	w2=w2-.0001*w2upd	
	w1=w1-.0001*w1upd	
	forward(x,w1,w2)
	if(epocs%1000==0):
		print(error(y,yhat))
	
final=[]
for i in range(99):
	if yhat[i]>.5:
		final.append(1)
	else:
		final.append(0)		
from sklearn.metrics import accuracy_score
acc_scr_model=accuracy_score( final,y)
	







