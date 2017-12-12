#Simple linear regression
import numpy as np
import matplotlib.pyplot as plt
def error_ana(data,m,b):
    total=0
    for i in range(len(data)):
        y=data[i][1]
        h=m*data[i][0]+b
        total+=(y-h)**2
    return total/len(data)
def run_grad(data,learning_rate,itr,m,b):
    n=len(data)
    for i in range(len(data)):
        temp_b=b-learning_rate*2/n*(m*data[i][0]+b-data[i][1])
        temp_m=b-learning_rate*2/n*(m*data[i][0]+b-data[i][1])*data[i][0]
        b=temp_b
        m=temp_m
    return [m,b]
def main():
    #y=m*x+b
    m=0
    b=0
    data=np.genfromtxt("data.csv",delimiter=',')
    print("initital error with m=",m,"and b=",b,"is ",error_ana(data,m,b))
    learning_rate=.0001
    itr=100
    
    for i in range(itr):
        m,b=run_grad(data,learning_rate,itr,m,b)
    print("final error with m=",m,"and b=",b,"is ",error_ana(data,m,b))
    plt.plot(data,'ro')
    x = np.linspace(0,100,100)
    plt.plot(x,m*x+b)
    plt.savefig("1.pdf")
    plt.show()
    

if __name__ == '__main__':
    main()