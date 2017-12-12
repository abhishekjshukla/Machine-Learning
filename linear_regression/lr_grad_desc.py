#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    x=points[:,0:1]
    y=points[:,1:2]
    yhat=m_current*x+b_current
    x_dummy=ones((1,len(points)))
    N = float(len(points))
    m_gradient = -2/N*((x.T).dot((y-(yhat))))
    b_gradient = -2/N*(x_dummy.dot(y-(yhat)))
    #print(m_gradient,b_gradient)
    new_b = b_current - (learningRate * b_gradient[0][0])
    new_m = m_current - (learningRate * m_gradient[0][0])
    #print(new_b,new_m)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data_siraj.csv", delimiter=",")
    learning_rate = 0.00001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    [b,m]=gradient_descent_runner(points,initial_b,initial_m,.0002,100000)
    print ("Aftdddder {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
if __name__ == '__main__':
    run()