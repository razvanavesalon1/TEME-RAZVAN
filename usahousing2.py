import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D
def plotData(X,y):
 plt.figure(1) 
 plt.scatter(X,y,c="red",
           marker="x")         
 plt.xlabel("Avg. Area Number of Rooms")
 plt.ylabel("Price")

def train_test_split(X, Y, split):

    indices = np.array(range(len(X)))
    
    train_size = round(split * len(X))

    random.shuffle(indices)

    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(X)]

    X_train = X[train_indices, :]
    X_test = X[test_indices, :]
    Y_train = Y[train_indices, :]
    Y_test = Y[test_indices, :]
    
    return X_train,Y_train, X_test, Y_test

def normal_equation(X, Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))
    return beta

def predict(X_test, beta):
    return np.dot(X_test, beta)   

def metrix(predictions, Y_test):

   
    MAE = np.mean(np.abs(predictions-Y_test))

    
    MSE = np.square(np.subtract(Y_test,predictions)).mean() 
    RMSE = math.sqrt(MSE)

   
    rss = np.sum(np.square((Y_test- predictions)))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    r_square = 1 - (rss/sst)
    

    return MAE, RMSE, r_square

def cost_function(X, y, theta):
    m = y.size
    error = np.dot(X, theta.T) - y
    cost = 1/(2*m) * np.dot(error.T, error)
    return cost, error

def gradient_descent(X, y, theta, alpha, iters):
    cost_array =[]
    m = y.size
    for i in range(iters):
        cost, error = cost_function(X, y, theta)
        theta =theta-(alpha * (1/m) * np.dot(X.T,error))
        cost_array.append(cost)
    return theta, cost_array

def plotChart(iterations, cost_num):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost_num, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs Iterations')
    plt.style.use('fivethirtyeight')
    plt.show()

def normal (x):    
 norm = np.linalg.norm(x) 
 normal_array = x/norm
 return x

    


df = pd.read_csv (r"D:\ai intro\USA_Housing.csv") 
x=df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']].values
y=df['Price'].values
m=len(y)
print(m)
copy_X=x
copy_y=y

one = np.ones((len(x),1))
x = np.append(one, x, axis=1)
y = np.array(y).reshape((len(y),1))
print(x.shape)
print(y.shape)
alpha = 0.01
iterations = 1000
split = 0.8
X_train, Y_train, X_test, Y_test = train_test_split(x, y, split)
print ("TRAINING SET")
print("X_train.shape: ", X_train.shape)
print("Y_train.shape: ", Y_train.shape)

print("TESTING SET")
print("X_test.shape: ", X_test.shape)
print("Y_test.shape: ", Y_test.shape)

X_train=normal(X_train)
X_test=normal(X_test)
beta = normal_equation(X_train, Y_train)
predictions = predict(X_test, beta)

print(predictions.shape)
mae, rmse, r_square = metrix(predictions, Y_test)
print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
print("R square: ", r_square)
X=normal(copy_X)
Y=normal(copy_y)

X = (X - X.mean()) / X.std()
X = np.c_[np.ones(X.shape[0]), X] 
theta = np.array([1,2,-3,7,8,-5])

initial_cost,errors = cost_function(X, Y, theta)
print('With initial theta values of {0}, cost error is {1}'.format(theta, initial_cost))

theta, cost_num = gradient_descent(X, Y, theta, alpha, iterations)
plotChart(iterations, cost_num)
final_cost,errors = cost_function(X, Y, theta)
print('With final theta values of {0}, cost error is {1}'.format(theta, final_cost))