import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random


def sigmoid(z):

    return 1 / (1 + np.exp(z))
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

def normal (an_array): 

 sum_of_rows = an_array.sum(axis=1)
 normalized_array = an_array / sum_of_rows[:, np.newaxis]

 return normalized_array  

def cost(theta, x, y):
    h = sigmoid(x)
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h)
    )
    grad = 1 / m * ((y - h) * x)
    return cost, grad 

def fit(x, y,max_iter, alpha):
    x = np.insert(x, 0, 1, axis=1)
    thetas = []
    classes = np.unique(y)
    costs = np.zeros(max_iter)

    for c in classes:
       
        binary_y = np.where(y == c, 1, 0)
        
        theta = np.zeros(x.shape[1])
        for epoch in range(max_iter):
            costs[epoch], grad = cost(theta, x, binary_y)
            theta += alpha * grad
            
        thetas.append(theta)
    return thetas, classes, costs

def normal (x):    
 norm = np.linalg.norm(x) 
 normal_array = x/norm
 return normal_array

def run ():
 #citire si pregatire a datelor   
 
 df.columns=["Col"+str(i) for i in range(0, 11)]
 x=df[['Col0','Col1','Col2','Col3','Col4','Col5','Col6','Col7','Col8','Col9']].values
 y=df[['Col10']].values
 benign = df.loc[y == 2]
 malignant = df.loc[y == 4]
 isinstance(x, np.float64)
 True
 isinstance(y,np.float64)
 True
 #Split

 split=0.8
 X_train,Y_train,X_test,Y_test=train_test_split(x,y,split)

 theta = np.zeros((1,10))
 learning_rate = 0.001
 no_of_iterations = 200000

 cost_1,grad=cost(X_train,Y_train,theta)
 print(cost_1,grad)

 thetas,classes,costs=fit(X_train,Y_train,no_of_iterations,learning_rate)
 print(thetas,classes,costs)







 

run() 