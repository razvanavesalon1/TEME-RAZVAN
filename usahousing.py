import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
def plotData(X,y):
 plt.figure(1) 
 plt.scatter(X,y,c="red",
           marker="x")         
 plt.xlabel("Avg. Area Number of Rooms")
 plt.ylabel("Price")

def h(theta,x):

    h=np.dot(x,theta)
    return h 

def computeCost(x,y,theta):
    m = y.size  
    J = 0
    o=np.ones(m)
    squared_error=(np.dot(x,theta)-y)**2
    J=np.dot(o,squared_error)/(2*m)
    return J

def gradientDescent(x, y, theta, alpha, num_iters):
    m = y.size 
    
  
    theta = theta.copy()
    J_history = [] 
    theta_history = []   
    
    for i in range(num_iters):
        
        theta_history.append(list(theta))
        h=np.dot(x,theta)
        theta[0]=theta[0]-(alpha/m)*(np.sum(h-y))
        theta[1]=theta[1]-(alpha/m)*(np.sum((h-y)*x[:,1]))
        J_history.append(computeCost(x, y, theta))
    
    return theta, J_history, theta_history


def plotConvergence(J_history):
    
    plt.figure(figsize=(10,6))
    plt.plot(range(len(J_history)),J_history,'bo')
    plt.grid(True)
    plt.title("Convergenta functiei eroare")
    plt.xlabel("Numarul de iteratii")
    plt.ylabel("Functia eroare")
               

df = pd.read_csv (r"D:\ai intro\USA_Housing.csv") 
x=df['Avg. Area Number of Rooms']
y=df['Price']

plotData(x,y)
plt.show()
m=y.size



x = np.stack([np.ones(m), x], axis=1)
theta=np.array([-8855800,1550000])
print(h(theta,x))

J=computeCost(x,y,theta)
print('Cu parametrii theta = [-1, 2]\nEroarea calculata = %.2f' % J)

theta = np.zeros(2)

iterations = 1500
alpha = 0.01

theta, J_history, theta_history = gradientDescent(x ,y, theta, alpha, iterations)
print('Parametrii theta obtinuti cu gradient descent: {:.4f}, {:.4f}'.format(*theta))

plotData(x[:, 1],y)
plt.plot(x[:, 1], h(theta,x), '-') 
plt.legend(['Training data', 'Linear regression' + ' h(x) = %0.2f + %0.2f x'  %(theta[0],theta[1])])
plt.show()

predict1 = h(theta, np.array([1, 3.5]))
print('Pentru Avg. Area Number of Rooms=3.5, modelul prezice un pret de {:.2f}'.format(predict1*10000))

predict2 = h(theta, np.array([1, 7]))
print('Pentru Avg. Area Number of Rooms=7, modelul prezice un pret de {:.2f}'.format(predict2*10000))

plotConvergence(J_history)
