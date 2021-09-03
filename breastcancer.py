import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
def plotData(X,y,admitted,not_admitted):
    plt.scatter(admitted.iloc[:, 3], admitted.iloc[:, 10], s=10, label='Benign')
    plt.scatter(not_admitted.iloc[:, 3], not_admitted.iloc[:, 10], s=10, label='Malignant')
    plt.legend()
    
def sigmoid(z):
    
    z = np.array(z)
    g = np.zeros(z.shape)
    g=1 / (1 + np.exp(-z))
    return g      

def h(theta,x):
    h=sigmoid(np.dot(x,theta))
    return h 
     
def computeCost(x, y, theta):
    m = y.size 
    J =((np.sum(-y*np.log(h(theta,x))-(1-y)*(np.log(1-h(theta,x))))))/m
    return J

def gradientDescent(x, y, theta, alpha, num_iters):
    
    m = y.size 
    
   
    theta = theta.copy()
    print(theta.shape)
    
  
    J_history = [] 
    theta_history = []
    
    for i in range(num_iters):
        
        theta_history.append(list(theta))
        gradient = np.dot(x.T, (h(theta,x) - y)) /m
        theta=theta-alpha*gradient
        J_history.append(computeCost(x, y, theta))
    
    return theta, J_history, theta_history\
    


def predict(theta, x):

    return h(theta, x)

def normal (x):    
 norm = np.linalg.norm(x) 
 normal_array = x/norm
 return x    

def run ():
    df = pd.read_csv (r"D:\ai intro\breast-cancer-wisconsin.csv")
    df.columns=["Col"+str(i) for i in range(0, 11)]
    x=df[['Col3']].values
    y=df[['Col10']].values
    benign = df.loc[y == 2]
    malignant = df.loc[y == 4]
    plotData(x,y,benign,malignant)
    plt.show()
    m, n = x.shape
    x = np.concatenate([np.ones((m, 1)), x], axis=1 ) 
    initial_theta = np.zeros(n+1)
    initial_theta = initial_theta[:, np.newaxis]
    cost = computeCost(x, y, initial_theta)
    print('Cu parametrii theta = [0, 0, 0] \nEroarea calculata = %.3f' % cost)
    test_theta = np.array([24, 2])
    test_theta= test_theta[:, np.newaxis]
    print(test_theta)
    cost = computeCost(x, y, test_theta)
    print('Cu parametrii theta = [-24, 0.2] \nEroarea calculata = %.3f' % cost)
    iterations = 10000
    alpha = 0.001
    theta, J_history, theta_history = gradientDescent(x ,y, initial_theta, alpha, iterations)
    print('Parametrii theta obtinuti cu gradient descent:',theta)
    p = predict(theta, x)
    print('Acuratetea pe setul de antrenare: {:.2f} %'.format(np.mean(p == y) * 100)) 
    print('Accuratetea asteptata (approx): 89.00 %')


run()