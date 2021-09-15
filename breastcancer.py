import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
    
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    # print(z)
    g = 1 / (1 + np.exp(-z))

    return g

def h_1(theta,x):
    h = sigmoid(np.dot(x, theta))

    return h

     
def computeCost(x, y, theta):
    m = y.size
    h = h_1(theta, x)
    J = -(np.sum(y * np.log(sigmoid(h)) + (1 - y) * np.log(1 - sigmoid(h))) / m)

    return J

def gradientDescent(x, y, theta, alpha, iterations):
    
    m = y.size
    theta = theta.copy()
    J_history = []
    theta_history = []

    for i in range(iterations):
        theta_history.append(list(theta))

        h = h_1(theta, x)
        theta[0] = theta[0] - (alpha/m) * (np.sum(h-y))
        theta[1] = theta[1] - (alpha/m) * (np.sum((h-y) * x[:, 1]))
        print(1)
        J_history.append(computeCost(x, y, theta))

    return theta, J_history, theta_history

    


def predict(theta, x):
    p = h_1(theta, x)
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p

def normal (x):    
    min = np.min(x)
    max = np.max(x)

    x_norm = (x - min) / (max - min)

    return x_norm

def run ():
    df = pd.read_csv (r"D:\ai intro\data.csv")
    x=df[['area_mean']].values
    y=df[['diagnosis']].values
    m, n = x.shape
    plt.hist(x[y == 'M'], bins=30)
    plt.xlabel("Aria medie - Malign")
    plt.ylabel("Frecventa") 
    plt.show()

    
    plt.hist(x[y == 'B'], bins=30)
    plt.xlabel("Aria medie - Belign")
    plt.ylabel("Frecventa")
    plt.show() 

    z = 0
    g = sigmoid(z)
    print('g(', z, ') =', g)
    x = np.concatenate([np.ones((m, 1)), x], axis=1)
    y[y == 'M'] = 1
    y[y == 'B'] = 0

    test_theta = np.array([13, 5])
    cost = computeCost(x, y, test_theta)
    print('Cu parametrii theta = [%d, %d] \nEroarea calculata = %.3f' % (test_theta[0], test_theta[1], cost))

    initial_theta = np.zeros(n+1)
    x=normal(x)

    iterations = 1000
    alpha = 0.001
    theta, J_history, theta_history = gradientDescent(x, y, initial_theta, alpha, iterations)
    print('Parametrii theta obtinuti cu gradient descent: {:.4f}, {:.4f}'.format(*theta))

    p = predict(theta, x)
    print('Acuratetea pe setul de antrenare: {:.2f} %'.format(np.mean(p == y) * 100))




run()