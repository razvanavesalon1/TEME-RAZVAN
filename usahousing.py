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

df = pd.read_csv (r"D:\ai intro\USA_Housing.csv") 
x=df['Avg. Area Number of Rooms']
y=df['Price']

plotData(x,y)
plt.show()
m=y.size
x = np.stack([np.ones(m), x], axis=1)
theta=np.array([-1,2])
print(h(theta,x))
J=computeCost(x,y,theta)
print('Cu parametrii theta = [-1, 2]\nEroarea calculata = %.2f' % J)