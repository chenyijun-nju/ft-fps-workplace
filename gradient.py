import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

data = scio.loadmat('D:/codetest/working_space/a1data.mat')

def J(t, X, Y):
    thou = np.transpose(Y - X.dot(t)).dot(Y - X.dot(t))
    return thou / 1000

X1 = data['x']
X1 = np.asarray(X1)
print(X1.shape)
Y = data['y']
Y = np.asarray(Y)
xx = np.squeeze(X1)
yy = np.squeeze(Y)

theta = np.array([[100,100]])
# theta = np.ones((2,1)) 
theta = theta.T
print(theta.shape)
itr =20000
sigma = 0.01
alpha = 0.001
fitness = []

X = np.array([np.ones((1,1000)), np.transpose(X1)])
X = np.transpose(X)
X = np.squeeze(X)
print(X.shape)

for i in range(itr):
    er = J(theta, X, Y)
    er = np.squeeze(er)
    #print(er) 
    theta = theta + alpha * np.transpose(X).dot(Y - X.dot(theta)) * 0.001
    # print(theta)
    fitness.append(er)

b = theta[0]
w = theta[1]
# print(theta)
b = np.squeeze(b)
w = np.squeeze(w)
print(b)
print(w)

plt.figure(1)
plt.title("gradient decent loss")
plt.xlabel("iterators", size=14)
plt.ylabel("fitness", size=14)
t = np.array([t for t in range(0, 20000)])
fitness = np.array(fitness)
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()

plt.figure(2)
plt.title("gradient decent line")
plt.xlabel("x", size=14)
plt.ylabel("y", size=14)
plt.xlim(xmax=20,xmin=-20)
plt.ylim(ymax=35,ymin=-20)
plt.scatter(xx, yy)
a = np.linspace(-20,20,10000)
d = w*a + b
plt.plot(a, d, c='b')
plt.show()

    
