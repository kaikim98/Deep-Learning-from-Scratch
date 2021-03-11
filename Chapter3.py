# %%Chapter3 Neural Network

import numpy as np

# %%3.2 Step function
# 3.2.2 Step function

# floating point
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
# include numpy array
def step_function_1(x):
    y = x > 0
    return y.astype(np.int)

# %%3.2.3 Graph of step function
import matplotlib.pylab as plt

def step_function_2(x):
    return np.array(x > 0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function_2(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

# %%3.2.4 Sigmoid function
def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()

# %%3.2.7 ReLU function
def relu(x):
    return np.maximum(0,x)

# %%3.3 Calculation of multidimensional arrays
# 3.3.1 multidimensional arrays-
import numpy as np
A = np.array([1, 2, 3,4])
print(A)
np.ndim(A)
A.shape
A.shape[0]

B = np.array([[1,2], [3,4], [5,6]])
print(B)
np.ndim(B)
B.shape

# %%3.3.2 Matrix multiplication
A = np.array([[1,2], [3,4]])
A.shape
B = np.array([[5,6], [7,8]])
B.shape
np.dot(A,B)

C = np.array([[1,2], [3,4]])
C.shape
A.shape
np.dot(A,C)

# %%3.3.3 Dot product of Neural network
X = np.array([1,2])
X.shape
W = np.array([[1,3,5], [2,4,6]])
print(W)
W.shape
Y = np.dot(X,W)
print(Y)

# %%3.4 Implement 3-layer Neural network
#3.4.2 Implement signal transduction of each layer
X = np.array([1.0, 0.5])
W1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape) # (2,3)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2,0.5], [0.3,0.6]])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
A2 = np.dot(Z1, W2) +B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

# %%3.4.3 Summary implement
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)

# %%3.5 Design output layer
#3.5.1 Implement Softmanx function
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)

# =============================================================================
# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a /sum_exp_a()
#     return y
# 
# =============================================================================
# %%3.5.2 Precaution during implementing Softmax function
a = np.array([1010, 1000,990])
np.exp(a)/np.sum(np.exp(a))
c = np.max(a)
a - c
np.exp(a-c)/ np.sum(np.exp(a-c))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    
    return y

# %%3.5.3 Feature of Softmax function
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
np.sum(y)

# %%3.6 Handwriting number recognition
# 3.6.1 MNIST Dataset
import tensorflow as tf
from keras.datasets import mnist
(x_train, t_train), (x_test, t_test) = mnist(flatten=True, normalize = False)

print(x_train.shape)
print(t_train,shape)
print(x_test.shape)
print(t_test.shape)