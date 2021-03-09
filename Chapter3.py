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
