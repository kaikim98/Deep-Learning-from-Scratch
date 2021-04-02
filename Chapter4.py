# %% Chapter 4 Neural Network
# 4.2 Loss function
# 4.2.1 Mean squared error
import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1,0.0,0.1,0.0,0.0]
mean_squared_error(np.array(y), np.array(t))

# 4.2.2 Cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1,0.0,0.1,0.0,0.0]
cross_entropy_error(np.array(y), np.array(t))

# %% 4.3 Numeriacal differentiation
# 4.3.1 Differentiation
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)

# 4.3.2 Example of numerical differentiation
def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x,y)
plt.show()
