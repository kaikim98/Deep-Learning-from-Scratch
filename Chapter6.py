# -*- coding: utf-8 -*-
# Chapter 6 Learning techniques
# Chapter 6.1 Renew Parameter
# Chapter 6.1.2 SGD(Stochastic Gradient Descent)
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr #learning rate
        
    def update(self, params, grads): # params, grads : Dictionary variable. 
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# %% 6.1.4 Momentum
class Momentum:
    def __init__(self, lr=0.01, momentum= 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None # v : velocity
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
    for key in params.keys():
        self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
        params[key] += self.v[key]
        
# %% 6.1.5 AdaGrad
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h = None:
            self.h = {}
            for key, val in params.items():
                self.h[key] =  np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key])+1e-7)
            

# %% 6.2.2 Distribution of Activation Value in Hidden Layer
# Initialize weight - Normal distribution with standard deviation value 0.01
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) *0.01
    a = np.dot(x,w)
    z = sigmoid(a)
    activations[i]=z
    
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# Xavier Initalization
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))


node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    a = np.dot(x,w)
    z = sigmoid(a)
    activations[i]=z
    
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

# %% 6.4.1 Overfitting

(x_trian, t_train), (x_test, t_test) = load_mnist(normalize = True)
# For implement overfitting, reduce the number of train data
x_train = x_train[:300]
t_train = t_train[:300]

network = MultiLayerNet(input_size = 784, hidden_size_list = [100,100,100,100,100,100], output_size = 10)
optimizer = SGD(lr = 0.01)
max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0
for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        epoch_cnt += 1
        if epoch_cnt > max_epochs:
            break
        