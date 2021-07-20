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
                
    def key in params.keys():
        self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
        params[key] += self.v[key]