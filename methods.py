import numpy as np
import math

def forward_backward(grad_g, prox_f, p, proj, gamma = 0.1, K = 1000):
    for k in range(K):
        y = x - gamma*grad_g(x, p ,proj)
        x = prox_f(y)
    return x

def grad_g(x, p, proj):
    res = np.zeros(p.shape[1])
    A = p.T @ p
    b = p.T @ proj
    for i in p.shape[1]:
        res[i] = 2*A[:,i].T@(A@x-b)
    
    return res

def prox_f(x, gamma):
    res = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        res[i] = np.sign(x[i])*max(x-gamma, 0)
    
    return res
