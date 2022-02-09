from matplotlib.pyplot import xcorr
from Mesh import Mesh
import numpy as np
import cvxpy as cp
import math
import nevergrad as ng

def mean_curv(x, Gx, Gy):

    grad = np.hstack( [Gx@x, Gy@x] )

    aux = np.zeros(Gx.shape[0])
    for i in range(Gx.shape[0]):
        aux[i] = 1/np.sqrt(1 + np.linalg.norm(grad[i])**2  )  

    aux_x = np.multiply(aux,Gx@x)
    aux_y = np.multiply(aux,Gy@x)
    reg = Gx.T@aux_x + Gy.T@aux_y

    regularizer = np.sum( np.abs(reg) )

    return regularizer

def loss_function(x,  p, proj):
    res = np.linalg.norm(p.T @ p @ x - p.T @ proj)**2
    return res

def final_total(x, Gx, Gy, p, proj, regularizer):
    res = loss_function(x, p, proj)
    reg = regularizer(x,Gx,Gy) 
    return res + 8e-8*reg