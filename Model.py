from cvxpy.problems.objective import Objective
from cvxpy.settings import CPLEX
from matplotlib.pyplot import xcorr
from Mesh import Mesh
import numpy as np
import cvxpy as cp
import math
import nevergrad as ng


def model(m, p, proj, shape):
    x = cp.Variable(shape)
    constraints = []

    regularizer = 0

    for i in range(shape[0]):
        for j in range(shape[1]):
            if i < (shape[0] - 1) and j < (shape[1] - 1):
                iM = min(shape[0]-1,i+1) 
                jM = min(shape[1]-1,j+1) 
                regularizer +=  (x[iM,j] - x[i,j])**2 + (x[i,jM] - x[i,j])**2 

    objective = cp.norm((p.T @ p )@cp.reshape(x, (shape[0]*shape[1],))  -   p.T @ proj ) 

    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model

def model2(m, p, proj, shape):
    x = cp.Variable(m.nb_nodes)
    constraints = []

    y = cp.reshape(x,(10,10))
    regularizer = 0

    for i in range(shape[0]):
        for j in range(shape[1]):
            if not i % 2:
                iM = min(shape[0]-1,i+1) 
                jM = min(shape[1]-1,j+1) 
                regularizer +=  (y[iM,j] - y[i,j])**2 + (y[i,jM] - y[i,j])**2 
            else:
                im = max(0,i-1) 
                jm = max(0,j-1) 
                regularizer +=  (y[iM,j] - y[i,j])**2 + (y[i,jM] - y[i,j])**2 

    objective = cp.norm((p.T @ p)@x  - p.T @ proj) + 0.001*regularizer

    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model

def modeltv(m, p, proj, shape):
    x = cp.Variable(shape)
    constraints = []

    regularizer = 0
    u, s, vh = np.linalg.svd(p)
    
    D = np.eye((shape[0]))
    D = D - np.eye((shape[0]), k=1)

    objective = cp.norm((p.T @ p )@cp.reshape(x, (shape[0]*shape[1],))  -   p.T @ proj ) + 1e-4*(cp.tv(x)) 
    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model

def modelD(m, p, proj, shape):
    x = cp.Variable(shape)
    M = np.zeros(shape)
    for i in range(shape[0]):
        M[i,shape[0]-i-1] = 1
    constraints = []

    
    
    D = np.eye((shape[0]))
    D = D - np.eye((shape[0]), k=1)

    objective = cp.norm((p.T @ p )@cp.reshape(x, (shape[0]*shape[1],))  -   p.T @ proj ) + 1e-4*(cp.tv(x) + cp.tv(M@x@M))  
    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model



def modelD2(m, p, proj, shape):
    x = cp.Variable(shape)
    constraints = []

    M = np.zeros(shape)
    for i in range(shape[0]):
        M[i,shape[0]-i-1] = 1
    
    D = -2*np.eye((shape[0])) + np.eye((shape[0]), k=1) +  np.eye((shape[0]), k=-1)
    D[0,:] = np.zeros(shape[0])
    D[shape[0] -1,:] = np.zeros(shape[0])

    objective = cp.norm((p.T @ p )@cp.reshape(x, (shape[0]*shape[1],))  -   p.T @ proj ) + 1e-4*(cp.norm( D@x + x@D.T, 'fro') + cp.norm( D@M@x@M + M@x@M@D.T, 'fro'))

    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model


class Model:
    def __init__(self, x, objective, constraints):
        self.objectives = objective
        self.x = x
        self.constraints = constraints
        self.problem = cp.Problem( cp.Minimize( objective ) , constraints ) 

    def solve( self , solver = CPLEX, verbose=False ):
        return self.problem.solve( solver = solver, verbose=verbose)

