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

def modelLap(m, p, proj, Gx, Gy):
    x = cp.Variable(m.nb_nodes)
    constraints = []

    regularizer = cp.norm(Gx.T@Gx@x + Gy.T@Gy@x, 'fro')
    objective = cp.norm((p.T @ p )@x  -   p.T @ proj ) + 8e-8*regularizer  
    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model


def model_curv(m, p, proj, Gx, Gy):
    x = cp.Variable(m.nb_nodes)
    #normal_vector = cp.Variable(m.nb_nodes)
    constraints = []
    regularizer = 0

    grad = cp.hstack( [Gx@x, Gy@x] )

    aux2 = np.zeros(Gx.shape[0])
    aux = cp.Variable(m.nb_nodes)
    for i in range(Gx.shape[0]):
        constraints += [aux[i] == 1/cp.norm( cp.hstack([1, cp.norm(grad[i])] ) ) ]

    aux_x = cp.multiply(aux,Gx@x)
    aux_y = cp.multiply(aux,Gy@x)
    aux = cp.hstack( [Gx.T@aux_x + Gy.T@aux_y] ) 

    regularizer = cp.norm(aux)

    objective = cp.norm((p.T @ p )@x  -   p.T @ proj ) + 8e-8*regularizer 
    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model


def modelgrad(m, p, proj, Gx, Gy):
    x = cp.Variable(m.nb_nodes)
    constraints = []
    regularizer = 0
    
    aux = cp.hstack( [Gx@x, Gy@x] ) 

    for i in range(Gx.shape[0]):
        regularizer += cp.norm(aux[i])
    objective = cp.norm((p.T @ p )@x  -   p.T @ proj ) + 8e-8*regularizer 
    model = cp.Problem( cp.Minimize(objective) , constraints )
    return x, model

def model_euler(m, p, proj, Gx, Gy):
    x = cp.Variable(m.nb_nodes)
    constraints = []
    regularizer = 0

    Gstack = cp.hstack( [Gx@x, Gy@x] ) 
    for i in range(Gx.shape[0]):
        aux[i] = cp.norm(Gstack[i])

    aux2 = Gx.T@Gx@x + Gy.T@Gy@x


    for i in range(Gx.shape[0]):
        regularizer += cp.pnorm(aux[i],1)
    objective = cp.norm((p.T @ p )@x  -   p.T @ proj ) + 8e-8*regularizer 
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

