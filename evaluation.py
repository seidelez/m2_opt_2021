from cvxpy.problems.objective import Objective
from cvxpy.settings import CPLEX
from matplotlib.pyplot import xcorr
from scipy import ndimage
from Mesh import Mesh
import numpy as np
import cvxpy as cp
import math

def R(x):
    return 0

def connected_comp(x, shape = (10,10)):
    X = np.reshape(x, shape)
    Xres = 0*X
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not Xres[i][j]:
                Xres = -round(X[i][j]) + np.ones(shape)

    labeled_array, num_features = ndimage.label(X)

    return num_features

def cost_eval(x, p, proj):
    res = np.linalg.norm(p.T @ p @ x - p.T @ proj) + 10*connected_comp(x) + R(x)
    return res