from proj import proj_rot
from Mesh import Mesh
import Model
import numpy as np
import cvxpy as cvx
import math
np.set_printoptions(threshold=np.inf)

# maillage, rotations et matrice de projection
m = Mesh.rect( [ 0, 0 ], [ 1, 1 ], [ 10, 10 ] )
#print(m.positions)
a = np.linspace( 0, np.pi, 10 )
p = proj_rot( m, np.array( [ 0.5, 0.5 ] ), a, 0, 1, 10 )


for i in range(p.shape[0]):
    _ = 0
    #print(np.around(p.T @ p,decimals=2))

# image avec un seul triangle à 1 + projection
ieco = np.zeros( m.nb_triangles )
ieco[ m.nb_triangles // 2 ] = 1
ieco[ m.nb_triangles // 2 + 1] = 1
proj = p @ ieco

pix = math.sqrt(ieco.shape[0]*0.5)
ieco2 = ieco.reshape((18,9))
p_bar = p [:,:18]
res = 0
print(p.shape)

proj2 = p_bar @ ieco2
shape = (18,9)
print(proj.shape)
print(ieco.shape)


# reconstruction avec Tikhonov de base
reco = np.linalg.solve( p.T @ p + 1e-3 * np.eye( m.nb_triangles ), p.T @ proj )
rec1 = np.linalg.solve( p.T @ p + 1e-3 * np.eye( m.nb_triangles ), p.T @ proj )

#CVXPY Model
x1, m1 = Model.model1(m, p, proj)
m1.solve()
rec1 = x1.value

x2, m2 = Model.model3(m, p, proj, shape)
print(p.shape)
#  print(np.around(p,3))
m2.solve()
rec2 = x2.value

#print("rec ", rec2.shape)

# 0-1 original
#m.draw_with_elem_field( ieco )

# 0-1 solution
m.draw_with_elem_field( reco )
m.draw_with_elem_field( rec1 )
test = np.zeros(m.nb_nodes)
m.draw_with_nodal_field( test )
