from Mesh import Mesh
import numpy as np

m = Mesh.disc( center = [ 0, 0 ], radius = 1, step = 1 )
( Gx, Gy ) = m.grad_matrices()

# test avec champ connu
print( Gx @ m.positions[ :, 0 ] )
print( Gy @ m.positions[ :, 0 ] )

print( Gx @ m.positions[ :, 1 ] )
print( Gy @ m.positions[ :, 1 ] )

# exemple avec système matriciel
P = np.zeros( [ m.nb_nodes, m.nb_nodes ] )
P[ 0, 0 ] += 10
P += Gx.T @ Gx
P += Gy.T @ Gy
f = np.linalg.solve( P, Gx.T @ np.ones( m.nb_triangles ) )
m.draw_with_nodal_field( f )
