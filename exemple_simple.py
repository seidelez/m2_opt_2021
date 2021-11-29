from matplotlib import pyplot
import numpy as np

from proj import proj_rot
from Mesh import Mesh
from Cut import Cut
    
# maillage, rotations et matrice de projection
m = Mesh.rect( [ 0, 0 ], [ 1, 1 ], [ 10, 10 ] )
a = np.linspace( 0, np.pi, 10 )
p = proj_rot( m, np.array( [ 0.5, 0.5 ] ), a, 0, 1, 10 )

# image avec un seul triangle à 1 + projection
ieco = np.zeros( m.nb_triangles )
ieco[ m.nb_triangles // 2 ] = 1
proj = p @ ieco

# pyplot.imshow( proj.reshape( [ a.size, -1 ] ) )
# pyplot.show()

# reconstruction avec Tikhonov de base
reco = np.linalg.solve( p.T @ p + 1e-3 * np.eye( m.nb_triangles ), p.T @ proj )
m.draw_with_elem_field( reco )
