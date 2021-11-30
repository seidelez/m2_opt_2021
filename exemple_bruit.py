from matplotlib import pyplot
from proj import proj_rot
from Mesh import Mesh
import numpy as np

# maillage
m = Mesh.disc( center = [ 0, 0 ], radius = 1, step = 2 / 50 )

# champ de référence
f = m.elem_field_from_img(
    pyplot.imread( "ex.png" )[ :, :, 0 ],
    np.array( [ -1, -1 ] ),
    np.array( [ +1, +1 ] )
)

# operateur de projection
a = np.linspace( 0, np.pi, 50, endpoint=False )
p = proj_rot( m, np.array( [ 0, 0 ] ), a, -1, +1, 50 )
proj = p @ f

# ajout de bruit (très peu en fait)
proj += np.random.normal( 0, 0.0025, proj.shape )

# reconstruction avec Tikhonov de base
reco = np.linalg.solve( p.T @ p + 1e-3 * np.eye( m.nb_triangles ), p.T @ proj )
m.draw_with_elem_field( reco )
