from matplotlib import pyplot
from proj import proj_rot
from Mesh import Mesh
import numpy as np

# maillage
s = 50
m = Mesh.disc( center = [ 0, 0 ], radius = 1, step = 2 / s )

# champ de référence
f = m.nodal_field_from_img(
    pyplot.imread( "ex.png" )[ :, :, 0 ],
    np.array( [ -1, -1 ] ),
    np.array( [ +1, +1 ] )
)
# m.draw_with_nodal_field( f )

# operateur de projection
a = np.linspace( 0, np.pi, s, endpoint=False )
p = proj_rot( m, np.array( [ 0, 0 ] ), a, -1, +1, s, nodal = True )
proj = p @ f

# # ajout de bruit (très peu en fait)
# proj += np.random.normal( 0, 0.0035, proj.shape )

reco = np.linalg.solve( p.T @ p + 1e-4 * np.eye( m.nb_nodes ), p.T @ proj )
m.draw_with_nodal_field( reco )
