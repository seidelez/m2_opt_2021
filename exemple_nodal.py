from matplotlib import pyplot
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
m.draw_with_elem_field( f )

# # operateur de projection
# a = np.linspace( 0, np.pi, s, endpoint=False )
# p = proj_rot( m, np.array( [ 0, 0 ] ), a, -1, +1, s )
# proj = p @ f

# # ajout de bruit (très peu en fait)
# proj += np.random.normal( 0, 0.0035, proj.shape )

# # reconstruction en forçant vers 0 ou 1. Pas spécialement rigoureux, résultat pas génial.
# lamb = 1e-5
# gamm = 1e-3
# reco = np.full( [ m.nb_triangles ], 0.5 )
# for i in range( 4 ):
#     print( i )
#     d = 0.49 ** ( 1 + i )
#     z = ( reco < 0.5 - d ) * 1.0
#     o = ( reco > 0.5 + d ) * 1.0
#     reco = np.linalg.solve( p.T @ p + np.diag( lamb + gamm * ( o + z ) ), p.T @ proj + 0.5 * lamb + gamm * o )
#     m.draw_with_elem_field( reco, "reco_{:03d}.png".format( i ) )
