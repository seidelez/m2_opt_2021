from matplotlib import pyplot
from numpy.core.numeric import cross
from Mesh import Mesh
import numpy as np
import itertools

x = np.linspace( 0, 1, 5 )
y = np.linspace( 0, 1, 5 )
p = np.array( [ [ x, y ] for ( x, y ) in itertools.product( x, y ) ] )
m = Mesh.rect( [ 0, 0 ], [ 1, 1 ], [ 2, 2 ] )
( nt, vi ) = m.nodal_sampling( p )

for d in range( 2 ):
    r = m.positions[ m.triangles[ nt, 0 ], d ] * ( 1 - vi[ :, 0 ] - vi[ :, 1 ] ) + \
        m.positions[ m.triangles[ nt, 1 ], d ] * vi[ :, 0 ] + \
        m.positions[ m.triangles[ nt, 2 ], d ] * vi[ :, 1 ]
    print( np.max( np.abs( r - p[ :, d ] ) ) )

# pyplot.imshow( np.swapaxes( vi[ :, 1 ].reshape( [ x.size, -1 ] ), 0, 1 ), origin = 'lower' )
# pyplot.show()
