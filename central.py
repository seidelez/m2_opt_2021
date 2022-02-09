from Mesh import Mesh
import numpy as np
import Model
import losses 
from scipy.optimize import minimize 
from matplotlib import pyplot
from proj import proj_rot

s = 5
m = Mesh.disc( center = [ 0, 0 ], radius = 1, step = 2 / s )
( Gx, Gy ) = m.grad_matrices()

# test avec champ connu
print( Gx.shape )
print( Gy.shape )
print( m.nb_nodes )
print( m.nb_triangles )
#print( Gx.T @ np.ones( m.nb_triangles )  )
#print( Gx @ m.positions[ :, 0 ] )
#print( Gy @ m.positions[ :, 0 ] )
#print( Gx @ m.positions[ :, 1 ] )
#print( Gy @ m.positions[ :, 1 ] )

# exemple avec système matriciel
P = np.zeros( [ m.nb_nodes, m.nb_nodes ] )
P[ 0, 0 ] += 1
P += Gx.T @ Gx
P += Gy.T @ Gy
f = np.linalg.solve( P, Gx.T @ np.ones( m.nb_triangles ) )
#m.draw_with_nodal_field( f )

f = m.nodal_field_from_img(
    pyplot.imread( "ex.png" )[ :, :, 0 ],
    np.array( [ -1, -1 ] ),
    np.array( [ +1, +1 ] )
)
m.draw_with_nodal_field( f )

# operateur de projection
nb_angles = 2
a = np.linspace( 0, np.pi, nb_angles * s, endpoint = False )
p = proj_rot( m, np.array( [ 0, 0 ] ), a, -1, +1, 10 * s, nodal = True )
proj = p @ f

rec0 = np.linalg.solve( p.T @ p , p.T @ proj )
reco = np.linalg.solve( p.T @ p + 1e-6 * np.eye( m.nb_nodes ), p.T @ proj )
#m.draw_with_nodal_field( rec0 )
#m.draw_with_nodal_field( reco )
#x1, m1 = Model.model_curv( m, p, proj, Gx, Gy)
#x2, m2 = Model.model_curv( m, p, proj, Gx, Gy)
x0 = reco
print(x0.shape)
rec1 =  minimize(losses.final_total, x0, args=(Gx, Gy, p, proj, losses.mean_curv))
print(rec1)

#m1.solve()
#m2.solve()

#rec1 = x1.value
#rec2 = x2.value


m.draw_with_nodal_field( rec1 )
m.draw_with_nodal_field( rec2 )