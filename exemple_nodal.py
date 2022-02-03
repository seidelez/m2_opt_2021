from matplotlib import pyplot
import scipy
from proj import proj_rot
from Mesh import Mesh
import numpy as np
import Model
import math
import evaluation as eval
from scipy import optimize
np.set_printoptions(threshold=np.inf)

# maillage
s = 50
m = Mesh.disc( center = [ 0, 0 ], radius = 1, step = 2 / s )

# champ de référence
f = m.nodal_field_from_img(
    pyplot.imread( "ex.png" )[ :, :, 0 ],
    np.array( [ -1, -1 ] ),
    np.array( [ +1, +1 ] )
)

m = Mesh.rect( [ 0, 0 ], [ 1, 1 ], [ 10, 10 ] )
a = np.linspace( 0, np.pi, 10 )

# operateur de projection
#a = np.linspace( 0, np.pi, s, endpoint=False )
#p = proj_rot( m, np.array( [ 0, 0 ] ), a, -1, +1, s, nodal = True )

a = np.linspace( 0, np.pi, 10 )
p = proj_rot( m, np.array( [ 0.5, 0.5 ] ), a, 0, 1, 10, nodal = True)
ieco = np.zeros( m.nb_nodes)
ieco = ieco.flatten()
ieco[50:60] = np.ones(10)

ieco[55] = 0

set = [44,45,46,64, 65, 66]
for i in set:
    #ieco[i] = 1
    _ = 0

ieco = ieco.reshape((10,10))
#ieco[:,0] = np.ones(10)
#ieco[:,9] = np.ones(10)
#ieco[0,:] = np.ones(10)
#ieco[9,:] = np.ones(10)
ieco = ieco.reshape((100,))
proj = p @ ieco
shape = (10,10)

# # ajout de bruit (très peu en fait)
# proj += np.random.normal( 0, 0.0035, proj.shape )

reco = np.linalg.solve( p.T @ p + 1e-3 * np.eye( m.nb_nodes ), p.T @ proj )
reco1 = np.linalg.solve( p.T @ p , p.T @ proj )
x1, m1 = Model.modelD(m, p, proj, shape)
x2, m2 = Model.modelD2(m, p, proj, shape)
x3, m3 = Model.modeltv(m, p, proj, shape)
m1.solve()
m2.solve()
m3.solve()
rec1 = x1.value
rec2 = x2.value
rectv = x3.value


rec3 = optimize.minimize(eval.cost_eval, np.ones(100,), args=(p,proj))

m.draw_with_nodal_field( ieco )
m.draw_with_nodal_field( reco1 )
m.draw_with_nodal_field( reco )
m.draw_with_nodal_field( np.reshape(rec1.T, (100,)))
m.draw_with_nodal_field( np.reshape(rec2.T, (100,)))
m.draw_with_nodal_field( np.reshape(rectv.T, (100,)))
m.draw_with_nodal_field( rec3.x )

