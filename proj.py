from Cut import Cut
import numpy as np

def proj_rot( mesh, center, angles, beg_x, end_x, len_x, nodal = False ):
    res = []
    for angle in angles:
        rot_mesh = mesh.rotated( center, angle ) 
        cut_mesh = Cut( rot_mesh, beg_x, end_x, len_x )
        res.append( cut_mesh.proj_mat( nodal = nodal ) )
    return np.concatenate( res, axis = 0 )
