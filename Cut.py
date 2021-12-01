from matplotlib import pyplot
import numpy as np

def area( poly ):
    x = poly[ :, 0 ]
    y = poly[ :, 1 ]
    return 0.5 * np.abs( np.dot( x, np.roll( y, 1 ) ) - np.dot( y, np.roll( x, 1 ) ) )

class Cut:
    def __init__( self, mesh, beg_cut, end_cut, nb_cuts ):
        self.cut_interps = {} # nb points dans polygone => variables d'interpolation pour chaque point de chaque coupe
        self.cut_trinums = {} # nb points dans polygone => num du triangle dans `mesh` pour chaque coupe avec ce nombre de points
        self.cut_raynums = {} # nb points dans polygone => num du triangle dans `mesh` pour chaque coupe avec ce nombre de points
        self.cut_pos = {} # nb points dans polygone => positions des points

        self.inc_cut = ( end_cut - beg_cut ) / nb_cuts
        self.beg_cut = beg_cut
        self.end_cut = end_cut
        self.nb_cuts = nb_cuts
        self.mesh = mesh

        self.make_cuts()

    def make_cuts( self ):
        # liste de coupe à concaténer pour obtenir les self.cut_...
        cut_trinums = []
        cut_raynums = []
        cut_interps = []
        cut_pos  = []

        # nombre de rayons
        p_tri = self.mesh.positions[ self.mesh.triangles ]
        i_tri = np.floor( ( p_tri[ :, :, 0 ] - self.beg_cut ) / self.inc_cut ).astype( int ) # position entière en x
        
        max_x = np.minimum( np.max( i_tri, axis = 1 ), self.nb_cuts - 1 )
        min_x = np.maximum( np.min( i_tri, axis = 1 ), 0 )
        n_cut = max_x - min_x + 1

        # triangles traversés par 1 rayon, puis par 2, etc...
        for n_tra in range( 1, max( n_cut ) + 1 ):
            n_tri_sel = np.arange( self.mesh.triangles.shape[ 0 ] )[ n_cut == n_tra ]
            p_tri_sel = self.mesh.positions[ self.mesh.triangles[ n_cut == n_tra ] ]
            r_tri_sel = min_x[ n_cut == n_tra ]
            i_tri_sel = i_tri[ n_cut == n_tra ]
            min_x_sel = min_x[ n_cut == n_tra ]
            if i_tri_sel.size == 0:
                continue

            # pour chaque rayon qui traverse le triangle
            for o_ray in range( n_tra ):
                # numéro de cas
                c_rel = ( i_tri_sel >= np.expand_dims( min_x_sel, axis = 1 ) + o_ray + 0 ).astype( int ) \
                      + ( i_tri_sel >= np.expand_dims( min_x_sel, axis = 1 ) + o_ray + 1 ).astype( int )
                cases = c_rel[ :, 0 ] + 3 * c_rel[ :, 1 ] + 9 * c_rel[ :, 2 ]
                x0 = self.beg_cut + ( min_x_sel + o_ray ) * self.inc_cut
                x1 = x0 + self.inc_cut
                for cc in range( 27 ):
                    self.add_cut_cases( cc, cut_trinums, cut_raynums, cut_interps, cut_pos, r_tri_sel[ cases == cc ] + o_ray, n_tri_sel[ cases == cc ], p_tri_sel[ cases == cc ], x0[ cases == cc ], x1[ cases == cc ] )

        # 
        for i in range( 3, 6 ):
            f = list( filter( lambda x: x[ 3 ].shape[ 1 ] == i, zip( cut_trinums, cut_raynums, cut_interps, cut_pos ) ) )
            if len( f ):
                self.cut_trinums[ i ] = np.concatenate( list( map( lambda x: x[ 0 ], f ) ), axis = 0 )
                self.cut_raynums[ i ] = np.concatenate( list( map( lambda x: x[ 1 ], f ) ), axis = 0 )
                self.cut_interps[ i ] = np.concatenate( list( map( lambda x: x[ 2 ], f ) ), axis = 0 )
                self.cut_pos[ i ] = np.concatenate( list( map( lambda x: x[ 3 ], f ) ), axis = 0 )

    def add_cut_cases( self, cc, cut_trinums, cut_raynums, cut_interps, cut_pos, r_tri, n_tri, p_tri, x0, x1 ):
        if p_tri.size == 0:
            return

        # position des points pour chaque droite
        p = [ ( cc // 1 ) % 3, ( cc // 3 ) % 3, ( cc // 9 ) % 3 ]
        l = []
        for i in range( 3 ):
            i0 = ( i + 0 ) % 3
            i1 = ( i + 1 ) % 3
            p0 = p[ i0 ]
            p1 = p[ i1 ]
            if p0 == 1:
                l.append( i0 )
            if p1 > p0:
                if p0 == 0:
                    l.append( ( i0, i1, x0 ) )
                if p1 == 2:
                    l.append( ( i0, i1, x1 ) )
            if p0 > p1:
                if p0 == 2:
                    l.append( ( i0, i1, x1 ) )
                if p1 == 0:
                    l.append( ( i0, i1, x0 ) )

        # creation des points
        res_p = np.empty( [ p_tri.shape[ 0 ], len( l ), 2 ] )
        res_i = np.empty( [ p_tri.shape[ 0 ], len( l ), 2 ] )
        i_tri = np.array( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ] )
        for ( i, interp ) in enumerate( l ):
            if type( interp ) == int:
                res_p[ :, i, : ] = p_tri[ :, interp, : ]
                res_i[ :, i, : ] = i_tri[ interp, : ]
            else:
                s = np.expand_dims( ( interp[ 2 ] - p_tri[ :, interp[ 0 ], 0 ] ) / ( p_tri[ :, interp[ 1 ], 0 ] - p_tri[ :, interp[ 0 ], 0 ] ), axis = 1 )
                res_p[ :, i, : ] = p_tri[ :, interp[ 0 ], : ] + s * ( p_tri[ :, interp[ 1 ], : ] - p_tri[ :, interp[ 0 ], : ] )
                res_i[ :, i, : ] = i_tri[ interp[ 0 ], : ] + s * ( i_tri[ interp[ 1 ], : ] - i_tri[ interp[ 0 ], : ] )
        cut_trinums.append( n_tri )
        cut_raynums.append( r_tri )
        cut_interps.append( res_i )
        cut_pos.append( res_p )
        
    def draw( self ):
        for n in self.cut_pos.keys():
            p = self.cut_pos[ n ]
            l = np.empty( p.shape + np.array( [ 0, 1, 0 ] ) )
            l[ :, 0:n, : ] = p
            l[ :, n, : ] = p[ :, 0, : ]
            for c in l:
                pyplot.plot( c[ :, 0 ], c[ :, 1 ] )
        pyplot.show()

    def proj_mat( self, nodal = False ):
        res = np.zeros( [ self.nb_cuts, self.mesh.triangles.shape[ 0 ] ] )
        for npo in self.cut_pos.keys():
            # TODO: en vectoriel
            for ( poly, num, ray ) in zip( self.cut_pos[ npo ], self.cut_trinums[ npo ], self.cut_raynums[ npo ] ):
                res[ ray, num ] += area( poly )
        return res

    def as_Mesh( self ):
        """
          Return a triangular mesh with provenance info.

          Nodes are not connected

          Adds, `trinums`
        """
        trinums = []
        raynums = []
        p = []
        t = []
        o = 0
        for n in self.cut_pos.keys():
            c = self.cut_pos[ n ]
            for nt in range( 2, c.shape[ 1 ] ):
                trinums.append( self.cut_trinums[ n ] )
                raynums.append( self.cut_raynums[ n ] )

                q = 3 * c.shape[ 0 ]
                t.append( o + np.arange( q ).reshape( [ -1, 3 ] ) )

                pa = np.empty( [ 3 * c.shape[ 0 ], 2 ] )
                pa[ 0::3, : ] = c[ :, 0     , : ]
                pa[ 1::3, : ] = c[ :, nt - 1, : ]
                pa[ 2::3, : ] = c[ :, nt - 0, : ]
                p.append( pa )

                o += q

        from Mesh import Mesh
        res = Mesh( np.concatenate( p, axis = 0 ), np.concatenate( t, axis = 0 ) )
        res.trinums = np.concatenate( trinums, axis = 0 )
        res.raynums = np.concatenate( raynums, axis = 0 )
        return res
