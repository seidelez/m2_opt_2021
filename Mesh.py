from matplotlib import pyplot
import matplotlib.tri as mtri
import numpy as np

class Mesh:
    def __init__( self, positions = np.empty( [ 0, 2 ] ), triangles = np.empty( [ 0, 3 ] ) ):
        self.positions = positions
        self.triangles = triangles

    @property
    def nb_triangles( self ):
        return self.triangles.shape[ 0 ]

    @property
    def nb_nodes( self ):
        return self.positions.shape[ 0 ]

    def draw( self ):
        _, ax = pyplot.subplots()
        ax.set_aspect( 'equal' )
        ax.triplot( self.positions[ :, 0 ], self.positions[ :, 1 ], triangles = self.triangles )

        pyplot.show()

    def draw_with_elem_field( self, field, img_name = "" ):
        _, ax = pyplot.subplots()
        ax.set_aspect( 'equal' )
        ax.tripcolor( self.positions[ :, 0 ], self.positions[ :, 1 ], triangles = self.triangles, facecolors = field )

        if img_name:
            pyplot.savefig( img_name )
        else:
            pyplot.show()

    def draw_with_nodal_field( self, field, img_name = "" ):
        _, ax = pyplot.subplots()
        ax.set_aspect( 'equal' )

        # ax.tripcolor( , facecolors = field )
        triang = mtri.Triangulation( self.positions[ :, 0 ], self.positions[ :, 1 ], triangles = self.triangles )
        ax.tricontourf( triang, field )
        # ax.triplot(triang, 'ko-')

        if img_name:
            pyplot.savefig( img_name )
        else:
            pyplot.show()

    def elem_field_from_img( self, img, beg_p, end_p, use_inversion = False ):
        from CutXY import CutXY

        cxy = CutXY( self, beg_p[ 0 ], end_p[ 0 ], img.shape[ 0 ], beg_p[ 1 ], end_p[ 1 ], img.shape[ 1 ] )
        M = cxy.elem_integration_matrix()

        if use_inversion:
            P = M.T @ M
            P += 1e-6 * np.max( np.diag( P ) ) * np.eye( P.shape[ 0 ] )
            return np.linalg.solve( P, M.T @ img.ravel() )

        D = np.sum( M, axis = 0 )
        return ( M.T @ img.ravel() ) / ( D + ( D == 0 ) )

    def nodal_field_from_img( self, img, beg_p, end_p ):
        from CutXY import CutXY

        cxy = CutXY( self, beg_p[ 0 ], end_p[ 0 ], img.shape[ 0 ], beg_p[ 1 ], end_p[ 1 ], img.shape[ 1 ] )
        M = cxy.nodal_integration_matrix()

        D = np.sum( M, axis = 0 )
        return ( M.T @ img.ravel() ) / ( D + ( D == 0 ) )

    def rotated( self, center, angle ):
        R = np.array( [
            [ + np.cos( angle ), + np.sin( angle ) ],
            [ - np.sin( angle ), + np.cos( angle ) ]
        ] )
        p = center + ( self.positions - center ) @ R
        return Mesh( p, self.triangles )

    def rect( p0, p1, di ):
        # Utiliser meshgrid ?
        p = []
        for y in range( di[ 1 ] ):
            for x in range( di[ 0 ] ):
                p.append( [
                    p0[ 0 ] + x * ( p1[ 0 ] - p0[ 0 ] ) / ( di[ 0 ] - 1 ),
                    p0[ 1 ] + y * ( p1[ 1 ] - p0[ 1 ] ) / ( di[ 1 ] - 1 ),
                ] )

        t = []
        for y in range( di[ 1 ] - 1 ):
            for x in range( di[ 0 ] - 1 ):
                t.append( [
                    ( x + 0 ) + ( y + 0 ) * di[ 0 ],
                    ( x + 1 ) + ( y + 0 ) * di[ 0 ],
                    ( x + 0 ) + ( y + 1 ) * di[ 0 ],
                ] )
                t.append( [
                    ( x + 1 ) + ( y + 0 ) * di[ 0 ],
                    ( x + 1 ) + ( y + 1 ) * di[ 0 ],
                    ( x + 0 ) + ( y + 1 ) * di[ 0 ],
                ] )

        return Mesh( np.array( p ), np.array( t ) )

    def disc( center, radius, step ):
        import pygmsh

        with pygmsh.geo.Geometry() as geom:
            geom.add_circle( center, radius, mesh_size = step )
            mesh = geom.generate_mesh()

        triangles = None
        for c in mesh.cells:
            if c.type.startswith( "triangle" ):
                triangles = c.data

        res = Mesh( mesh.points[ :, 0:2 ] * 1.0, triangles )
        res.remove_unused_nodes()
        return res

    def remove_unused_nodes( self ):
        used = np.zeros( [ self.nb_nodes ], dtype = bool )
        used[ np.ravel( self.triangles ) ] = True
        idx = np.cumsum( used ) - 1

        self.positions = self.positions[ used, : ] * 1.0
        self.triangles = idx[ self.triangles ]

    def grad_matrices( self ):
        """ 
           return ( mat_x, max_y ) where mat_i * nodal_value return [ grad_i ]_e giving a value for each element e
        """
        res_x = np.zeros( [ self.nb_triangles, self.nb_nodes ] )
        res_y = np.zeros( [ self.nb_triangles, self.nb_nodes ] )
        for ( num_tr, tr ) in enumerate( self.triangles ):
            x1 = self.positions[ tr[ 1 ], 0 ] - self.positions[ tr[ 0 ], 0 ]
            x2 = self.positions[ tr[ 2 ], 0 ] - self.positions[ tr[ 0 ], 0 ]
            y1 = self.positions[ tr[ 1 ], 1 ] - self.positions[ tr[ 0 ], 1 ]
            y2 = self.positions[ tr[ 2 ], 1 ] - self.positions[ tr[ 0 ], 1 ]
            dt = x1 * y2 - x2 * y1

            res_x[ num_tr, tr[ 0 ] ] = ( y1 - y2 ) / dt
            res_x[ num_tr, tr[ 1 ] ] = + y2 / dt
            res_x[ num_tr, tr[ 2 ] ] = - y1 / dt

            res_y[ num_tr, tr[ 0 ] ] = ( x1 - x2 ) / dt
            res_y[ num_tr, tr[ 1 ] ] = + x2 / dt
            res_y[ num_tr, tr[ 2 ] ] = - x1 / dt


        return ( res_x, res_y )

