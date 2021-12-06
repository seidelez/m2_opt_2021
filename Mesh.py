from matplotlib import pyplot, use
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

    def nodal_sampling( self, point_list ):
        """
           Given a list of positions (shape=[nb_points,2]), get an array of triangle indices and interpolation variables
        """

        # limits
        min_x = np.min( self.positions[ :, 0 ] )
        min_y = np.min( self.positions[ :, 1 ] )
        max_x = np.max( self.positions[ :, 0 ] )
        max_y = np.max( self.positions[ :, 1 ] )

        # grid size and init
        csize = 0.25 * ( ( max_x - min_x ) * ( max_y - min_y ) / self.nb_triangles ) ** 0.5
        nx = int( np.ceil( ( max_x - min_x ) / csize ) )
        ny = int( np.ceil( ( max_y - min_y ) / csize ) )

        grid = np.empty( [ nx, ny ], dtype = object )
        for x in range( nx ):
            for y in range( ny ):
                grid[ x, y ] = []

        # add the triangles in box lists
        for num_tr, tr in enumerate( self.triangles ):
            l_pos = self.positions[ tr, : ]
            lin_x = max( 0 , int( np.floor( ( np.min( l_pos[ :, 0 ] ) - min_x ) * nx / ( max_x - min_x ) ) ) )
            lin_y = max( 0 , int( np.floor( ( np.min( l_pos[ :, 1 ] ) - min_y ) * ny / ( max_y - min_y ) ) ) )
            lax_x = min( nx, int( np.ceil ( ( np.max( l_pos[ :, 0 ] ) - min_x ) * nx / ( max_x - min_x ) ) ) )
            lax_y = min( ny, int( np.ceil ( ( np.max( l_pos[ :, 1 ] ) - min_y ) * ny / ( max_y - min_y ) ) ) )
            for x in range( lin_x, lax_x ):
                for y in range( lin_y, lax_y ):
                    grid[ x, y ].append( num_tr )

        #
        triangles = np.full( [ point_list.shape[ 0 ] ], -1 )
        var_interps = np.full( [ point_list.shape[ 0 ], 2 ], np.nan )
        for ( num_point, point ) in enumerate( point_list ):
            # pos of the point in the grid
            grid_pos_x = min( nx - 1, max( 0, int( np.floor( ( point[ 0 ] - min_x ) * nx / ( max_x - min_x ) ) ) ) )
            grid_pos_y = min( ny - 1, max( 0, int( np.floor( ( point[ 1 ] - min_y ) * ny / ( max_y - min_y ) ) ) ) )

            # find the enclosing triangle
            for num_triangle in grid[ grid_pos_x, grid_pos_y ]:
                pt = self.positions[ self.triangles[ num_triangle ], : ]
                ma = np.array( [ [ pt[ 1, 0 ] - pt[ 0, 0 ], pt[ 2, 0 ] - pt[ 0, 0 ] ], [ pt[ 1, 1 ] - pt[ 0, 1 ], pt[ 2, 1 ] - pt[ 0, 1 ] ] ] )
                ve = np.array( [ point[ 0 ] - pt[ 0, 0 ], point[ 1 ] - pt[ 0, 1 ] ] )
                vi = np.linalg.solve( ma, ve )
                ep = 1e-6

                if vi[ 0 ] > - ep and vi[ 1 ] > - ep and 1 - vi[ 0 ] - vi[ 1 ] > - ep:
                    triangles[ num_point ] = num_triangle
                    var_interps[ num_point, : ] = vi
                    break

        return ( triangles, var_interps )

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
                    ( x + 1 ) + ( y + 1 ) * di[ 0 ],
                    ( x + 1 ) + ( y + 0 ) * di[ 0 ],
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

        return Mesh( mesh.points[ :, 0:2 ] * 1.0, triangles )