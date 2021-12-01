from matplotlib import pyplot
import numpy as np

class Mesh:
    def __init__( self, positions = np.empty( [ 0, 2 ] ), triangles = np.empty( [ 0, 3 ] ) ):
        self.positions = positions
        self.triangles = triangles

    def draw( self ):
        _, ax = pyplot.subplots()
        ax.set_aspect('equal')
        ax.triplot( self.positions[ :, 0 ], self.positions[ :, 1 ], triangles = self.triangles )

        pyplot.show()

    def draw_with_elem_field( self, field, img_name = "" ):
        _, ax = pyplot.subplots()
        ax.set_aspect('equal')
        ax.tripcolor( self.positions[ :, 0 ], self.positions[ :, 1 ], triangles = self.triangles, facecolors = field )

        if img_name:
            pyplot.savefig( img_name )
        else:
            pyplot.show()

    @property
    def nb_triangles( self ):
        return self.triangles.shape[ 0 ]

    def elem_field_from_img( self, img, beg_p, end_p ):
        xy = ( self.positions[ self.triangles[ :, 0 ] ] + self.positions[ self.triangles[ :, 1 ] ] + self.positions[ self.triangles[ :, 2 ] ] ) / 3
        xy = ( ( xy - beg_p ) * img.shape / ( end_p - beg_p ) ).astype( int )
        return img[ xy[ :, 0 ], xy[ :, 1 ] ]

    def rotated( self, center, angle ):
        R = np.array( [
            [   np.cos( angle ), np.sin( angle ) ],
            [ - np.sin( angle ), np.cos( angle ) ]
        ] )
        p = center + ( self.positions - center ) @ R
        return Mesh( p, self.triangles )

    def rect( p0, p1, di ):
        # Utiliser meshgrid ?
        p = []
        for x in range( di[ 0 ] ):
            for y in range( di[ 1 ] ):
                p.append( [
                    p0[ 0 ] + x * ( p1[ 0 ] - p0[ 0 ] ) / ( di[ 0 ] - 1 ),
                    p0[ 1 ] + y * ( p1[ 1 ] - p0[ 1 ] ) / ( di[ 1 ] - 1 ),
                ] )

        t = []
        for x in range( di[ 0 ] - 1 ):
            for y in range( di[ 1 ] - 1 ):
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

        return Mesh( mesh.points[ :, 0:2 ] * 1.0, triangles )