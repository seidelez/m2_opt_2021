from Cut import Cut, area_polygon, integration_polygon
import numpy as np
    
class CutXY:
    def __init__( self, mesh, beg_cut_x, end_cut_x, nb_cuts_x, beg_cut_y, end_cut_y, nb_cuts_y ):
        self.beg_cut_x = beg_cut_x
        self.end_cut_x = end_cut_x
        self.nb_cuts_x = nb_cuts_x
        self.beg_cut_y = beg_cut_y
        self.end_cut_y = end_cut_y
        self.nb_cuts_y = nb_cuts_y
        self.mesh = mesh

        self.cut_x = Cut( self.mesh, beg_cut_x, end_cut_x, nb_cuts_x )
        for n in self.cut_x.cut_pos.keys():
            self.cut_x.cut_pos[ n ][ :, :, 0 ], self.cut_x.cut_pos[ n ][ :, :, 1 ] = self.cut_x.cut_pos[ n ][ :, :, 1 ] * 1, self.cut_x.cut_pos[ n ][ :, :, 0 ] * 1

        self.mint = self.cut_x.as_Mesh()

        self.cut_y = Cut( self.mint, beg_cut_y, end_cut_y, nb_cuts_y )
        for n in self.cut_y.cut_pos.keys():
            self.cut_y.cut_pos[ n ][ :, :, 0 ], self.cut_y.cut_pos[ n ][ :, :, 1 ] = self.cut_y.cut_pos[ n ][ :, :, 1 ] * 1, self.cut_y.cut_pos[ n ][ :, :, 0 ] * 1

    def elem_integration_matrix( self ):
        res = np.zeros( [ self.nb_cuts_x * self.nb_cuts_y, self.mesh.nb_triangles ] )
        for n in self.cut_y.cut_pos.keys():
            for num_cut in range( self.cut_y.cut_pos[ n ].shape[ 0 ] ):
                trinum_y = self.cut_y.cut_trinums[ n ][ num_cut ]
                raynum_y = self.cut_y.cut_raynums[ n ][ num_cut ]
                raynum_x = self.mint.raynums[ trinum_y ]
                trinum = self.mint.trinums[ trinum_y ]

                cut_pos = self.cut_y.cut_pos[ n ][ num_cut, :, : ]
                num_pix = raynum_y * self.nb_cuts_x + raynum_x
                res[ num_pix, trinum ] += area_polygon( cut_pos )
        area_pix  = ( self.end_cut_x - self.beg_cut_x ) / self.nb_cuts_x
        area_pix *= ( self.end_cut_y - self.beg_cut_y ) / self.nb_cuts_y
        return res / area_pix

    def nodal_integration_matrix( self ):
        res = np.zeros( [ self.nb_cuts_x * self.nb_cuts_y, self.mesh.nb_nodes ] )
        for n in self.cut_y.cut_pos.keys():
            for num_cut in range( self.cut_y.cut_pos[ n ].shape[ 0 ] ):
                interp_y = self.cut_y.cut_interps[ n ][ num_cut ] # vi[ 2 ] pour chaque noeud de la coupe finale
                trinum_y = self.cut_y.cut_trinums[ n ][ num_cut ]
                raynum_y = self.cut_y.cut_raynums[ n ][ num_cut ]
                triang_x = self.mint.triangles[ trinum_y, : ]
                raynum_x = self.mint.raynums[ trinum_y ]
                trinum_x = self.mint.trinums[ trinum_y ]
                triang_o = self.mesh.triangles[ trinum_x, : ]

                cut_pos = self.cut_y.cut_pos[ n ][ num_cut, :, : ]
                num_pix = raynum_y * self.nb_cuts_x + raynum_x

                inter = np.empty( [ interp_y.shape[ 0 ], 2 ] )
                for j in range( 2 ):
                    inter[ :, j ] = ( 1 - interp_y[ :, 0 ] - interp_y[ :, 1 ] ) * self.mint.interps[ triang_x[ 0 ], j ] + \
                                    interp_y[ :, 0 ]                            * self.mint.interps[ triang_x[ 1 ], j ] + \
                                    interp_y[ :, 1 ]                            * self.mint.interps[ triang_x[ 2 ], j ]

                res[ num_pix, triang_o[ 0 ] ] += integration_polygon( cut_pos, 1 - inter[ :, 0 ] - inter[ :, 1 ] )
                res[ num_pix, triang_o[ 1 ] ] += integration_polygon( cut_pos, inter[ :, 0 ] )
                res[ num_pix, triang_o[ 2 ] ] += integration_polygon( cut_pos, inter[ :, 1 ] )

        area_pix  = ( self.end_cut_x - self.beg_cut_x ) / self.nb_cuts_x
        area_pix *= ( self.end_cut_y - self.beg_cut_y ) / self.nb_cuts_y
        return res / area_pix
