import tensorflow as tf
from sentenceencoder    import SentenceEncoder
from networkcomponents  import *

class CNNSentenceEncoder( SentenceEncoder ):
    def __init__( self, config_encoder, tokenizer ):
        super( CNNSentenceEncoder, self ).__init__( config_encoder, tokenizer )

        self._vec_cnns  = []
        for kernel_size in config_encoder[ "kernel_size" ]:
            self._vec_cnns.append( tf.keras.layers.Conv1D( config_encoder[ "filter_size" ], kernel_size, activation = "relu", padding = "SAME" ) )

        self._result_dim    = len( config_encoder[ "kernel_size" ] ) * config_encoder[ "filter_size" ]            
        self._dense         = tf.keras.layers.Dense( self._result_dim )
        print ( "Applying CNN Encoder." )            

    def get_result_dim( self ):
        return self._result_dim

    # token_id: BS X T.
    # token_mask: BS>
    def call( self, token_id, token_mask ):
        token_embed = self._embed_token( token_id )     # Masked.

        # 1. CNNs
        vec_cnn_result  = []
        for cnn in self._vec_cnns:
            vec_cnn_result.append( cnn( token_embed ) ) 

        cnn_result  = tf.concat( vec_cnn_result, -1 )   # BS X T X D.
        
        # 2. Max-Pooling.
        cnn_result  = apply_mask( cnn_result, token_mask, float( "-inf" ) )
        sen_encoded = tf.reduce_max( cnn_result, 1 )
        sen_encoded = tf.expand_dims( sen_encoded, 1 )  # BS X 1 X D.

        # 3. Dense.
        sen_encoded = self._dense( sen_encoded )


        return sen_encoded

        


