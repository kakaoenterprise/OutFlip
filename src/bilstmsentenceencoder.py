from sentenceencoder    import SentenceEncoder
from networkcomponents  import *

class BiLSTMSentenceEncoder( SentenceEncoder ):
    def __init__( self, config_encoder, tokenizer ):
        super( BiLSTMSentenceEncoder, self ).__init__( config_encoder, tokenizer )

        self._vec_bilstm  = []
        for _ in range( config_encoder[ "bilstm_num" ] ):
            self._vec_bilstm.append( tf.keras.layers.Bidirectional( tf.keras.layers.LSTM( config_encoder[ "bilstm_dim" ], return_sequences = True ) ) )

        self._sa_1  = tf.keras.layers.Dense( config_encoder[ "selfatt_dim" ], activation = tf.tanh )
        self._sa_2  = tf.keras.layers.Dense( 1 )

    # token_id: BS X T.
    # token_mask: BS>
    def call( self, token_id, token_mask ):
        token_embed = self._embed_token( token_id )     # Masked.

        for bilstm in self._vec_bilstm:
            token_embed = bilstm( token_embed )     # What do you return?

        att = self._sa_2( self._sa_1( token_embed ) )   # BS X T X 1.
        att = apply_mask( tf.transpose( att, [ 0, 2, 1 ] ), token_mask, float( "-inf" ) )
        att = tf.nn.softmax( att, -1 )

        sen_encoded = tf.matmul( att, token_embed )   # BS X 1 X D.

        return sen_encoded

        


