import tensorflow as tf
from tensorflow.keras       import Model
from sentenceencoder        import CreateEncoder
#from networkcomponents      import *

class SentenceClassifier( Model ):
    def __init__( self, config, tokenizer, num_class, train_batch_num = 1 ):
        super( SentenceClassifier, self ).__init__()

        self._config        = config
        self._sen_encoder   = CreateEncoder( config[ "encoder" ], tokenizer )
        self._dense         = tf.keras.layers.Dense( num_class )
        self._num_class     = num_class

        lr_schedule         = tf.keras.optimizers.schedules.ExponentialDecay( config[ "learning" ][ "initial_lr" ], \
                                        decay_steps = train_batch_num * config[ "learning" ][ "decay_step" ], \
                                        decay_rate = config[ "learning" ][ "decay_rate" ], staircase = True )
        self._optimizer     = tf.keras.optimizers.Adam( learning_rate = lr_schedule )

    @tf.function( input_signature = ( \
        tf.TensorSpec( shape = [ None, None ], dtype = tf.int32, name = "token_id" ), \
        tf.TensorSpec( shape = [ None ], dtype = tf.int32, name = "token_mask" ) ) )
    def call( self, token_id, token_mask ):
        sen_encoded     = self._sen_encoder( token_id, token_mask )
        sen_encoded     = tf.squeeze( sen_encoded, 1 )                  # BS X D.
        
        logits_class    = self._dense( sen_encoded )
        return logits_class, sen_encoded

    @tf.function( input_signature = ( \
        tf.TensorSpec( shape = [ None, None ], dtype = tf.int32, name = "token_id" ), \
        tf.TensorSpec( shape = [ None ], dtype = tf.int32, name = "token_mask" ), \
        tf.TensorSpec( shape = [ None ], dtype = tf.int32, name = "class" ) ) )
    def train( self, token_id, token_mask, class_labels ):
        with tf.GradientTape() as tape:
            logits_class, sen_encoded = self.call( token_id, token_mask ) # BS X C, BS X D.
            loss    = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels = class_labels, logits = logits_class ) )

        gradients = tape.gradient( loss, self.trainable_variables )
        self._optimizer.apply_gradients( [ ( grad, var ) for ( grad, var ) in zip( gradients, self.trainable_variables ) if grad is not None ] )

        return loss

    @tf.function( input_signature = ( \
        tf.TensorSpec( shape = [ None, None ], dtype = tf.int32, name = "token_id" ), \
        tf.TensorSpec( shape = [ None ], dtype = tf.int32, name = "token_mask" ), \
        tf.TensorSpec( shape = [ None ], dtype = tf.int32, name = "class" ), \
        tf.TensorSpec( shape = [ None ], dtype = tf.int32, name = "mi_tok_loc" ) ) )
    def get_outflip_gradient( self, token_id, token_mask, class_labels, mi_tok_loc ):
        with tf.GradientTape() as tape:
            logits_class, sen_encoded = self.call( token_id, token_mask ) # BS X C, BS X D.
            loss        = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels = class_labels, logits = logits_class ) )

        # Gradient for Embedding token.
        grad_embed  = tape.gradient( loss, [ self._sen_encoder._embed_token.embeddings ] )[0]   # |V| X D.
        grad_embed  = tf.gather_nd( grad_embed, tf.expand_dims( token_id, -1 ) )            # Gradient for each token. BS X T X D.
        grad_embed  = tf.gather_nd( grad_embed, tf.expand_dims( mi_tok_loc, -1 ), batch_dims = 1 )  # Gradient for MI token. BS X D.
        src_embeds  = self._sen_encoder._embed_token( token_id )                                    # BS X T X D. Token embeddings.
        src_embeds  = tf.gather_nd( src_embeds, tf.expand_dims( mi_tok_loc, -1 ), batch_dims = 1 )  # BS X D. MI token embed.
        embedding_matrix    = self._sen_encoder._embed_token.embeddings                             # |V| X D. Embedding matrix.

        # Gradient for the destination * source Token.
        grad_dst    = tf.matmul( grad_embed, embedding_matrix, transpose_b = True ) # BS X |V|
        grad_src    = tf.einsum( "bd,bd->b", grad_embed, src_embeds )               # BS

        return grad_dst - tf.expand_dims( grad_src, -1 )        # BS X |V|.

