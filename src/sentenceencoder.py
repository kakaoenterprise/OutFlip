import tensorflow           as tf
from tensorflow.keras       import Model


def CreateEncoder( config_encoder, tokenizer ):
    if config_encoder[ "type" ] == "bilstm":
        from bilstmsentenceencoder  import BiLSTMSentenceEncoder
        return BiLSTMSentenceEncoder( config_encoder, tokenizer )
    elif config_encoder[ "type" ] == "cnn":
        from cnnsentenceencoder  import CNNSentenceEncoder
        return CNNSentenceEncoder( config_encoder, tokenizer )

    print ( "Unknown Encoder Type: [%s]" % config_encoder[ "type" ] )

class SentenceEncoder( Model ):
    def __init__( self, config, tokenizer ):
        super( SentenceEncoder, self ).__init__()
        self._config    = config

        # 0 is reserved for padding.
        self._embed_token   = tf.keras.layers.Embedding( tokenizer._glove_vec.shape[0], tokenizer._glove_vec.shape[1], \
                                    embeddings_initializer = tf.keras.initializers.Constant( tokenizer._glove_vec ), trainable = False, mask_zero = True ) 

    def call( self, token_id, token_mask ):
        pass
