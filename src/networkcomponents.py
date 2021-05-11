import tensorflow as tf

# target: BS X L X D
# mask: BS. Mask on L.
def apply_mask( target, mask, mask_seed ):
    if mask.shape.ndims == 1:
        mask_key    = tf.expand_dims( tf.sequence_mask( mask, tf.shape( target )[1] ), -1 )   # BS X L X 1
    elif mask.shape.ndims == 2:
        mask_key    = tf.expand_dims( mask_key, -1 )
    else:
        mask_key    = mask

    if target.get_shape().as_list()[-1] != None:
        mask_key    = tf.tile( mask_key, [ 1, 1, target.get_shape().as_list()[-1] ] )
    else:        
        mask_key    = tf.tile( mask_key, [ 1, 1, tf.shape( target )[-1] ] )
    return tf.where( mask_key, target, mask_seed * tf.ones_like( target ) )
