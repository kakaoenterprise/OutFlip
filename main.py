import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append( os.path.join( BASE_DIR, "./src" ) )

import random
import argparse
import tensorflow       as tf
from tokenizer          import Tokenizer
from sentenceclassifier import SentenceClassifier
from testor             import Testor

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

if __name__ == "__main__":
    parser  = argparse.ArgumentParser( description = "Experiments on Outline Sentence Classification" )
    parser.add_argument( "--config", type = str, help = "Configuration File.", required = True )
    parser.add_argument( "--store", type = str, help = "Checkpoint storage directory", required = True )
    parser.add_argument( "--train", type = str, help = "Location of the training data", required = True )
    parser.add_argument( "--dev", type = str, help = "Location of the development data", required = True )
    parser.add_argument( "--test", type = str, help = "Location of the test data", default = None )
    parser.add_argument( "--attack_num", type = int, help = "# of Attack", required = True )


    args    = parser.parse_args()
    with open( args.config, "r" ) as fin:
        config = json.load( fin )

    tokenizer   = Tokenizer( config[ "glove" ][ "file" ], config[ "glove" ][ "pickle" ], config[ "glove" ][ "lower" ] )

    if not os.path.exists( args.store ):
        os.mkdir( args.store )

    testor              = Testor( config, tokenizer )
    vec_iter_result    =  testor.do_hotflip_iteration( args.store, args.train, args.dev, args.test, args.attack_num )

    print ( vec_iter_result )
        
    print ( "=============== SUMMARY ===============" )
    for attack_idx in range( args.attack_num + 1 ):
        print ( "ATTACK %d: %.3f" % ( attack_idx, vec_iter_result[ attack_idx ] ) )
