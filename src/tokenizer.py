import numpy as np

import nltk
import os
import sys
sys.path.append( "../../util" )
import pickle

from timemeasurer import TimeMeasurer

TOK_UNK     = "<unk>"
TOK_START   = "<s>"
TOK_END     = "</s>"

class Tokenizer:
    def __init__( self, glove_path, glove_pickle, is_uncased ):
        self._tokenizer = nltk.tokenize.word_tokenize

        self._glove_dic_rev     = None
        self._is_normalized     = False
        self._is_uncased        = is_uncased
        if os.path.exists( glove_pickle ):
            with open( glove_pickle, "rb" ) as fin:
                tm  = TimeMeasurer()
                tm.start( "pickle_load" )
                self._glove_vec, self._glove_dic    = pickle.load( fin )
                print ( "[%d] Vectors are loaded from pickle" % len( self._glove_vec ) )
                if TOK_START in self._glove_dic:
                    pass
                else:
                    print ( "START NOT FOUND. EXITING..." )
                    sys.exit(0)
                tm.end( "pickle_load" )
                tm.printall()
                return
    
        tm  = TimeMeasurer()
        tm.start( "read" )
        self._glove_dic     = dict()
        with open( glove_path, 'r') as f:
            n_vec   = 0
            for i, line in enumerate(f):
                n_vec   += 1
            hidden_dim = len(line.split(' ')) - 1
        self._glove_vec     = np.zeros( ( n_vec + 4, hidden_dim ), dtype = np.float32 )     # Element 0 is for padding.

        dup_idx = 1
        with open( glove_path, "r" ) as f:
            for i, line in enumerate(f):
                vecline = line.strip().split()
                if len( vecline ) != 301:
                    vecline = [ "" ] + vecline
                self._glove_vec[ i + 1 ]  = np.array( [ float(s) for s in vecline[1:] ] ) 

                if vecline[0] in self._glove_dic:
                    print ( "DUPLICATED: [%s]"  % vecline[0] )
                    vecline[0] += "_DUP%d" % dup_idx
                    print ( "CHANGED TO: [%s]" % vecline[0] )
                    dup_idx += 1
                self._glove_dic[ vecline[0] ]  = i + 1
        print ( "Glove Reading Complete. # VOCS: [%d]" % n_vec )
        tm.end( "read" )

        # Add <s> and </s> tokens.
        print ( "s token not found." )

        
        tm.start( "unk" )
        self._unk   = np.mean( self._glove_vec, axis = 0 )
        tm.end( "unk" )
        tm.printall()

        self._glove_vec[ n_vec + 1 ]    = np.random.normal( 0, 1, size = ( hidden_dim ) )
        self._glove_vec[ n_vec + 2 ]    = np.random.normal( 0, 1, size = ( hidden_dim ) )
        self._glove_vec[ n_vec + 3 ]    = self._unk
        self._glove_dic[ TOK_START ]    = n_vec + 1
        self._glove_dic[ TOK_END ]      = n_vec + 2
        self._glove_dic[ TOK_UNK ]      = n_vec + 3

        with open( glove_pickle, "wb" ) as fout:
            pickle.dump( ( self._glove_vec, self._glove_dic ), fout )

    def tokenize( self, sen ):
        vec_tokens  = self._tokenizer( sen )
        if self._is_uncased:
            vec_tokens  = [ t.lower() for t in vec_tokens ]
    
        vec_tokens  = [ TOK_START ] + [ t if t in self._glove_dic else TOK_UNK for t in vec_tokens ] + [ TOK_END ]
        return vec_tokens

    def convert_tokens_to_ids( self, vec_tokens ):
        return [ self._glove_dic[t] for t in vec_tokens ]

    def convert_id_to_token( self, token_id ):
        if self._glove_dic_rev == None:
            self._glove_dic_rev  = { v:k for k, v in self._glove_dic.items() }

        return self._glove_dic_rev[ token_id ]

    def _set_attack_matrix( self, vec_target_token_ids ):
        # 1. Calculate normalized glove.
        l2_norm = np.sqrt( np.sum( np.multiply( self._glove_vec, self._glove_vec ), axis = -1, keepdims = True ) ) + 1e-10
        norm_glove_vec  = self._glove_vec / l2_norm # |V| X D .

        # 2. Calculate Cosign similarity for target vectors.
        vec_target_tokens   = sorted( vec_target_token_ids )
        self._map_token_to_cosmat   = { v:k for k, v in enumerate( vec_target_tokens ) }
        self._cos_matrix    = np.matmul( norm_glove_vec[ vec_target_tokens ], norm_glove_vec.transpose() )  # T x |V|.
        

    def get_cosign_dist( self, target_tok_id, arr_attack_tok_id ):
        return self._cos_matrix[ self._map_token_to_cosmat[ target_tok_id ] ][ arr_attack_tok_id ]   # |arr_attack_tok_id|.

    def get_vocab_size( self ):
        return len( self._glove_dic )


if __name__ == "__main__":
    t   = Tokenizer( "../../glove/glove.300d.kor.txt", "../../glove/glove.300d.kor.withs.pickle", True )
    print ( t.convert_id_to_token( 283527 ) )
