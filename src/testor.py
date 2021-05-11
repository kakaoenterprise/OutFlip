import json
import os
import sys

import random
import argparse
import tensorflow       as tf
from tokenizer          import Tokenizer
from sentenceclassifier import SentenceClassifier
from timemeasurer       import TimeMeasurer
from padding_func       import *

import nltk
from nltk.corpus import wordnet

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

class Testor:
    def __init__( self, config, tokenizer ):
        self._config    = config
        self._tokenizer = tokenizer
        if config[ "lang" ] == "en":
            self._stopwords = nltk.corpus.stopwords.words( "english" ) + [ "<s>", "</s>", "<unk>", ".", ",", "!", "?" ]
        else:
            self._stopwords =[ u'은',u'는',u'이',u'가', u'을', u'수', u'시', u'제', u'ㄴ', u'ㄹ', u'ㄴ가요', u'나요', u'중', u'과', u'만', u'면', u'어', u'후', u'요', u'또는', u'및', u'인가', u'에서', u'의', u'자', u'로', u'에', u'를', u'ㄴ데', u'고', u'어요', u'여요', u'게', u'와', u'등에', u'다면', u'도', u'네요', u'ㄴ지', u'ㄴ지요', u'ㄴ다고', u'어야', u'는데요', u'으로', u'것', u'습니다', u'이러다', u'여야', u'ㄴ가', u'ㅂ니다', u'지', u'었', u'였', u'기', u'는데', u'려는데', u'는데도', u'거', u'라고', u'ㅁ', u'여', u'세요', u'은데', u'았', u'더니', u'오', u'는지', u'이나', u'란', u'에게', u'어떠', u'죠', u'야', u'니', u'나', u'다', u'무엇', u'누구', u'어디', u'언제', u'어떻게', u'뭐', u'누가', u'왜', u'어디서', u'얼마나', u'몇' ] +  [ "<s>", "</s>", "<unk>", ".", ",", "!", "?" ]

    def _read_data( self, train_fn, dev_fn, test_fn ):        
        vec_train, vec_dev, vec_test    = [], [], []
        with open( train_fn, "r" ) as fin:
            for line in fin:
                vec_line    = line.strip().split( "\t" )
                vec_train.append( [ vec_line[0], vec_line[1], self._tokenizer.convert_tokens_to_ids( self._tokenizer.tokenize( vec_line[1] ) ) ] )
        
        with open( dev_fn, "r" ) as fin:
            for line in fin:
                vec_line    = line.strip().split( "\t" )
                vec_dev.append( [ vec_line[0], vec_line[1], self._tokenizer.convert_tokens_to_ids( self._tokenizer.tokenize( vec_line[1] ) ) ] )
        
        if test_fn != None:
            with open( test_fn, "r" ) as fin:
                for line in fin:
                    vec_line    = line.strip().split( "\t" )
                    vec_test.append( [ vec_line[0], vec_line[1], self._tokenizer.convert_tokens_to_ids( self._tokenizer.tokenize( vec_line[1] ) ) ] )
        else:
            vec_test    = []

        vec_class   = list( set( [ v[0] for v in vec_train ] ) )
        vec_class   = sorted( vec_class )   # Guarentee the same ID for same dataset.

        # Set attack matrix for the tokenizer.
        set_train_tok_id    = set()
        for intent, _, vec_tok_id in vec_train:
            for t in set( vec_tok_id ):
                set_train_tok_id.add( t )
        self._tokenizer._set_attack_matrix( list( set_train_tok_id ) )
        print ( "TRAIN TOK #: ", len( set_train_tok_id ) )
              
        return vec_train, vec_dev, vec_test, vec_class                

    def _generate_feed( self, vec_data, map_class, is_train = False ):
        vec_token_id    = []
        vec_answer      = []
        for intent, _, tok_id in vec_data:
            vec_token_id.append( tok_id )
            if is_train:
                vec_answer.append( map_class[ intent ] )
        
        vec_token_id, vec_mask  = padding_2D( vec_token_id, float( 0.0 ) )
        return vec_token_id.astype( "int32" ), np.array( vec_mask ).astype( "int32" ), np.array( vec_answer ).astype( "int32" )
        
    def _shuffle( self, vec ):
        for _ in range( len( vec ) ):
            rand1   = random.randrange( len( vec ) )
            rand2   = random.randrange( len( vec ) )
            tmp     = vec[ rand1 ]
            vec[ rand1 ]    = vec[ rand2 ]
            vec[ rand2 ]    = tmp

        return vec            

    def _get_macro_f1( self, sc, vec_test, vec_class, is_test = False, flog = None ):        
        map_stat    = { c:[ 0, 0, 0, 0.0 ] for c in vec_class }  # Correctly classified, Total Classified, Total Class, F1.
        map_class   = { v:k for k, v in enumerate( vec_class ) }
        batch_size  = self._config[ "learning" ][ "batch_size" ]
        batch_num   = int( len( vec_test ) / batch_size )
        if len( vec_test ) % batch_size != 0:
            batch_num += 1
    
        map_stat[ "reject" ]    = [ 0, 0, 0, 0.0 ]
        intent_classified_as_reject = 0
        for batch_idx in range( batch_num ):
            vec_data    = vec_test[ batch_idx * batch_size: ( batch_idx + 1 ) * batch_size ]
            token_id, mask, _       = self._generate_feed( vec_data, dict() )
            logits, sen_encoded     = sc( token_id, mask )
            logits  = tf.nn.softmax( logits, -1 )   # BS X C.
            logits  = logits.numpy().tolist()
            sen_encoded     = sen_encoded.numpy()
            for data_idx, ( data, logit ) in enumerate( zip( vec_data, logits ) ):
                max_idx     = np.argmax( np.array( logit ) )
                max_score   = logit[ max_idx ]
                intent      = vec_class[ max_idx ]
                if intent == "reject":
                    intent_classified_as_reject += 1

                if intent in data[0]:   # Correct.
                    map_stat[ intent ][0] += 1
                    map_stat[ intent ][1] += 1
                    map_stat[ intent ][2] += 1
                else:   # Wrong.
                    if flog != None:
                        print ( "------------------------------", file = flog )
                        print ( "[%s] --> [%s]" % ( data[0], intent ), file = flog )
                        _, mits = self._get_important_tokens( sc, data[2], map_class[ intent ] )
                        print ( mits, file = flog ) 
                        print ( data[1], file = flog )
                    map_stat[ intent ][1]   += 1
                    map_stat[ data[0].split( "#" )[0] ][2]  += 1        # so.. the total # could change.

    
        print ( "Intent Classified as REJECT: ", intent_classified_as_reject )

        # Get F1 for each class.
        map_updated_stat    = dict()
        for c, stat in map_stat.items():
            if stat[2]  == 0:
                continue

            prec    = stat[0] / stat[1] if stat[1] > 0 else 0.0
            rec     = stat[0] / stat[2]
            stat[3] = 2 * prec * rec / ( prec + rec ) if prec + rec > 0.0 else 0.0
            map_updated_stat[c] = stat
        map_stat    = map_updated_stat
        macro_f1    = sum( [ stat[3] for c, stat in map_stat.items() ] ) / len( map_stat )
        return map_stat, macro_f1

    # Do outflip iteration.
    def do_outflip_iteration( self, store_dir, train_fn, dev_fn, test_fn, iter_num ):
        iter_log_fn     = os.path.join( store_dir, "iter.log" )
        flog            = open( iter_log_fn, "w" )
        
        vec_train, vec_dev, vec_test, vec_class = self._read_data( train_fn, dev_fn, test_fn )
        vec_train_orig  = [ x for x in vec_train ]
        vec_class.append( "reject" )    # At first iteration, this class has NO instance.
        num_class       = len( vec_class )
        map_class       = { v:k for k, v in enumerate( vec_class ) }
        
        print ( "# Known: [%d]. # Train: [%d]. # Dev: [%d]. # Test: [%d]." % ( num_class, len( vec_train ), len( vec_dev ), len( vec_test ) ) )
        print ( "# Known: [%d]. # Train: [%d]. # Dev: [%d]. # Test: [%d]." % ( num_class, len( vec_train ), len( vec_dev ), len( vec_test ) ), file = flog )

        vec_train_reject    = []        # Current training data to be added.
        vec_dev_reject      = []        # Current dev data to be added.
        set_reject_shown    = set()     # Shown sentences.
        vec_iter_results    = []        
        for iter_idx in range( iter_num + 1 ):  
            # 1. Train  & Evaluate the model using current dataset.
            periter_log_fn              = os.path.join( store_dir, "iter_%d.log" % iter_idx )
            periter_attack_log_fn       = os.path.join( store_dir, "iter_attack_%d.log" % iter_idx )
            vec_train                   = vec_train_orig
            print ( "# TRAIN: [%d]" % len( vec_train + vec_train_reject ) )
            print ( "# DEV: [%d]" % len( vec_dev + vec_dev_reject ) )
            checkpoint_fn   = os.path.join( store_dir, "checkpoint_%d" % iter_idx )
            sc, map_test_stat, macro_f1 = self._evaluate_single( checkpoint_fn, vec_train + vec_train_reject, vec_dev + vec_dev_reject, vec_test, vec_class, map_class, periter_log_fn )
            print ( "=========== ITERATION %d ==============" % ( iter_idx + 1 ), file = flog )
            self._print_stat( map_test_stat, macro_f1, flog )
            vec_iter_results.append( macro_f1 )

            if iter_idx == iter_num:
                break
            
            # 2. Generate Reject data.
            tm          = TimeMeasurer()
            tm.start( "DOING TIME" )
            fattack     = open( periter_attack_log_fn, "w" )
            attack_num  = 0
            vec_new_reject  = []

            attack_bs       = 128
            attack_bsnum    = int( len( vec_train ) / attack_bs )
            if len( vec_train ) % attack_bs != 0:
                attack_bsnum    += 1


            # 1. Get Most-Important tokens.
            tm.start( "IM" )
            vec_mi_flags    = []
            vec_mi_tok_pos  = []
            map_intent_core_tokens  = dict()

            for train_idx, ( class_name, _, sen_tok_id ) in enumerate( vec_train ):  # Process One-by-One.
                class_idx                   = map_class[ class_name ]
                flag, vec_token_importance  = self._get_important_tokens( sc, sen_tok_id, class_idx )
                vec_mi_flags.append( flag )
                most_important_tok_idx  = 0
                if flag:
                    for tok_idx, _ in vec_token_importance:
                        if self._tokenizer.convert_id_to_token( sen_tok_id[ tok_idx ] ) not in self._stopwords:
                            most_important_tok_idx  = tok_idx
                            break

                    if class_name not in map_intent_core_tokens:
                        map_intent_core_tokens[ class_name ]    = dict()

                    mi_token    = self._tokenizer.convert_id_to_token( sen_tok_id[ most_important_tok_idx ] )
                    if mi_token not in map_intent_core_tokens[ class_name ]:
                        map_intent_core_tokens[ class_name ][ mi_token ]    = 1
                    else:
                        map_intent_core_tokens[ class_name ][ mi_token ]    += 1

                vec_mi_tok_pos.append( most_important_tok_idx )
            tm.end( "IM" )

            map_ci_token    = dict()
            for class_name, map_mi_info in map_intent_core_tokens.items():
                vec_mi_info = [ [ k, v ] for k, v in map_mi_info.items() ]
                vec_mi_info = sorted( vec_mi_info, key = lambda x:x[1], reverse = True )
                cnt_tot     = int( sum( [ v[1] for v in vec_mi_info ] ) * 0.8 )
                cum_sum     = 0
                cut_idx     = len( vec_mi_info ) - 1 
                for idx , ( k, v ) in enumerate( vec_mi_info ):
                    cum_sum += v
                    if cum_sum > cnt_tot:
                        cut_idx = idx - 1
                        break
                if cut_idx == -1:
                    cut_idx = 0
                cut_val = vec_mi_info[ cut_idx ][1]
                vec_updated_mi_info = [ [ k, v ] for k, v in vec_mi_info if v > cut_val ]
                if len( vec_updated_mi_info ) == 0:
                    vec_updated_mi_info = [ vec_mi_info[0] ]

                map_ci_token[ class_name ]  = set( [ v[0] for v in vec_updated_mi_info[:5] ] )
          
            idx_tot         = 0 
            assert ( len( vec_mi_tok_pos ) == len( vec_train ) )
            for attack_bsidx in range( attack_bsnum ):
                vec_attack_batch    = vec_train[ attack_bsidx * attack_bs: min( (attack_bsidx + 1 ) * attack_bs, len( vec_train ) ) ]
                vec_batch_mi        = vec_mi_tok_pos[ attack_bsidx * attack_bs: min( (attack_bsidx + 1 ) * attack_bs, len( vec_train ) ) ]

                # 2. Get Batch Gradient.
                tm.start( "GRAD" )  
                token_id_b, mask_b, class_b = self._generate_feed( vec_attack_batch, map_class, True )
                outflip_grad_b    = sc.get_outflip_gradient( token_id_b, mask_b, class_b, vec_batch_mi ).numpy()  # BS X |V|.
                tm.end( "GRAD" )

                # 3. Attack!
                for train_idx, ( class_name, _, sen_tok_id ) in enumerate( vec_attack_batch ):  # Process One-by-One.
                    class_idx   = map_class[ class_name ]
                    orig_sen    = " ".join( [ self._tokenizer.convert_id_to_token( tid ) for tid in sen_tok_id ] )
                    most_important_tok_idx   = vec_mi_tok_pos[ idx_tot ]
                    orig_tok_id     = sen_tok_id[ most_important_tok_idx ]  # Considering <s>.      
                    if vec_mi_flags[ idx_tot ] and self._tokenizer.convert_id_to_token( orig_tok_id ) in map_ci_token[ class_name ]:
                        outflip_grad    = outflip_grad_b[ train_idx ]   # |V|.
                    
                        # ATTACK!
                        # We need to "Shuffle" the tokens, otherwise they will all be replaced to the same token!
                        tm.start( "DIST" )
                        # 1 % of the total vocab. 
                        attack_cand_num = int( self._config[ "attack_ratio" ] * self._tokenizer.get_vocab_size()  )
                        outflip_cand    = np.argpartition( outflip_grad, attack_cand_num )[ :attack_cand_num ]
                        cand_cos_dist   = self._tokenizer.get_cosign_dist( orig_tok_id, outflip_cand )    
                        final_cand      = [ cand for cand, dist in zip( outflip_cand.tolist(), cand_cos_dist.tolist() ) if dist < 0.0 ]

                        if self._config[ "use_antonym" ]:
                            for syn in wordnet.synsets( self._tokenizer.convert_id_to_token( orig_tok_id ) ):
                                for l in syn.lemmas(): 
                                    if l.antonyms(): 
                                        vec_ant.append( l.antonyms()[0].name() ) 
                            
                            final_cand  += [ self._tokenizer.convert_tokens_to_ids( list( set( vec_ant ) ) ) ]

                        final_cand      = self._shuffle( final_cand )
                        tm.end( "DIST" )

                        tm.start( "EVAL" )
                        vec_adv_sen_cand    = [ sen_tok_id[ :most_important_tok_idx ] + [ adv_cand_tok_id ] + sen_tok_id[ most_important_tok_idx + 1: ] for adv_cand_tok_id in final_cand ]
                        adv_sen_bn          = int( len( vec_adv_sen_cand ) / attack_bs  )
                        if len( vec_adv_sen_cand ) % attack_bs != 0: adv_sen_bn += 1
                        for adv_sen_batch_idx in range( adv_sen_bn ):
                            adv_sen_target  = vec_adv_sen_cand[ adv_sen_batch_idx * attack_bs: min( len( vec_adv_sen_cand ), ( adv_sen_batch_idx + 1 ) * attack_bs ) ]
                            adv_token_id, mask, _   = self._generate_feed( [ [ 0, 0, adv_sen ] for adv_sen in adv_sen_target ], dict() )
                            logits_adv_class, _     = sc( adv_token_id, mask )  # BS X C.
                            adv_class_idx           = np.argmax( logits_adv_class, axis = -1 ).tolist() # BS

                            is_generated    = False
                            for sen_cand, sen_class in zip( adv_sen_target, adv_class_idx ):
                                if sen_class != class_idx:
                                    continue
                                    
                                new_sen                 = " ".join( [ self._tokenizer.convert_id_to_token( tid ) for tid in sen_cand ] )
                                if new_sen not in set_reject_shown:
                                    print ( "%s\t%s" % ( orig_sen, new_sen ), file = fattack )
                                    fattack.flush()
                                    attack_num  += 1
                                    set_reject_shown.add( new_sen )
                                    vec_new_reject.append( sen_cand )
                                    is_generated    = True
                                    break

                            if is_generated:
                                break

                        tm.end( "EVAL" )
                
                    if idx_tot % 1000 == 0:
                        print ( "ATTACK PROGRESS: %d/%d" % ( idx_tot, len( vec_train ) ) )
                    idx_tot += 1
            print ( "ATTACK RATE: %d / %d" % ( attack_num, len( vec_train ) ) )
            fattack.close()

            tm.end( "DOING TIME" )
            tm.printall() 

            # 3. Add new rejects to training / dev set.
            vec_new_reject  = self._shuffle( vec_new_reject )
            num_train       = int( len( vec_new_reject ) * 0.9 )
                
            vec_train_reject    += [ [ "reject", "", r ] for r in vec_new_reject[ :num_train ] ]
            vec_dev_reject      += [ [ "reject", "", r ] for r in vec_new_reject[ num_train + 1: ] ]

        flog.close()

        return vec_iter_results

    def _get_important_tokens( self, sc, sen_tok_id, class_idx ):
        vec_run_cands       = [ [ sen_tok_id, -1 ] ]    # Vector of: Tok ID - removed tok ID. First one is always the original one.
        for tidx, _ in enumerate( sen_tok_id ):
            if tidx == 0 or tidx == len( sen_tok_id ) - 1:  # <s> / </s>
                continue    

            vec_tok_partial     = sen_tok_id[ :tidx ] + [0] + sen_tok_id[ tidx + 1: ]   # MASKED.
            vec_run_cands.append( [ vec_tok_partial, tidx ] )
        
        token_id, mask, _   = self._generate_feed( [ [ 0, 0, vec_tok_id ] for vec_tok_id, _ in vec_run_cands ], dict(), False )
        logits_class, _     = sc( token_id, mask )
        logits_score        = logits_class.numpy()[:, class_idx ].tolist()  # BS.
        vec_class_score     = sorted( [ [ r, s ] for r, s in enumerate( logits_class.numpy()[0].tolist() ) ], key = lambda x: x[-1], reverse = True )
        if vec_class_score[0][0] != class_idx:
            return False, []
    
        result_score        = logits_score[0]
        vec_cont            = [ [ tidx, result_score - cand_score ] for ( _, tidx ), cand_score in zip( vec_run_cands[1:], logits_score[1: ] ) ]
        return True, sorted( vec_cont, key = lambda x: x[1], reverse = True )

    def _print_stat( self, map_stat, macro_f1, stream = sys.stdout ):
        print ( "Intent\tCorrect\tClassified\tTotal\tF1", file = stream )
        for c, stat in map_stat.items():
            print ( "%s\t%d\t%d\t%d\t%.2f" % ( c, stat[0], stat[1], stat[2], stat[3] ), file = stream )
        print ( "Macro F1: %.3f" % ( macro_f1 * 100.0 ), file = stream )                

    def _evaluate_single( self, checkpoint_fn, vec_train, vec_dev, vec_test, vec_class, map_class, log_path ):
        flog            = open( log_path, "w" )
        batch_size      = self._config[ "learning" ][ "batch_size" ]
        train_batch_num = int( len( vec_train ) / batch_size )
        if len( vec_train ) % batch_size != 0:
            train_batch_num += 1

        num_class   = len( map_class )
        sc      = SentenceClassifier( self._config, self._tokenizer, num_class, train_batch_num = train_batch_num )
        step    = 1
        prev_dev= 0.0
        tol_cnt = 0
        while True:
            print ( "STEP: %d" % step )
            vec_train   = self._shuffle( vec_train )

            for batch_idx in range( train_batch_num ):
                vec_data    = vec_train[ batch_idx * batch_size: ( batch_idx + 1 ) * batch_size ]
                token_id, mask, answers = self._generate_feed( vec_data, map_class, True )
                loss    = sc.train( token_id, mask, answers )
            
            map_dev_stat, macro_f1_dev  = self._get_macro_f1( sc, vec_dev, vec_class )
            self._print_stat( map_dev_stat, macro_f1_dev )

            if macro_f1_dev < prev_dev or ( macro_f1_dev == prev_dev and step > 5 ):
                if tol_cnt >= self._config[ "learning" ][ "tolerance" ]:
                    print ( "Training Complete." )
                    break
                else:
                    tol_cnt += 1
                    print ( "Tolerance Cnt %d" % tol_cnt )
            else:
                sc.save_weights( checkpoint_fn )
                print ( "Model Saved." )
                prev_dev    = macro_f1_dev
                tol_cnt     = 0

            step += 1
        
        # Restore the Best configuration.
        sc.load_weights( checkpoint_fn )   

        # Do Test Here.
        map_test_stat, macro_f1_test    = self._evaluate( sc, vec_test, vec_class, flog = flog )
        flog.close()
        return sc, map_test_stat, macro_f1_test

    
    def evaluate_only( self, checkpoint_fn, input_dir, log_path ):
        flog    = open( log_path, "w" )

        vec_train, vec_dev, vec_test, vec_class = self._read_data( input_dir )
        num_class                               = len( vec_class )
        map_class                               = { v:k for k, v in enumerate( vec_class ) }
        batch_size                              = self._config[ "learning" ][ "batch_size" ]
        train_batch_num                         = int( len( vec_train ) / batch_size )
        if len( vec_train ) % batch_size != 0:
            train_batch_num += 1

        print ( "File Loaded From: [%s]" % input_dir )
        print ( "# Known: [%d]. # Train: [%d]. # Dev: [%d]. # Test: [%d]." % ( num_class, len( vec_train ), len( vec_dev ), len( vec_test ) ) )
        print ( "File Loaded From: [%s]" % input_dir, file = flog )
        print ( "# Known: [%d]. # Train: [%d]. # Dev: [%d]. # Test: [%d]." % ( num_class, len( vec_train ), len( vec_dev ), len( vec_test ) ), file = flog )

        sc      = SentenceClassifier( self._config, self._tokenizer, num_class, train_batch_num = train_batch_num )
        sc.load_weights( checkpoint_fn )   
        
        map_test_stat, macro_f1_test    = self._evaluate( sc, vec_test, vec_class, flog = flog )
        flog.close()
        return map_test_stat, macro_f1_test

    def _evaluate( self, sc, vec_test, vec_class, flog = None ):
        if len( vec_test ) == 0:
            return dict(), 0.0

        print ( "Testing..." )

        map_test_stat, macro_f1_test    = self._get_macro_f1( sc, vec_test, vec_class, True, flog = flog)    
        self._print_stat( map_test_stat, macro_f1_test )
        self._print_stat( map_test_stat, macro_f1_test, flog )

        return map_test_stat, macro_f1_test

