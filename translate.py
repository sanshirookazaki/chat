#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import data_utils
from tensorflow.models.rnn.translate import seq2seq_model
from tensorflow.python.platform import gfile


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 500, "input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 500, "output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "datas", "Data directory")       
tf.app.flags.DEFINE_string("train_dir", "datas", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  source_file = open(source_path,"r")
  target_file = open(target_path,"r")

  source, target = source_file.readline(), target_file.readline()     
  counter = 0
  while source and target and (not max_size or counter < max_size):
    counter += 1
    if counter % 50 == 0:                         
      print("  reading data line %d" % counter)
      sys.stdout.flush()

    source_ids = [int(x) for x in source.split()]   
    target_ids = [int(x) for x in target.split()]    
    target_ids.append(data_utils.EOS_ID)             
    for bucket_id, (source_size, target_size) in enumerate(_buckets):        
      if len(source_ids) < source_size and len(target_ids) < target_size:    
        data_set[bucket_id].append([source_ids, target_ids])                 
        break
    source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.in_vocab_size, FLAGS.out_vocab_size, _buckets,                           
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,      
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,                       
      forward_only=forward_only)                                                    
  
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)                             
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):                              
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)          
    model.saver.restore(session, ckpt.model_checkpoint_path)                        
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())                                      
  return model                                                                      


def train():
                                                                                    
  print("Preparing data in %s" % FLAGS.data_dir)                                
  in_train, out_train, in_dev, out_dev, _, _ = data_utils.prepare_wmt_data(           
      FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)                     
                                                  

  with tf.Session() as sess:
    

    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))       
    model = create_model(sess, False)                                               

    print ("Reading development and training data (limit: %d)."     
           % FLAGS.max_train_data_size)                                             
    dev_set = read_data(in_dev, out_dev)                                             
    train_set = read_data(in_train, out_train, FLAGS.max_train_data_size)            
                                                                                    
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]         
    train_total_size = float(sum(train_bucket_sizes))                               

 
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size     
                           for i in xrange(len(train_bucket_sizes))]              

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
     
      random_number_01 = np.random.random_sample()                     
      bucket_id = min([i for i in xrange(len(train_buckets_scale))      
                       if train_buckets_scale[i] > random_number_01])

      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(   
          train_set, bucket_id)                                           
                                                                          
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, 
                                   target_weights, bucket_id, False)      
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      
      if current_step % FLAGS.steps_per_checkpoint == 0:
     
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))


        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)

        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        for bucket_id in xrange(len(_buckets)):
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    print ("Hello!!")
    model = create_model(sess, True)                         
    model.batch_size = 1  
    
    in_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab_in.txt")     
    out_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab_out.txt" )
                                                                        
    in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)        
    _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)    

    
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()    
    while sentence:

      token_ids = data_utils.sentence_to_token_ids(sentence, in_vocab)   
      
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])               

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
    
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,      
                                       target_weights, bucket_id, True)

      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]       

      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]                      

      print(" ".join([rev_out_vocab[output] for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()                                             
                                                                                  

def self_test():
  
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
