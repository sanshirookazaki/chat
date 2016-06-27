#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import data_utils


class Seq2SeqModel(object):   

  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
   
    self.source_vocab_size = source_vocab_size    
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    
    output_projection = None
    softmax_loss_function = None
    
    if num_samples > 0 and num_samples < self.target_vocab_size:     
      with tf.device("/cpu:0"):
        w = tf.get_variable("proj_w", [size, self.target_vocab_size])            
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)                                        


      def sampled_loss(inputs, labels):
        with tf.device("/cpu:0"):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                            self.target_vocab_size)
      softmax_loss_function = sampled_loss

    
    single_cell = rnn_cell.GRUCell(size)
    if use_lstm:                                    
      single_cell = rnn_cell.BasicLSTMCell(size)    
    cell = single_cell
    if num_layers > 1:
      cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)


    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell, source_vocab_size,
          target_vocab_size, output_projection=output_projection,
          feed_previous=do_decode)

    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))


    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]


    if forward_only:
      self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, self.target_vocab_size,
          lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)

      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [tf.matmul(output, output_projection[0]) +
                             output_projection[1]
                             for output in self.outputs[b]]
    else:
      self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, self.target_vocab_size,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)


    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):

    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)   

  
    if not forward_only:
      output_feed = [self.updates[bucket_id], 
                     self.gradient_norms[bucket_id], 
                     self.losses[bucket_id]]  
    else:
      output_feed = [self.losses[bucket_id]]  
      for l in xrange(decoder_size):  
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  
    else:
      return None, outputs[0], outputs[1:]  



  def get_batch(self, data, bucket_id):
    
    encoder_size, decoder_size = self.buckets[bucket_id]   
    encoder_inputs, decoder_inputs = [], []
    
    for _ in xrange(self.batch_size):                                     
      encoder_input, decoder_input = random.choice(data[bucket_id])        

      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input)) 
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))       
                                                                      
      decoder_pad_size = decoder_size - len(decoder_input) - 1               
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +             
                            [data_utils.PAD_ID] * decoder_pad_size)          
                                                                       

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []


    for length_idx in xrange(encoder_size):                                  
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))


    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))


      batch_weight = np.ones(self.batch_size, dtype=np.float32)           
      for batch_idx in xrange(self.batch_size):

        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:   
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights          