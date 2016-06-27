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

import gzip
import os
import re
import tarfile

from tensorflow.python.platform import gfile
from six.moves import urllib

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  
  if os.path.exists(vocabulary_path):       
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    
    f = open(data_path,"r")
    counter = 0
    for line in f:
      counter += 1
      if counter % 100 == 0:
        print("  processing line %d" % counter)
      tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
      for w in tokens:
        word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w   
        if word in vocab:       
          vocab[word] += 1
        else:
          vocab[word] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]       
      
    vocab_file = open(vocabulary_path,"w")
    for w in vocab_list:
      vocab_file.write(w + "\n")
    vocab_file.close()  
    f.close()

def initialize_vocabulary(vocabulary_path):
  
  if os.path.exists(vocabulary_path):       
    rev_vocab = []

    f = open(vocabulary_path,"r")
    rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    f.close()
    return vocab, rev_vocab

  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  
  print("Tokenizing data in %s" % data_path)
  vocab, _ = initialize_vocabulary(vocabulary_path)                 
  data_file = open(data_path,"r")
  tokens_file = open(target_path,"w")
  counter = 0
  for line in data_file:
    counter += 1
    if counter % 100 == 0:
      print("  tokenizing line %d" % counter)
    token_ids = sentence_to_token_ids(line, vocab, tokenizer,         
                                            normalize_digits)
    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")   
  data_file.close()
  tokens_file.close()      

def prepare_wmt_data(data_dir, in_vocabulary_size, out_vocabulary_size):
                    
  train_path = os.path.join(data_dir, "train_data")               
  dev_path = os.path.join(data_dir, "test_data")                     


  out_vocab_path = os.path.join(data_dir, "vocab_out.txt" )   
  in_vocab_path = os.path.join(data_dir, "vocab_in.txt" )   
  create_vocabulary(out_vocab_path, train_path + "_out.txt", out_vocabulary_size)    
  create_vocabulary(in_vocab_path, train_path + "_in.txt", in_vocabulary_size)    

  out_train_ids_path = train_path + ("_ids_out.txt" )         
  in_train_ids_path = train_path + ("_ids_in.txt" )        
  data_to_token_ids(train_path + "_out.txt", out_train_ids_path, out_vocab_path)    
  data_to_token_ids(train_path + "_in.txt", in_train_ids_path, in_vocab_path)    


  out_dev_ids_path = dev_path + ("_ids_out.txt" )             
  in_dev_ids_path = dev_path + ("_ids_in.txt" )             
  data_to_token_ids(dev_path + "_out.txt", out_dev_ids_path, out_vocab_path)         
  data_to_token_ids(dev_path + "_in.txt", in_dev_ids_path, in_vocab_path)         

  return (in_train_ids_path, out_train_ids_path,           
          in_dev_ids_path, out_dev_ids_path,
          in_vocab_path, out_vocab_path)
