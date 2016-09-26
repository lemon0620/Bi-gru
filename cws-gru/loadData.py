# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 13:37:05 2016

@author: Lemon
"""

import theano
import numpy as np

__author__ = 'Lemon'

#load character embedding from file
def loadCharacterEmb(file):
    #save index for words
    word_idx = {}
    for index, name in enumerate(['[OOV]','[BOS]','[EOS]']):
        word_idx[name] = index
        
    with open(file, 'r') as fin:
        line = fin.readline().decode('utf-8').strip('\r\n ')
        n_dict, emb_size = line.split()
        n_dict, emb_size = np.int32(n_dict), np.int32(emb_size)
        #save emb for words
        emb = np.random.normal(loc = 0.0, scale = 0.01, size = (n_dict+3, emb_size)).astype(theano.config.floatX)
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            items = line.split()
            assert len(items) == emb_size + 1
            word = items[0]
            word_idx[word] = len(word_idx)
            for index , value in enumerate(items[1:]):
                emb[word_idx[word]][index] = np.float32(value)
            
    return word_idx, emb
    
#random emb for words
def randomCharacterEmb(file):
    #save index for words 
    word_idx = {}
    for index, name in enumerate(['[OOV]','[BOS]','[EOS]']):
        word_idx[name] = index
    
    with open(file, 'r') as fin:
        line = fin.readline().decode('utf-8').strip('\r\n ')
        n_dict, emb_size = line.split()
        n_dict, emb_size = np.int32(n_dict), np.int32(emb_size)
        #save emb for words
        emb = np.random.normal(loc = 0.0, scale = 0.01, size = (n_dict+3, emb_size)).astype(theano.config.floatX)
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            items = line.split()
            assert len(items) == emb_size + 1
            word = items[0]
            word_idx[word] = len(word_idx)
            
    return word_idx, emb
    
#load data(train\dev\test) from file:
def loadData(file):
    DataSet = []
    Tag = []
    with open(file, 'r') as fin:
        for line in fin:
            line = line.decode('utf-8').strip('\r\n ')
            items = line.split()
            if len(items) == 0:
                continue
            sentence = []
            tag = []
            for word_label in items:
                word, label = word_label.split('_')
                sentence.append(word)
                tag.append(label)
            DataSet.append(sentence)
            Tag.append(tag)

    return DataSet, Tag
    
#get one-hot vector for each instance    
def word2id(word_idx, dataSet, Tag):
    codedSet = []
    for sen in dataSet:
        codeSen = []
        for word in sen:
            if(word not in word_idx):
                codeSen.append(word_idx['[OOV]'])
            else:
                codeSen.append(word_idx[word])
        codedSet.append(codeSen)
    tags = {'S':0, 'B':1, 'M':2, 'E':3}
    codedTag = []
    for tag in Tag:
        tmp = []
        for t in tag:
            tmp.append(tags[t])
        codedTag.append(tmp)
    return codedSet, codedTag
    

#add windows for data,lsize is the left window_size,rsize is the right window_size for word
def addWindows(dataSet, lsize, rsize):
    reSet = []
    for sen in dataSet:
        m = len(sen)
        senS = []
        for wIndex in range(m):
            word = []
            tidx = wIndex
            winsize = lsize + rsize
            
            while(tidx - lsize < 0):
                #add dic_index for [BOS] -- 1
                word.append(1)
                tidx = tidx + 1
                winsize = winsize - 1
                
            word.append(sen[wIndex])
            tidx = wIndex
            while(tidx + 1 < m and winsize > 0):
                tidx = tidx + 1
                word.append(sen[tidx])
                winsize = winsize - 1
                
            while(winsize > 0):
                #add dic_index for [EOS] -- 2
                word.append(2)
                winsize = winsize - 1
            senS.append(word)
        reSet.append(senS)
    return reSet

#get minibatch index
def getMinibatchesIdx(n, minibatch_size, shuffle = False):
    idx_list = np.arange(n, dtype = "int32")

    if shuffle:
        np.random.shuffle(idx_list)
    
    minibatches = []
    minibatch_start = 0
    #print 'get batch: %d,%d,%d' % (n, minibatch_size, n // minibatch_size)
    for i in range(n // minibatch_size):
        #print 'batch %d' % i
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size 
        
    if(minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
        #while(minibatch_start <= n):
            
        # Make a minibatch out of what is left
            
    return minibatches

def prepareData(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen,n_samples,3)).astype('int32')
    mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    y = np.zeros((maxlen, n_samples)).astype('int32')
    
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx,:] = s
        mask[:lengths[idx], idx] = 1.
    for idx, t in enumerate(labels):
        y[:lengths[idx], idx] = t
    
    return x, mask, y
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
