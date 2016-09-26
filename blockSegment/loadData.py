# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:15:16 2016

@author: Lemon
"""

import theano
import numpy as np
import re
import cPickle

#load character embedding from file
def RandomWordEmb(file, emb_size):
    #save index for words
    word_idx = {}
    for index, name in enumerate(['<e>','[BOS]','[EOS]']):
        word_idx[name] = index
    
    fd = open(file,'rb')    
    vocab = cPickle.load(fd)
    fd.close()  
    
    n_dict = len(vocab.keys())
    n_dict, emb_size = np.int32(n_dict), np.int32(emb_size)
    emb = np.random.normal(loc = 0.0, scale = 0.01, size = (n_dict+3, emb_size)).astype(theano.config.floatX)
    
    for word in vocab:        
        word = word.decode('utf-8')
        word_idx[word] = len(word_idx)
        
    return word_idx, emb
    

def loadData(data_f, index_f):
    DataSet = []
    Index = []
    f_data = open(data_f, 'r')
    f_index = open(index_f, 'r')
    
    data = f_data.readline().decode('utf-8').strip('\r\n ')
    indexx = f_index.readline().strip('\r\n ') 
    
    while data:
        sentence = []
        index = []
        
        for word in data.split(" "):
            sentence.append(word)
        DataSet.append(sentence)
        
        for ind in indexx.split(" "):  
            if ind.isdigit():
                index.append(int(ind))
            else:
                continue
        Index.append(index)   
        
        data = f_data.readline().decode('utf-8').strip('\r\n ')
        indexx = f_index.readline().strip('\r\n ')   
        
        
    return DataSet, Index
    
#get one-hot vector for each instance    
def word2id(word_idx, dataSet, Index_pos):
    codedSet = []
    IndexSet = []
    codedTag = []
    pat = ur"[！，；：、？）。),:;.!?]"
    
    for sen_id, sen in enumerate(dataSet):
        codeSen = []
        index = []
        tag = []
        for ind, word in enumerate(sen):
            if(re.match(pat, word)):
                index.append(1.)
                if(ind in Index_pos[sen_id]):
                    tag.append(1)
                else:
                    tag.append(0)
            else:
                index.append(0.)
                tag.append(0)
                
            if(word not in word_idx):
                codeSen.append(word_idx['UNK'])
            else:
                codeSen.append(word_idx[word])    
        
        if(not index[-1] == 1.):
            index.pop()
            index.append(1.)
            tag.pop()
            tag.append(1)                    
        codedSet.append(codeSen)
        IndexSet.append(index)
        codedTag.append(tag)
        
    return codedSet, IndexSet, codedTag
    
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

def prepareData(seqs, indexs, labels, winsize, maxlen=None):
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
        new_indexs = []
        new_labels = []
        new_lengths = []
        for l, l_y, s, i, y in zip(lengths, lengths_y, seqs, indexs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_indexs.append(i)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs
        indexs = new_indexs        
        
        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen,n_samples,winsize)).astype('int32')
    mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)    
    index = np.zeros((maxlen,n_samples)).astype(theano.config.floatX)    
    y = np.zeros((maxlen, n_samples)).astype('int32')
    
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx,:] = s
        mask[:lengths[idx], idx] = 1.
    
    for idx, i in enumerate(indexs):
        index[:lengths[idx],idx] = i
        
    for idx, t in enumerate(labels):
        y[:lengths[idx], idx] = t
    
    return x, mask, index, y














