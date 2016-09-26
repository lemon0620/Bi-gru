# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 13:40:03 2016

@author: Lemon
"""

import numpy as np
from train_lstm import train_lstm
from loadData import loadCharacterEmb,loadData,word2id,addWindows,getMinibatchesIdx,prepareData


__author__ = 'Lemon'

if __name__ == '__main__':
    '''  
    #load embedding for each character
    print "load embedding......"
    word_idx, emb = loadCharacterEmb('data/pku_character_emb.txt')
    
    #load train\dev\test
    print "loadDate......"
    x_train, y_train = loadData('data/pku.train')
    x_dev, y_dev = loadData('data/pku.dev')
    x_test, y_test = loadData('data/pku.test')

    #get one-hot vectors for data
    x_train, y_train = word2id(word_idx, x_train, y_train)
    x_dev, y_dev = word2id(word_idx, x_dev, y_dev)
    x_test, y_test = word2id(word_idx, x_test, y_test)

    
    #add windows to data
    x_train = addWindows(x_train, 0, 2)
    x_dev = addWindows(x_dev, 0, 2)
    x_test = addWindows(x_test, 0, 2)
    
    
    #get minibatch
    kf_train = getMinibatchesIdx(len(x_train[0]), 10)
    kf_dev = getMinibatchesIdx(len(x_dev[0]),10)
    kf_test = getMinibatchesIdx(len(x_test[0]),10)
    
    for train_idx in kf_train:
        x = [x_train[t] for t in train_idx]
        y = [y_train[t] for t in train_idx]
        
        x, mask, y = prepareData(x, y)
        print np.shape(x),np.shape(mask),np.shape(y)
    '''

    train_lstm(max_epochs=100)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
