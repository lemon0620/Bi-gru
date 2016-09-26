# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 15:40:42 2016

@author: Lemon
"""
from cws_lstm import LSTMEncoder
import numpy as np
#from datatime import datetime
from loadData import RandomWordEmb, loadData, word2id, addWindows, getMinibatchesIdx, prepareData
import sys

#calculate accuracy for data(train\dev\test)
def cal_acc(model, x_data, y_data, Index, kf_data):
    accuracy = 0.0
    total_num = 0
    for ind in kf_data:
        x = [x_data[t] for t in ind]
        index = [Index[t] for t in ind]
        y = [y_data[t] for t in ind] 
        
        x, mask, index, y = prepareData(x,index, y, 3)  
        pred = model.predict(x, mask)
        accuracy += np.sum((pred == y) * index)
        total_num += np.sum(index)
        #print >>f,pred       
        #print >>f,y
    acc = accuracy / total_num
    return acc

def cal_loss(model, x_data, y_data, Index, kf_data):
    losses = 0.0
    batch_num = 0
    for ind in kf_data:
        x = [x_data[t] for t in ind]
        index = [Index[t] for t in ind]
        y = [y_data[t] for t in ind] 
        
        x, mask, index, y = prepareData(x, index, y, 3)  
        loss = model.ce_loss(x, mask, y, index)
        losses += loss
        batch_num += 1
    return losses / batch_num
    
#train model
def train_lstm(
    word_dim = 100,
    hidden_dim = 150,
    lwindow = 0,
    rwindow = 2,
    Regularization = 0.001,
    batch_size = 20,
    y_dim = 2,
    max_epochs = 100,
    learning_rate=0.2,
    evaluate_loss_after=2,
    drop_rate = 0.2
):    
    
    f = open(sys.argv[1],'w')
    #load embedding for each character
    print "load embedding......"
    print >>f,"load embedding......"
    word_idx, emb = RandomWordEmb('data/vocab.zh.pkl',100)

    #load train\dev\test
    print "loadDate......"
    print >>f,"loadDate......"
    DataSet, Index = loadData('data/ch-en.ch','data/ch.index')
    #x_train, y_train = loadData('data/pku.train')
    #x_dev, y_dev = loadData('data/pku.dev')
    #x_test, y_test = loadData('data/pku.test')

    #get one-hot vectors for data
    codedSet, IndexSet, codedTag = word2id(word_idx, DataSet, Index)
    #x_train, y_train = word2id(word_idx, x_train, y_train)
    #x_dev, y_dev = word2id(word_idx, x_dev, y_dev)
    #x_test, y_test = word2id(word_idx, x_test, y_test)

    #add windows to data
    codedSet = addWindows(codedSet, 0, 2)
    #x_train = addWindows(x_train, lwindow, rwindow)
    #x_dev = addWindows(x_dev, lwindow, rwindow)
    #x_test = addWindows(x_test, lwindow, rwindow)
    
    #get minibatch
    kf_data = getMinibatchesIdx(len(codedSet), 20)
    #kf_train = getMinibatchesIdx(len(y_train), batch_size)
    #kf_dev = getMinibatchesIdx(len(y_dev), batch_size)
    #kf_test = getMinibatchesIdx(len(y_test), batch_size)
    
    #build model
    print "build..."
    print >>f,"build..."
    model = LSTMEncoder(word_dim * 3, hidden_dim, emb, y_dim, Regularization,drop_rate, use_drop=True)
    use_noise = model.__build_batch__()
    #start training
    print "start training..."
    print >>f,"start training..."
    
    losses = []
    for epoch in range(max_epochs+1):    
        i = 0
        for train_idx in kf_data:
            use_noise.set_value(1.)
            if(i % 200 == 0):
                #print i
                print >>f,'---> training (%d : %d)' % (i,len(codedTag))
                f.flush()
            i += batch_size
            x = [codedSet[t] for t in train_idx]
            index = [IndexSet[t] for t in train_idx]
            y = [codedTag[t] for t in train_idx]       
            x, mask, index, y =  prepareData(x, index, y, 3)
            model.train_batch(x, mask, y, index, learning_rate)
        train_loss = cal_loss(model, codedSet, codedTag, IndexSet, kf_data)
        #losses.append(train_loss)
        print >>f, 'epoch %d ,loss: %f' % (epoch,train_loss)
        use_noise.set_value(0.)
        train_acc = cal_acc(model, codedSet, codedTag, IndexSet, kf_data)
        #dev_acc = cal_acc(model, x_dev, y_dev, kf_dev)
        #test_acc = cal_acc(model, x_test, y_test, kf_test)
        print 'train_acc = %f' % train_acc
        #print 'dev_acc = %f' % dev_acc
        #print 'test_acc = %f' % test_acc
        print >>f,'train_acc = %f' % train_acc
        #print >>f,'dev_acc = %f' % dev_acc
        #print >>f,'test_acc = %f' % test_acc
        f.flush()
        '''
        # Adjust the learning rate if loss increases
        if (len(losses) > 1 and losses[-1] > losses[-2]):
           learning_rate = learning_rate * 0.5
           print "Setting learning rate to %f" % learning_rate
        '''
        epoch = epoch + 1
    f.close()














