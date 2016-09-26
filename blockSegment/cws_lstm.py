# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 14:47:31 2016

@author: Lemon
"""

import theano
import theano.tensor as T
import numpy as np
from optimizer import AdaDeltaOptimizer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

__author__ = 'Lemon'

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)
trng = RandomStreams(SEED)
   
class LSTMEncoder(object):

    def __init__(self, word_dim, hidden_dim, embedding, y_dim, Regularization,  dropout=0, verbose=True, use_drop = False):
        if verbose:
            print('Building {}...'.format(self.__class__.__name__))

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.dropout = dropout
        self.embedding = theano.shared(embedding)
        self.y_dim = y_dim
        self.Regularization = Regularization                
        #Used for dropout
        self.use_noise = theano.shared(np.asarray(0., dtype=theano.config.floatX))
        self.use_drop = use_drop

        # forward GRU , for x are word_dim * hidden_dim,  for h/c are hidden_dim * hidden_dim
        self.W_zx_forward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_zh_forward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))

        self.W_rx_forward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_rh_forward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))

        self.W_hx_forward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_hh_forward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))
        
        
        # backword GRU, for x are word_dim * hidden_dim,  for h/c are hidden_dim * hidden_dim
        self.W_zx_backward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_zh_backward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))

        self.W_rx_backward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_rh_backward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX))

        self.W_hx_backward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (word_dim,hidden_dim)).astype(theano.config.floatX))
        self.W_hh_backward = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (hidden_dim,hidden_dim)).astype(theano.config.floatX)) 

            
        # for softmax y
        self.U = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (2 * hidden_dim,y_dim)).astype(theano.config.floatX))
        self.b = theano.shared(np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),
            (y_dim,)).astype(theano.config.floatX))


        self.params = [self.W_zx_forward,self.W_zh_forward,
                       self.W_rx_forward,self.W_rh_forward,
                       self.W_hx_forward,self.W_hh_forward,
                       self.embedding,
                       self.U,self.b,
                      ]

        if verbose:
            print('Architecture of {} built finished'.format(self.__class__.__name__))
            print('Word dimension:  %d' % self.word_dim)
            print('Hidden dimension: %d' % self.hidden_dim)
            print('Dropout Rate:     %f' % self.dropout)

        self.theano = {}
        #self.__build__()
        #self.__build_batch__()
        
    def _step_batch(self, x_t, m_t, h_t_1, W_zx_forward, W_zh_forward, W_rx_forward, W_rh_forward, W_hx_forward, W_hh_forward):
        
        
        z_t = T.nnet.sigmoid(T.dot(x_t, W_zx_forward) + T.dot(h_t_1, W_zh_forward))
        r_t = T.nnet.sigmoid(T.dot(x_t, W_rx_forward) + T.dot(h_t_1, W_rh_forward))
        c_t = T.tanh(T.dot(x_t, W_hx_forward) + T.dot(r_t * h_t_1, W_hh_forward))
        h_t = (T.ones_like(z_t) - z_t) * h_t_1 + z_t * c_t
        
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_1
        
        if self.use_drop:
            self.dropout_layer(h_t, self.use_noise, trng, self.dropout)
            
        #batch * 4
        #y_t = T.nnet.softmax(T.dot(h_t,U) + b)
               
        return h_t                 
        
    def forward_sequence_batch(self,x,mask,batch_size):
        h0 = T.zeros((batch_size,self.hidden_dim), dtype=theano.config.floatX)
        
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[x,mask],
                            outputs_info=[h0],
                            non_sequences= [self.W_zx_forward,self.W_zh_forward,
                                            self.W_rx_forward,self.W_rh_forward,                                            
                                            self.W_hx_forward,self.W_hh_forward,
                                           ]) 
                                  
        return hs
        
    def _step_batch_backward(self, x_t, m_t, h_t_1, W_zx_backward, W_zh_backward, W_rx_backward, W_rh_backward, W_hx_backward, W_hh_backward):
                
        z_t = T.nnet.sigmoid(T.dot(x_t, W_zx_backward) + T.dot(h_t_1, W_zh_backward))
        r_t = T.nnet.sigmoid(T.dot(x_t, W_rx_backward) + T.dot(h_t_1, W_rh_backward))
        c_t = T.tanh(T.dot(x_t, W_hx_backward) + T.dot(r_t * h_t_1, W_hh_backward))
        h_t = (T.ones_like(z_t) - z_t) * h_t_1 + z_t * c_t
        
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_1
        
        if self.use_drop:
            self.dropout_layer(h_t, self.use_noise, trng, self.dropout)
            
        #batch * 4
        #y_t = T.nnet.softmax(T.dot(h_t,U) + b)
               
        return h_t[:,::-1]                 
        
    def forward_sequence_batch_backward(self, x, mask, batch_size):
        h0 = T.zeros((batch_size, self.hidden_dim), dtype=theano.config.floatX)
       
        hs, _ = theano.scan(fn=self._step_batch_backward,
                            sequences=[x,mask],
                            outputs_info=[h0],
                            non_sequences= [self.W_zx_backward,self.W_zh_backward,
                                            self.W_rx_backward,self.W_rh_backward,                                            
                                            self.W_hx_backward,self.W_hh_backward,
                                           ])
        #hs[2] = theano.printing.Print('h2')(hs[2])  
                                  
        return hs
    
    def forward_batch(self, x, mask, batch_size):
        
        h_t = T.concatenate([self.forward_sequence_batch(x, mask, batch_size),
                             self.forward_sequence_batch_backward(x, mask, batch_size),
                             ],axis = 2)
        
        
        return h_t
                
    def __build_batch__(self):
        
        
        x = T.itensor3('x')
        index = T.matrix('index')
        y = T.imatrix('y')
        mask = T.matrix('mask')

        n_step = x.shape[0]
        batch_size = x.shape[1]
        
        emb = self.embedding[x.flatten()].reshape((n_step, batch_size, self.word_dim))
        
        h_t = self.forward_batch(emb, mask, batch_size)
        #h_t = theano.printing.Print('h_t',attrs=['shape'])(h_t)
        
        #h_t = h_t.reshape([n_step * batch_size, self.hidden_dim])[index.reshape([n_step * batch_size])]
        
        proj = T.nnet.softmax((T.dot(h_t,self.U) + self.b).reshape([n_step * batch_size, self.y_dim]))\
                                                          .reshape([n_step , batch_size, self.y_dim])                                       
        #proj = theano.printing.Print('proj')(proj)
        
        pred = T.argmax(proj, axis=2)
        
        #pred = theano.printing.Print('pred')(pred)
        proj = proj.reshape([n_step * batch_size, self.y_dim])
        
        regularization = 0.0
        for params in self.params:
            regularization = regularization +(T.sum(params ** 2))**0.5                                    
    
        neg_log_likelihoods = T.log(proj[T.arange(n_step * batch_size), y.flatten()]).reshape([n_step,batch_size]) * index * 1.0           
        loss = -T.mean(T.sum(neg_log_likelihoods,axis=0)) + (self.Regularization / 2) * regularization

        # SGD
        learning_rate = T.scalar('learning_rate')
        ada_optimizer = AdaDeltaOptimizer(lr=learning_rate, norm_lim=-1)
        except_norm_list = []
        updates = ada_optimizer.get_update(loss, self.params, except_norm_list)

        self.train_batch = theano.function([x, mask, y, index, learning_rate],[],
                        updates = updates)

        self.ce_loss = theano.function([x, mask, y, index], loss)
        self.predict = theano.function([x ,mask], pred)
        return self.use_noise
    
    def save_model(self, filename):
        from six.moves import cPickle
        with open(filename, 'wb') as fout:
            for param in self.params:
                cPickle.dump(param.get_value(), fout, protocol = -1)

    def load_model(self, filename):
        from six.moves import cPickle
        with open(filename, 'rb') as fin:
            for param in self.params:
                param.set_value(cPickle.load(fin))
    
    #dropout_layer
    def dropout_layer(self, state_before, use_noise, trng, dropout_rate):
        proj = T.switch(use_noise,
                        (state_before *
                         trng.binomial(state_before.shape,
                                        p=dropout_rate, n=1,
                                        dtype=state_before.dtype)),
                         state_before * (1-dropout_rate))
        return proj




















