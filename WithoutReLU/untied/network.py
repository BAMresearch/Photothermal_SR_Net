from __future__ import division
from __future__ import print_function
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# %tensorflow_version 1.14
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy.linalg as la
import scipy.io as sio
import math
import sys
import time
import pdb

import keras.backend as K

def newtfdct(X_):
    return tf.transpose(tf.signal.dct(tf.transpose(X_, [0, 2, 1]), norm='ortho'), [0, 2, 1])
def newtfidct(X_):
    return tf.transpose(tf.signal.idct(tf.transpose(X_, [0, 2, 1]), norm='ortho'), [0, 2, 1])

def block_soft_threshold(X_, al_):
    #pdb.set_trace()
    al_ = tf.maximum(al_, 0)
    r_ = tf.sqrt(tf.reduce_sum((X_ ** 2), 2))
    r_ = tf.maximum(.0, 1 - tf.math.divide_no_nan(al_ , r_))
    r_ = tf.expand_dims(r_, 2)
    return tf.multiply(r_, X_)

def block_soft_threshold_elastic_net(X_, al_, la_):
    #pdb.set_trace()
    al_ = tf.maximum(al_, 0)
    r_ = tf.sqrt(tf.reduce_sum((X_ ** 2), 2))
    r_ = tf.maximum(.0, 1 - tf.math.divide_no_nan(al_ , r_))
    r_ = tf.expand_dims(r_, 2)
    return tf.multiply(r_, X_)*(1+la_)**(-1)


def build_LBISTA(prob, T, initial_lambda=.1):
    """
    Builds a LBISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN

    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block ISTA

    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LBISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft = block_soft_threshold
    layers = []
    A = prob.A
    M, N = A.shape
    e = np.ones((M, N))
    B = A * 2 / (1.0 * np.sqrt(2))
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
    S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_0')

    initial_lambda = np.array(initial_lambda).astype(np.float32)

    lam0_ = tf.Variable(initial_lambda, name='lam_0')
    xhat_ = blocksoft(By_, lam0_)
    layers.append(('LBISTA T=1', xhat_, (lam0_, B_, S_)))
    for t in range(1, T):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        B_ = tf.Variable(B, dtype=tf.float32, name='B_{0}'.format(t))
        By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
        S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_{0}'.format(t))
        xhat_ = blocksoft(newtfidct(tf.multiply(S_, newtfdct(xhat_))) + By_, lam_)
        layers.append(('LBISTA T=' + str(t + 1), xhat_, (lam_, B_, S_)))

    return layers

def build_LBFISTA(prob,T,initial_lambda=.1):
    """
    Builds a LBFISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block FISTA

    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LBFISTA T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    blocksoft=block_soft_threshold
    layers = []
    A = prob.A
    M,N = A.shape
    B = A*2 / (1.0 * np.sqrt(2))
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
    e = np.ones((M, N))
    S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_0')
    initial_lambda = np.array(initial_lambda).astype(np.float32)

    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = blocksoft( By_, lam0_)
    tk = (1+np.sqrt(1+4*1**2))*2**(-1)
    z_ = xhat_
    layers.append( ('LBFISTA T=1',xhat_, (lam0_, B_, S_) ) )
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        B_ = tf.Variable(B, dtype=tf.float32, name='B_{0}'.format(t))
        By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
        S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_{0}'.format(t))
        xhat_ = blocksoft(newtfidct(tf.multiply(S_, newtfdct(z_))) + By_, lam_)
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBFISTA T='+str(t+1),xhat_,(lam_, B_, S_)) )

    return layers

def build_LBelastic_net(prob,T,initial_lambda=.1):
    """
    Builds a LBElastic Net network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block Elastic Net

    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LBelastic_netT=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta=block_soft_threshold_elastic_net
    layers = []
    A = prob.A
    M,N = A.shape
    B = A*2 / (1.0 * np.sqrt(2))
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0')
    By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
    e = np.ones((M, N))
    S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_0')
    
    initial_lambda = np.array(initial_lambda).astype(np.float32)
    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_)
    layers.append( ('LBelastic_net T=1',xhat_, (al0_, lam0_, B_, S_) ) )
    #pdb.set_trace()
    for t in range(1,T):
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        B_ = tf.Variable(B, dtype=tf.float32, name='B_{0}'.format(t))
        By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
        S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_{0}'.format(t))
        xhat_ = eta(newtfidct(tf.multiply(S_, newtfdct(xhat_))) + By_, al_, lam_ )
        layers.append( ('LBelastic_net T='+str(t+1),xhat_,(al_, lam_, B_, S_)) )

    return layers

def build_LBFastelastic_net(prob,T,initial_lambda=.1):
    """
    Builds a LBFastElastic_net network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    prob            - is a TFGenerator which contains problem parameters and def of how to generate training data
    initial_lambda  - could be some parameter of Block Fast Elastic Net

    Return a list of layer info (name,xhat_,newvars)
     name : description, e.g. 'LBFelastic_net T=1'
     xhat_ : that which approximates x_ at some point in the algorithm
     newvars : a tuple of layer-specific trainable variables
    """
    eta=block_soft_threshold_elastic_net
    layers = []
    A = prob.A
    M,N = A.shape
    B = A*2 / (1.0 * np.sqrt(2))
    B_ = tf.Variable(B, dtype=tf.float32, name='B_0')
    By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
    e = np.ones((M,N))
    e[0]=1
    S_ = tf.Variable( e - np.multiply(B,A),dtype=tf.float32,name='S_0')

    initial_lambda = np.array(initial_lambda).astype(np.float32)

    al0_ = tf.Variable( initial_lambda,name='al_0')
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, al0_, lam0_)
    xhat_ = tf.maximum(xhat_, .0)
    tk = (1+np.sqrt(1+4*1**2))*2**(-1)
    z_ = xhat_
    layers.append( ('LBFelastic_net T=1',xhat_, (al0_, lam0_, B_, S_) ) )
    for t in range(1,T):
        t_prev = tk
        xhat_prev_ = xhat_ 
        al_ = tf.Variable( initial_lambda,name='al_{0}'.format(t) )
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        B_ = tf.Variable(B, dtype=tf.float32, name='B_{0}'.format(t))
        By_ = newtfidct(tf.multiply(B_, newtfdct(prob.y_)))
        S_ = tf.Variable(e - np.multiply(B, A), dtype=tf.float32, name='S_{0}'.format(t))
        xhat_ = eta(newtfidct(tf.multiply(S_, newtfdct(z_))) + By_, al_, lam_ )
        tk = (1+np.sqrt(1+4*t_prev**2))*2**(-1)
        z_ = xhat_ + (t_prev-1)*(tk)**(-1)*(xhat_-xhat_prev_)
        layers.append( ('LBFelastic_net T='+str(t+1),xhat_,(al_, lam_, B_, S_) ) )

    return layers
