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
from scipy.fftpack import idct, dct

def newdct(X):  # spalten weise durchgehen
    x = np.zeros(X.shape)
    for i in range(0, X.shape[1]):
        y = dct(X[:, i], norm='ortho')
        x[:, i] = y
    return x


def newidct(X):  # spalten weise durchgehen
    x = np.zeros(X.shape)
    for i in range(0, X.shape[1]):
        y = idct(X[:, i], norm='ortho')
        x[:, i] = y
    return x

def newbatchdct(X):
    x = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        x[i,:]=newdct(X[i, :])
    return x

def newbatchidct(X):
    x = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        x[i, :] = newidct(X[i, :])
    return x

def newtfdct(X_):
    return tf.transpose(tf.signal.dct(tf.transpose(X_, [0, 2, 1]), norm='ortho'), [0, 2, 1])
def newtfidct(X_):
    return tf.transpose(tf.signal.idct(tf.transpose(X_, [0, 2, 1]), norm='ortho'), [0, 2, 1])

class Generator(object):
    def __init__(self, A, **kwargs):
        self.A = A
        #M, N = A.shape
        #MC = 100
        vars(self).update(kwargs)
        self.x_ = tf.placeholder(tf.float32, (None, 1280, 147), name='x')
        self.y_ = tf.placeholder(tf.float32, (None, 1280, 147), name='y')


class TFGenerator(Generator):
    def __init__(self, **kwargs):
        Generator.__init__(self, **kwargs)

    def __call__(self, sess):
        "generates y,x pair for training"
        return sess.run((self.ygen_, self.xgen_))

def generating_abs_pattern(MC, L, B, pnz, breite):
    DB=breite
    active_blocks_val = np.zeros((MC, L))
    active_blocks_val[:, DB - 1:L - DB - 1] = (np.random.uniform(0, 1, (MC, L - 2 * DB)) < pnz).astype(np.float32)
    active_entries_val = np.repeat(active_blocks_val, B, axis=1)

    xval = np.multiply(active_entries_val, np.ones((MC, L * B)))
    xval = np.reshape(xval, (MC, L, B))
    #pdb.set_trace()
    for i in range(0, xval.shape[0]):  # batch
        # for t in range(0, xval.shape[1]): #t spalten, k Zeilen
        t = 0
        while t < xval.shape[1]:
            if (xval[i, t, 0] == 1):
                # pdb.set_trace()
                if t - DB >= 0:
                    if t + DB + 1 > xval.shape[1]:
                        xval[i, t - DB:-1,
                        :] = 1  # kritisch, da dann Absorptionslinie nicht so dick wie andere; dieser Fall sollte nicht auftreten
                    else:
                        xval[i, t - DB:t + DB + 1, :] = 1
                else:
                    if t + DB + 1 > xval.shape[1]:
                        xval[i, 0:-1,:] = 1  # kritisch, da dann Absorptionslinie nicht so dick wie andere; dieser Fall sollte nicht auftreten
                    else:
                        xval[i, 0:t + DB + 1, :] = 1
                t = t + DB + 1
            else:
                t = t + 1

    return xval

def thermoprob2(L=32, B=16, MC=1000, pnz=.01, SNR_dB=20, DB = 5, abs_hi = 0.9):
    # L is the number of pixels
    # B is the number of measurments
    # MC is the training batch size
    # pnz refers to the number of non zero blocks
    Amat = sio.loadmat('matrix.mat')
    A = Amat.get('A')

    A_ = tf.constant(A, name='A')
    prob = TFGenerator(A=A, A_=A_, kappa=None, SNR=SNR_dB)

    abs_lo = 1 - abs_hi

    Rfkt = np.zeros(A.shape)
    Rfkt[0:DB] = 5
    Rfkt = newdct(Rfkt)

    active_blocks_ = tf.to_float(tf.random_uniform((L-2*DB,1,MC))< pnz)
    active_blocks_ = tf.concat([tf.zeros((DB, 1, MC)), active_blocks_, tf.zeros((DB, 1, MC))], axis=0)

    ones_ = tf.ones([L,B,MC])
    product_ = tf.multiply(active_blocks_, ones_)

    xgen_ = tf.reshape(product_, [L*B,MC])
    xgen_ = tf.transpose(xgen_, [1, 0])
    xgen_ = tf.reshape(xgen_, [MC, L, B])

    xgen_ = tf.multiply(tf.constant(Rfkt, dtype=tf.float32), newtfdct(xgen_))
    xgen_ = newtfidct(xgen_)
    xgen_ = tf.where(xgen_ <= 0.04, tf.zeros(xgen_.shape), xgen_) # 0.4 occurs by convolution, it's not pretty, but it works...
    xgen_ = tf.where(xgen_ > 0.04, tf.ones(xgen_.shape), xgen_)
    xgen_ = tf.where(xgen_ >= 1, abs_hi*tf.ones(xgen_.shape), xgen_)
    xgen_ = tf.where(xgen_ <= 0, abs_lo*tf.ones(xgen_.shape), xgen_)

    prob.name = 'Thermo Problem'
    prob.L = L
    prob.B = B

    prob.pnz = pnz

    noise_var = pnz * 1 * math.pow(10., -SNR_dB / 10.)

    ygen_ = newtfidct(tf.multiply(A_, newtfdct(xgen_)))
    ygen_ = ygen_ + tf.random_normal((MC, L, B), stddev=math.sqrt(noise_var))
    print(noise_var)

    abs_pattern = generating_abs_pattern(MC, L, B, pnz, DB)

    xval=abs_pattern
    xval[xval == 1] = abs_hi
    xval[xval == 0] = abs_lo

    yval = newbatchidct(np.multiply(A, newbatchdct(xval)))
    yval = yval + np.random.normal(0, math.sqrt(noise_var), (MC, L, B))

    prob.xgen_ = xgen_
    prob.ygen_ = ygen_
    prob.xval = xval
    prob.yval = yval
    prob.noise_var = noise_var
    return prob
