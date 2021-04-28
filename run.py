from __future__ import division
from __future__ import print_function
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
import matplotlib.pyplot as plt
np.random.seed(1) # numpy is good about making repeatable output
tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

T = 6     # the number of layers
Net = 'LBFISTA' #possible: 'LBElastic_Net', 'LBFastElastic_Net', 'LBISTA', 'LBFISTA'

WithReLU = True #Decide if you want to train/test with or without an ReLU after the gradient step
Untied = True   

if WithReLU == True:
    if Untied == False:
        path = 'WithReLU/tied'
    else:
        path = 'WithReLU/untied'
else:
    if Untied == False:
        path = 'WithoutReLU/tied'
    else:
        path = 'WithoutReLU/untied'

sys.path.insert(0, path)

#import the networks
import problem, network, train

prob = problem.thermoprob2(L=1280, B=147, MC=150, pnz=.01, SNR_dB=8, DB = 10, abs_hi = 1)
                            # Parameters: L Number of x-Dimension, B Number of Measurments, pnz possibility of nonzero blocks,
                            # DB Defect width and abs_hi the value of the highest absorption coefficient
if Net == 'LBElastic_Net':
    #generating the network w.r.t. the underlying iterative method
    layers = network.build_LBelastic_net(prob,T=T,initial_lambda=0.1)
    # setup the training stages
    training_stages = train.setup_training(layers,prob,trinit=1e-3, refinements=(.5,))
    # and do the training
    sess = train.do_training(training_stages,prob,path+'/trained_networks/LBElastic_net_ThermoProbT'+str(T)+'.npz', maxit = 5000)
if Net == 'LBFastElastic_Net':
    layers = network.build_LBFastelastic_net(prob, T=T, initial_lambda=0.1)
    training_stages = train.setup_training(layers, prob, trinit=1e-3, refinements=(.5,))
    sess = train.do_training(training_stages, prob,path+'/trained_networks/LBFAST_net_ThermoProbT' + str(T)+'.npz',maxit=5000)
if Net == 'LBISTA':
    layers = network.build_LBISTA(prob, T=T, initial_lambda=0.1)
    training_stages = train.setup_training(layers, prob, trinit=1e-3, refinements=(.5,))
    sess = train.do_training(training_stages, prob, path+'/trained_networks/LBISTA_ThermoProbT' + str(T)+'.npz',maxit=5000)
if Net == 'LBFISTA':
    layers = network.build_LBFISTA(prob, T=T, initial_lambda=0.1)
    training_stages = train.setup_training(layers, prob, trinit=1e-3, refinements=(.5,))
    sess = train.do_training(training_stages, prob, path+'/trained_networks/LBFISTA_ThermoProbT' + str(T)+'.npz',maxit=5000)

"""# Evaluating"""

Data = sio.loadmat('matrix.mat')
a = Data.get('a_true')          # this is the true defect pattern
Y=problem.newidct(np.multiply(prob.A, problem.newdct(a))) #generating measurments
a_2D = np.zeros((450, 1280))    # generating a 2 dimensional matrix of the defect pattern
Y_3D = np.zeros((450,1280,147))
a_2D[0:450, ] = a[:, 0]         # dim: N_y, N_x
Y_3D[0:-1,:,0:-1]=Y             #creating 3D image of measurments

Y_3D = Y_3D + np.random.normal(0, math.sqrt(prob.noise_var), (450, 1280, 147))
def plot(X, label):
    X = sum(X.T)

    X = abs(X) * np.max(abs(X)) ** (-1)

    fig = plt.plot(X, label=label)
    plt.show()

def nmse(xhat, x):
    normx = la.norm(x, ord='fro') ** 2
    return la.norm((xhat - x), ord='fro') ** 2 / normx

def nmse_db(xhat, x):
    return 10 * np.log10(nmse(xhat, x))

def implot(X):
    Xplot = np.zeros((X.shape[1], X.shape[2]))
    for k in range(X.shape[3]):
        Xplot = Xplot + X[0, :, :, k]

    fig = plt.imshow(Xplot)
    plt.show()

binning = False     # binning won't change the result in this case,
                    # because our Y is so 'pretty'
if binning:
    N = [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 30, 45, 50, 75, 90, 150, 225, 450]
    error = np.zeros(len(N))
    for i in range(len(N)):
        Nbin = N[i]
        Y = Y_3D

        Ymean = np.zeros((int(Y.shape[0] / Nbin), Y.shape[1], Y.shape[2]))
        k = 0
        for j in range(0, Y.shape[0], Nbin):
            Ymean[k, :] = np.mean(Y[j:j + Nbin, :, :], axis=0)
            k = k + 1

        Y = Ymean
        Y = Y[np.newaxis, :]  # Dim: N_batch, N_y, N_x, N_meas
        X = np.zeros((1, 450, 1280, 147))
        Xplots = np.zeros((T, 450, 1280))

        for j in range(0, Y.shape[1]):  #
            xhat = sess.run(layers[T - 1][1], feed_dict={prob.y_: Y[:, j, :, :], prob.x_: X[:, j, :, :]})
            X[:, j * Nbin:j * Nbin + Nbin + 1, :, :] = xhat

        Xplot = np.zeros((X.shape[1], X.shape[2]))
        for k in range(X.shape[3]):
            Xplot = Xplot + X[0, :, :, k]
else:
    Nbin = 1
    Y = Y_3D[np.newaxis, :]  # Dim: N_batch, N_y, N_x, N_meas
    X = np.zeros((1, 450, 1280, 147))
    Xplots = np.zeros((T, 450, 1280))

    for j in range(0, Y.shape[1]):  #apply network to each pixel row in the y-dimension
        xhat = sess.run(layers[T - 1][1], feed_dict={prob.y_: Y[:, j, :, :], prob.x_: X[:, j, :, :]})
        X[:, j * Nbin:j * Nbin + Nbin + 1, :, :] = xhat

    Xplot = np.zeros((X.shape[1], X.shape[2]))
    for k in range(X.shape[3]):
        Xplot = Xplot + X[0, :, :, k]

fig, axes = plt.subplots(3, 1)
fig.suptitle('Evaluating')
axes[0].imshow(a_2D)
axes[0].set_title('defect pattern a')
axes[1].imshow(Y_3D[:,:,0])
axes[1].set_title('raw data')
axes[2].imshow(X[0,:,:,0])
axes[2].set_title(Net)
plt.show()
