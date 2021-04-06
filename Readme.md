# Photothermal-SR-Net
### Underlying Problem: 
Find X s.t. Y = P * X + \epsilon where P is a given
Point Spread Function.
We present deep unfolding for Block ISTA/FISTA and Block Elastic Net/Fast Elastic Net,
which solves the least square problem connected to the underlying problem with an
additional block sparsity inducing term (l_{2,1} regularization) respectively a
block sparsity inducing term and an additional Tikhonov regularization term (l_{2,1} + l_{2} regularization). These will be trained for X,Y with dimension N_{x}, N_{meas} and then applied to each pixel row of a given measurement Y. Block ISTA and Block Elastic Net use the following two steps in the iteration to approximate X:
1. Gradient step (obtained by least square term)
2. use Proximal Operator (obtained by regularization)

If Fast- is used, an additional step to increase convergence speed is added.
   For every network, the pretrained models for all trainings for T=6 Layers are provided.

## Provided Validation Data and Pretrained Models
   To demonstrate the performance, we provide the data with the point spread function of the underlying problem, a synthetic generated defect pattern, the computed "raw data" Y obtained by convolution of the PSF and the defect pattern in a matrix.mat. An additional noise np.random.normal(0, math.sqrt(prob.noise_ var), (450, 1280, 147) is added to the synthetically generated data. 
We could not provide the real measurement presented in the paper due to copyright issues.

## Run Code
0. To test the code, we provided all dependencies inside the dependencies.txt and a conda environment environment.yml. We highly recommend to use a conda virtual environment to install the dependencies. 

2. Decide if you want to test the networks with or without a ReLU step after the
Gradient step and with tied or untied training by changing the boolean value for the variable Untied={False,True}. The difference between tied and untied is the following:
* tied: we learn the regularization parameters of each layer
* untied: we learn the regularization parameters and also the weight matrices of the gradient for each layer.
(for more information contact me or see https://arxiv.org/abs/2012.03547)

2. Open run.py and select which kind of network (with or without ReLU, untied or tied) and how many layers you want to test and run the script by changing s WithReLU={False,True} and T={1,..6}.
The pretrained networks with T=6 layers are provided.
3. Start the training with 
```
python run.py
```
After building and training the network, the evaluation begins. For this purpose, we
will test the trained network on Y and plot the result. 
  Note: Also included in
this Part is the binning of pixel rows in the y-dimension, because the synthetically
generated "raw data" is prettier, i.e. does not have blurring in the upper and lower
part, this this will not have an effect on the solution but is provided for the sake
of completeness.
