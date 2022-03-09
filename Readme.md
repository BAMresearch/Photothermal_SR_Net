# Photothermal-SR-Net
### Underlying Problem: 
Find <img src="https://render.githubusercontent.com/render/math?math=x^m"> s.t. <img src="https://render.githubusercontent.com/render/math?math=t^m = \phi \ast x^m">, <img src="https://render.githubusercontent.com/render/math?math=m=1,\dots,N_{meas}"> where <img src="https://render.githubusercontent.com/render/math?math=t^m"> are measurements obtained from an IR camera and <img src="https://render.githubusercontent.com/render/math?math=\phi"> represents  the  discrete equivalent  of  the  fundamental  solution  of  the  heat  diffusion equation. The goal is to reconstruct the defect pattern <img src="https://render.githubusercontent.com/render/math?math=a"> in <img src="https://render.githubusercontent.com/render/math?math=x^m">.
We present deep unfolding for Block-ISTA type algorithms, to solve the least square problem 
<img src="https://render.githubusercontent.com/render/math?math=\min_{X} \sum_{m=1}^{N_{meas}} \sum_{k=1}^{N_r}| (\phi \ast x^m)[k] - t^m[k]|^2  %2B  \lambda \|X\|_{2,1}">
with 
<img src="https://render.githubusercontent.com/render/math?math=\|X\|_{2,1} = \sum_{k=1}^{N_r} \sqrt{\sum_{m=1}^{N_{\text{meas}}} |\mathbf{\mathbf{x}}_\text{reduc}^m[k]|^2}">
Block ISTA is an iterative algorithm and consists of the following two steps :
1. Gradient step (obtained by least square term)
2. Proximal Operator (obtained by regularization)

With deep unfolding we train certain parameters inside this algorithm to increase convergence speed and bypass the empiricall choice of the regularization parameters.

If Fast- is used, an additional step to increase convergence speed is added.
   For every network, the pretrained models for all trainings for T=6 Layers are provided.

## Provided Validation Data and Pretrained Models
   To demonstrate the performance, we provide the data with the point spread function of the underlying problem, a synthetic generated defect pattern, the computed "raw data" Y obtained by convolution of the PSF and the defect pattern in a matrix.mat. An additional noise np.random.normal(0, math.sqrt(prob.noise_ var), (450, 1280, 147) is added to the synthetically generated data. 
We could not provide the real measurement presented in the paper due to copyright issues.

## Run Code
0. To test the code, we provided all dependencies inside the dependencies.txt and a conda environment environment.yml. We highly recommend to use a conda virtual environment to install the dependencies. 

1. Open run.py and select which kind of network (with or without ReLU step after the
Gradient step, untied or tied) and how many layers you want to test and run the script by changing WithReLU={False,True} and T={1,..6}.
The pretrained networks with T=6 layers are provided. The difference between tied and untied is the following:
* tied: we learn the regularization parameters of each layer
* untied: we learn the regularization parameters and also the weight matrices of the gradient for each layer 
(for more information contact @janhauffen, @samimahmadi or see https://arxiv.org/abs/2012.03547).
2. Start the training with 
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
