import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    C, H, W = input_dim
    ## Initial 1st layer (conv): [32, 3, 7, 7] 
    ## there are 32 kernals in this layer, and each kernal has 3 dimensions (corresponding to original graph's channel RGB),
    ## and each kernal is 7x7
    stride = 1
    pad = (filter_size - 1) / 2
    out_conv_height = (H + 2 * pad - filter_size) / stride + 1
    out_conv_width = (W + 2 * pad - filter_size) / stride + 1
    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_size, filter_size])
    self.params['b1'] = np.zeros([num_filters]) # each filter (kernal) has a bias
    
    ## Initial 2nd layer (fc): after conv and max pooling, the weight and height are half, and channel changes from 3 to 32
    ## so for fully connected layer, the shape would be [N, 16x16x32].
    ## First, we will flatten the graph after conv from [32, 16, 16] to [32x16x16],
    ## then, do fc and reduce the dimension from 32x16x16 to 100
    out_pool_height = int((out_conv_height - 2) / 2 + 1)
    out_pool_width = int((out_conv_width - 2) / 2 + 1)
    self.params['W2'] = np.random.normal(0, weight_scale, [out_pool_height*out_pool_width*num_filters, hidden_dim])
    self.params['b2'] = np.zeros([hidden_dim])
    
    ## Initial 3rd layer (fc): keep reducing the dimension from 100 to 10 (cifar dataset has 10 classes)
    self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b3'] = np.zeros([num_classes])

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #
    
    ## forward: [N, 3, 32, 32] -> [N, 10]
    layer1_out, combined_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) # 1st layer: conv + relu + 2x2 max pool
    fc1_out, fc1_cache = affine_relu_forward(layer1_out, W2, b2) # 2nd layer: fc and relu
    scores, fc2_cache = affine_forward(fc1_out, W3, b3) # 3rd layer: fc

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #

    loss, dscores = softmax_loss(scores, y)  # 3rd layer: softmax
    loss += self.reg * 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

    ## backward: [N, 10] -> [N, 3, 32, 32]
    dx3, dw3, db3 = affine_backward(dscores, fc2_cache) # oppo 3rd layer: fc
    dx2, dw2, db2 = affine_relu_backward(dx3, fc1_cache) # oppo 2nd layer: fc
    dx1, dw1, db1 = conv_relu_pool_backward(dx2, combined_cache) # oppo 1st layer: conv + relu + pool

    grads['W3'], grads['b3'] = dw3 + self.reg * W3, db3
    grads['W2'], grads['b2'] = dw2 + self.reg * W2, db2
    grads['W1'], grads['b1'] = dw1 + self.reg * W1, db1

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
pass
