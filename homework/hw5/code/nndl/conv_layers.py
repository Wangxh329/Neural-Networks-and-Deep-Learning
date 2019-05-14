import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  N, C, H, W = x.shape    # [N, 3, 32, 32]
  F, C, HH, WW = w.shape  # [32, 3, 7, 7]

  padded_x = (np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant'))  # [N, 3, 38, 38]
  out_height = np.int(((H + 2 * pad - HH) / stride) + 1)  # 32
  out_width = np.int(((W + 2 * pad - WW) / stride) + 1)  # 32
  out = np.zeros([N, F, out_height, out_width])  # [N, 32, 32, 32]

  for img in range(N):  # for each image, do convolutional process
    for kernal in range(F):  # for each channel, there are 3 W (7x7) and 1 b (scalar), linear sum together
      for row in range(out_height):  # from top to bottom
        for col in range(out_width):  # from left to right
          # each kernal has 3 W (7x7), for each elements in W, multiply it with corresponding elements in original graph
          # then add up these 49 numbers together -> 1 scalar
          # then add up three scalar and 1 bias, as the current position's convolutional result
          out[img, kernal, row, col] = np.sum(w[kernal, ...] * \
                                              padded_x[img, :, row*stride:row*stride+HH, col*stride:col*stride+WW]) + b[kernal]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #

  _, _, H, W = x.shape  # [N, 3, 32, 32]
  dx_temp = np.zeros_like(xpad)  # initial to all zeros
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # Calculate dB.
  for kernal in range(F):
    db[kernal] += np.sum(dout[:, kernal, :, :])  # sum all N img's kernal -> [32, 32], then sum all 32x32 elements -> 1 scalar
  
  # Calculate dw.
  for img in range(N):  # for each image
    for kernal in range(F):  # for each kernal
      for row in range(out_height):  # from top to bottom
        for col in range(out_width):  # from left to right
          dw[kernal, ...] += dout[img, kernal, row, col] * xpad[img, :, row*stride:row*stride+f_height, col*stride:col*stride+f_width]
  
  # Calculate dx.
  for img in range(N):  # for each image
    for kernal in range(F):  # for each kernal
      for row in range(out_height):  # from top to bottom
        for col in range(out_width):  # from left to right
          dx_temp[img, :, row*stride:row*stride+f_height, col*stride:col*stride+f_width] += dout[img, kernal, row,col] * w[kernal, ...]
  
  dx = dx_temp[:, :, pad:H+pad, pad:W+pad]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #

  pool_height = pool_param.get('pool_height')
  pool_width = pool_param.get('pool_width')
  stride = pool_param.get('stride')
  N, C, H, W = x.shape   # [N, 3, 32, 32]

  out_height = np.int(((H - pool_height) / stride) + 1)  # calculate output height
  out_width = np.int(((W - pool_width) / stride) + 1)    # calculate output width
  out = np.zeros([N, C, out_height, out_width])

  for img in range(N): # for each image
    for channel in range(C): # for each channel
      for row in range(out_height): # from top to bottom
        for col in range(out_width): # from left to right
          out[img, channel, row, col] = np.max(x[img, channel, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width])

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  
  N, C, H, W = x.shape  # [N, 3, 32, 32]
  _, _, dout_height, dout_width = dout.shape
  dx = np.zeros_like(x)

  for img in range(N): # for each image
    for channel in range(C): # for each channel
      for row in range(dout_height): # from top to bottom
        for col in range(dout_width): # from left to right
          max_idx = np.argmax(x[img, channel, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width])
          max_position = np.unravel_index(max_idx, [pool_height, pool_width])
          dx[img, channel, row*stride:row*stride+pool_height, col*stride:col*stride+pool_width][max_position] = dout[img, channel, row, col]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  N, C, H, W = x.shape  # [N, 3, 32, 32]
  x_transpose = x.transpose(0, 2, 3, 1)
  x_reshape = np.reshape(x_transpose, (N*H*W, C)) # reshape to 2D to do batchnorm
  out_2d, cache = batchnorm_forward(x_reshape, gamma, beta, bn_param) 
  out = out_2d.reshape((N, H, W, C)).transpose(0, 3, 1, 2) # reshape back

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  dx = np.zeros_like(dout)
  N, C, H, W = dout.shape
  dout_transpose = dout.transpose((0, 2, 3, 1))
  dout_reshape = np.reshape(dout_transpose, (N*H*W, C)) # reshape to 2D to do batchnorm
  dx_2d, dgamma, dbeta = batchnorm_backward(dout_reshape, cache)
  dx = dx_2d.reshape((N, H, W, C)).transpose(0, 3, 1, 2) # reshape back

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta