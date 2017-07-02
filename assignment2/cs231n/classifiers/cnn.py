import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


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
               dtype=np.float32):
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
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    std = weight_scale
    input_c, input_h, input_w = input_dim
    self.params['W1'] = std * np.random.randn(num_filters, input_c, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    fc_input = num_filters * input_h * input_w / 4
    self.params['W2'] = std * np.random.randn(fc_input, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
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
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    fc_input, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out, cache2 = affine_relu_forward(fc_input, W2, b2)
    scores, cache3 = affine_forward(out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, ds = softmax_loss(scores, y)
    dx, grads['W3'], grads['b3'] = affine_backward(ds, cache3)
    grads['W3'] += reg * W3
    loss += 0.5 * reg * np.sum(W3 ** 2)
    dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache2)
    grads['W2'] += reg * W2
    loss += 0.5 * reg * np.sum(W2 ** 2)
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)
    grads['W1'] += reg * W1
    loss += 0.5 * reg * np.sum(W1 ** 2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class FourLayerConvNet(object):
  """
  A four-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - conv - relu - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
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
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    std = weight_scale
    C, H, W = input_dim
    for i in range(1, 3):
      self.params['W' + str(i)] = std * np.random.randn(num_filters, C, filter_size, filter_size)
      self.params['b' + str(i)] = np.zeros(num_filters)
      self.params['gamma' + str(i)] = np.ones(num_filters)
      self.params['beta' + str(i)] = np.zeros(num_filters)
      C = num_filters
    fc_input = num_filters * H * W / 4
    self.params['W3'] = std * np.random.randn(fc_input, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)
    self.params['W4'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['b4'] = np.zeros(num_classes)
    self.bn_params = [{'mode': 'train'} for i in range(3)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.params['W1'].shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = []
    out, cache1 = conv_forward_fast(X, W1, b1, conv_param)
    cache.append(cache1)
    out, cache1 = spatial_batchnorm_forward(out, gamma1, beta1, self.bn_params[0])
    cache.append(cache1)
    out, cache1 = relu_forward(out)
    cache.append(cache1)
    out, cache1 = max_pool_forward_fast(out, pool_param)
    cache.append(cache1)
    out, cache1 = conv_forward_fast(out, W2, b2, conv_param)
    cache.append(cache1)
    out, cache1 = spatial_batchnorm_forward(out, gamma2, beta2, self.bn_params[1])
    cache.append(cache1)
    out, cache1 = relu_forward(out)
    cache.append(cache1)
    out, cache1 = affine_forward(out, W3, b3)
    cache.append(cache1)
    out, cache1 = batchnorm_forward(out, gamma3, beta3, self.bn_params[2])
    cache.append(cache1)
    out, cache1 = relu_forward(out)
    cache.append(cache1)
    scores, last_cache = affine_forward(out, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    reg = self.reg
    loss, ds = softmax_loss(scores, y)
    dx, grads['W4'], grads['b4'] = affine_backward(ds, last_cache)
    grads['W4'] += reg * W4
    loss += 0.5 * reg * np.sum(W4 ** 2)

    dx = relu_backward(dx, cache.pop())
    dx, grads['gamma3'], grads['beta3'] = batchnorm_backward_alt(dx, cache.pop())
    dx, grads['W3'], grads['b3'] = affine_backward(dx, cache.pop())
    grads['W3'] += reg * W3
    loss += 0.5 * reg * np.sum(W3 ** 2)

    dx = relu_backward(dx, cache.pop())
    dx, grads['gamma2'], grads['beta2'] = spatial_batchnorm_backward(dx, cache.pop())
    dx, grads['W2'], grads['b2'] = conv_backward_fast(dx, cache.pop())
    grads['W2'] += reg * W2
    loss += 0.5 * reg * np.sum(W2 ** 2)

    dx = max_pool_backward_fast(dx, cache.pop())
    dx = relu_backward(dx, cache.pop())
    dx, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dx, cache.pop())
    dx, grads['W1'], grads['b1'] = conv_backward_fast(dx, cache.pop())
    grads['W1'] += reg * W1
    loss += 0.5 * reg * np.sum(W1 ** 2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
