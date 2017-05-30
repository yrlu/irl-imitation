"""Utility functions for tensorflow"""
import tensorflow as tf
import numpy as np


def max_pool(x, k_sz=[2, 2]):
  """max pooling layer wrapper
  Args
    x:      4d tensor [batch, height, width, channels]
    k_sz:   The size of the window for each dimension of the input tensor
  Returns
    a max pooling layer
  """
  return tf.nn.max_pool(
      x, ksize=[
          1, k_sz[0], k_sz[1], 1], strides=[
          1, k_sz[0], k_sz[1], 1], padding='SAME')


def conv2d(x, n_kernel, k_sz, stride=1):
  """convolutional layer with relu activation wrapper
  Args:
    x:          4d tensor [batch, height, width, channels]
    n_kernel:   number of kernels (output size)
    k_sz:       2d array, kernel size. e.g. [8,8]
    stride:     stride
  Returns
    a conv2d layer
  """
  W = tf.Variable(tf.random_normal([k_sz[0], k_sz[1], int(x.get_shape()[3]), n_kernel]))
  b = tf.Variable(tf.random_normal([n_kernel]))
  # - strides[0] and strides[1] must be 1
  # - padding can be 'VALID'(without padding) or 'SAME'(zero padding)
  #     - http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
  conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
  conv = tf.nn.bias_add(conv, b)  # add bias term
  # rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  return tf.nn.relu(conv)


def fc(x, n_output, scope="fc", activation_fn=None, initializer=None):
  """fully connected layer with relu activation wrapper
  Args
    x:          2d tensor [batch, n_input]
    n_output    output size
  """
  with tf.variable_scope(scope):
    if initializer is None:
      # default initialization
      W = tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
      b = tf.Variable(tf.random_normal([n_output]))
    else:
      W = tf.get_variable("W", shape=[int(x.get_shape()[1]), n_output], initializer=initializer)
      b = tf.get_variable("b", shape=[n_output],
                          initializer=tf.constant_initializer(.0, dtype=tf.float32))
    fc1 = tf.add(tf.matmul(x, W), b)
    if not activation_fn is None:
      fc1 = activation_fn(fc1)
  return fc1


def flatten(x):
  """flatten a 4d tensor into 2d
  Args
    x:          4d tensor [batch, height, width, channels]
  Returns a flattened 2d tensor
  """
  return tf.reshape(x, [-1, int(x.get_shape()[1] * x.get_shape()[2] * x.get_shape()[3])])


def update_target_graph(from_scope, to_scope):
  from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
  to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

  op_holder = []
  for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign(from_var))
  return op_holder


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
  def _initializer(shape, dtype=None, partition_info=None):
      out = np.random.randn(*shape).astype(np.float32)
      out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
      return tf.constant(out)
  return _initializer
