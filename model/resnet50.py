# model structure http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
# https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm(inputs, training):
  return tf.layers.batch_normalization(
      inputs=inputs, axis=3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size):
"""Pads the input along the spatial dimensions independently of input size."""
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides):
  """Strided 2-D convolution with explicit padding."""
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())


def resnet50(features, labels, mode):
  training = (mode == tf.estimator.ModeKeys.TRAIN)

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])

  # Convolutional Layer #1
  conv1 = conv2d_fixed_padding(
      inputs=input_layer, filters=64, strides=2, kernel_size=7)
  conv1_norm = batch_norm(conv1, training)
  conv1_relu = tf.nn.relu(norm1)
  pool1 = tf.layers.max_pooling2d(
    inputs=conv1_relu, pool_size=3, strides=2, padding='SAME',
    data_format='channels_last')
  
  #res2a
  res2a_branch1 = conv2d_fixed_padding(
      inputs=pool1, filters=256, strides=1, kernel_size=1)
  norm2a_branch1 = batch_norm(res2a_branch1, training)
  
  res2a_branch2a = conv2d_fixed_padding(
      inputs=pool1, filters=64, strides=1, kernel_size=1)
  norm2a_branch2a = batch_norm(res2a_branch2a, training)
  res2a_branch2b = conv2d_fixed_padding(
      inputs=norm2a_branch2a, filters=64, strides=1, kernel_size=3)
  norm2a_branch2b = batch_norm(res2a_branch2b, training)
  res2a_branch2c = conv2d_fixed_padding(
      inputs=norm2a_branch2b, filters=256, strides=1, kernel_size=1)
  norm2a_branch2c = batch_norm(res2a_branch2c, training)

  res2a = norm2a_branch2c + norm2a_branch1
  res2a_relu = tf.nn.relu(res2a)
  

"""
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)
"""
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
