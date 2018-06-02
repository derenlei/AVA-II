
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
    inputs = fixed_padding(inputs, kernel_size)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer())

def resnetUnit(input, filters, kernel_size, strides, bottleneck, training):
  """A resnet50 unit with two branches."""
  shortcut = input
  if bottleneck:
    shortcut = conv2d_fixed_padding(
      inputs=shortcut, filters=filters[3], strides=strides[3], kernel_size=kernel_size[3])
    shortcut = batch_norm(shortcut, training)
  
  input = conv2d_fixed_padding(
      inputs=input, filters=filters[0], strides=strides[0], kernel_size=kernel_size[0])
  input = batch_norm(input, training)
  input = conv2d_fixed_padding(
      inputs=input, filters=filters[1], strides=strides[1], kernel_size=kernel_size[1])
  input = batch_norm(input, training)
  input = conv2d_fixed_padding(
      inputs=input, filters=filters[2], strides=strides[2], kernel_size=kernel_size[2])
  input = batch_norm(input, training)
  
  return tf.nn.relu(input + shortcut)
  

def resnet50(features, labels, mode):
  training = (mode == tf.estimator.ModeKeys.TRAIN)

  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  #input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])


  # Convolutional Layer #1
  conv1 = conv2d_fixed_padding(
      inputs=input_layer, filters=64, strides=2, kernel_size=7)
  conv1_norm = batch_norm(conv1, training)
  conv1_relu = tf.nn.relu(conv1_norm)
  pool1 = tf.layers.max_pooling2d(
    inputs=conv1_relu, pool_size=3, strides=2, padding='SAME',
    data_format='channels_last')
  
  kernel_size = [1, 3, 1, 1]
  filters_2 = [64, 64, 256, 256]
  strides_2 = [1, 1, 1, 1]
  
  #res2a
  res2a_relu = resnetUnit(pool1, filters_2, kernel_size, strides_2,
                          bottleneck = True, training = training)
  res2b_relu = resnetUnit(res2a_relu, filters_2, kernel_size, strides_2,
                          bottleneck = False, training = training)
  res2c_relu = resnetUnit(res2b_relu, filters_2, kernel_size, strides_2,
                          bottleneck = False, training = training)
  
  filters_3 = [128, 128, 512, 512]
  strides_3 = [2, 1, 1, 2]
  res3a_relu = resnetUnit(res2c_relu, filters_3, kernel_size, strides_3,
                          bottleneck = True, training = training)
  res3b_relu = resnetUnit(res3a_relu, filters_3, kernel_size, strides_2,
                          bottleneck = False, training = training)
  res3c_relu = resnetUnit(res3b_relu, filters_3, kernel_size, strides_2,
                          bottleneck = False, training = training)
  res3d_relu = resnetUnit(res3c_relu, filters_3, kernel_size, strides_2,
                          bottleneck = False, training = training)
  
  filters_4 = [256, 256, 1024, 1024]
  res4a_relu = resnetUnit(res3d_relu, filters_4, kernel_size, strides_3,
                          bottleneck = True, training = training)
  res4b_relu = resnetUnit(res4a_relu, filters_4, kernel_size, strides_2,
                          bottleneck = False, training = training)  
  res4c_relu = resnetUnit(res4b_relu, filters_4, kernel_size, strides_2,
                          bottleneck = False, training = training)  
  res4d_relu = resnetUnit(res4c_relu, filters_4, kernel_size, strides_2,
                          bottleneck = False, training = training)  
  res4e_relu = resnetUnit(res4d_relu, filters_4, kernel_size, strides_2,
                          bottleneck = False, training = training)  
  res4f_relu = resnetUnit(res4e_relu, filters_4, kernel_size, strides_2,
                          bottleneck = False, training = training) 
  
  filters_5 = [512, 512, 2048, 2048]
  res5a_relu = resnetUnit(res4f_relu, filters_5, kernel_size, strides_3,
                          bottleneck = True, training = training)
  res5b_relu = resnetUnit(res5a_relu, filters_5, kernel_size, strides_2,
                          bottleneck = False, training = training) 
  res5c_relu = resnetUnit(res5b_relu, filters_5, kernel_size, strides_2,
                          bottleneck = False, training = training) 

  # ResNet does an Average Pooling layer over pool_size.
  # Do a reduce_mean because it performs better than AveragePooling2D.
  axes = [1, 2]
  inputs = tf.reduce_mean(res5c_relu, axes, keepdims=True)
  inputs = tf.reshape(inputs, [-1, 1*1*2048])
  
  dense = tf.layers.dense(inputs=inputs, units=1000, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=training)

  logits = tf.layers.dense(inputs=dropout, units=10)
  #logits = tf.layers.dense(inputs=dropout, units=14)
  
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
  tf.summary.scalar('Loss', loss)
  accuracy = tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  
  tf.summary.scalar('Accuracy', accuracy[1])

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
    	train_op = optimizer.minimize(
        	loss=loss,
        	global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = { "accuracy": accuracy }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  config = tf.estimator.RunConfig(session_config = session_config)

  mnist_classifier = tf.estimator.Estimator(
      model_fn=resnet50, model_dir="test2",
      config = config)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=1,
      shuffle=True)
  """
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=1500,
      hooks=[logging_hook])
  """
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
 
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=3000)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
  tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

 # eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
 # print(eval_results)
 
if __name__ == "__main__":
  tf.app.run()

