import tensorflow as tf
import numpy as np
import resnet50_AVA
def main(unused_argv):
  eval_data=np.load('train_image_matrix.npy').astype(np.float32)
  #print('eval_data',eval_data.shape)
  eval_labels=np.load('train_labels.npy').astype(np.int64)
  eval_labels = eval_labels-1

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  #session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
  config = tf.estimator.RunConfig(session_config = session_config)
  mnist_classifier = tf.estimator.Estimator(
      model_fn=resnet50_AVA.resnet50, model_dir="model_resnet50_0.0001", config = config)
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  # Evaluate the model and print result

  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 0]
  index1_data = eval_data[eval_labels[:] == 0]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 1",label1_results)

  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 1]
  index1_data = eval_data[eval_labels[:] == 1]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=2,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 2",label1_results)

  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 2]
  index1_data = eval_data[eval_labels[:] == 2]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 3",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 3]
  index1_data = eval_data[eval_labels[:] == 3]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 4",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 4]
  index1_data = eval_data[eval_labels[:] == 4]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 5",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 5]
  index1_data = eval_data[eval_labels[:] == 5]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 6",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 6]
  index1_data = eval_data[eval_labels[:] == 6]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 7",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 7]
  index1_data = eval_data[eval_labels[:] == 7]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 8",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 8]
  index1_data = eval_data[eval_labels[:] == 8]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 9",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 9]
  index1_data = eval_data[eval_labels[:] == 9]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 10",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 10]
  index1_data = eval_data[eval_labels[:] == 10]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 11",label1_results)


  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 11]
  index1_data = eval_data[eval_labels[:] == 11]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 12",label1_results)
  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 12]
  index1_data = eval_data[eval_labels[:] == 12]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 12",label1_results)

  #eval_input_fn = evalInput_fn(eval_data,eval_labels
  index1_labels = eval_labels[eval_labels[:] == 13]
  index1_data = eval_data[eval_labels[:] == 13]
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": index1_data},
      y=index1_labels,
      num_epochs=1,
      shuffle=False)
  label1_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Result for label 14",label1_results)


if __name__ == "__main__":
  tf.app.run()
