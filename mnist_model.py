from constants import get_hp_range_definition
import random
import numpy as np

import tensorflow as tf
import os
import csv
import gzip

from model_base import ModelBase

def initializer_func(initializer_name): # Xinyi add
  # Ref: https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
  # Noted: I like the normal myself because it allows more diversity in the training. 
  # But I have no empirical evidence or theoretical argument that it works better than uniform
  if initializer_name == 'glorot_normal':
      initializer = tf.glorot_normal_initializer()
  elif initializer_name == 'orthogonal':
      initializer = tf.orthogonal_initializer(gain=1.0)
  elif initializer_name == 'he_init':
      initializer = tf.keras.initializers.he_normal()
  else:
      initializer = None
      # Noted if None, tf.layers.conv2d and tf.layers.dense take glorot_uniform_initializer
  return initializer

def solver_func(hparams):
    opt_name = hparams['opt_case']['optimizer']
    learning_rate = hparams['opt_case']['lr']

    if opt_name == 'Momentum' or opt_name == 'RMSProp':
        momentum = hparams['opt_case']['momentum']
    if opt_name == 'RMSProp':
        grad_decay = hparams['opt_case']['grad_decay']
    
    if opt_name == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer( \
            learning_rate=learning_rate)
    elif opt_name == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer( \
            learning_rate=learning_rate)
    elif opt_name == 'Momentum':
        optimizer = tf.train.MomentumOptimizer( \
            learning_rate=learning_rate, \
            momentum=momentum)
    elif opt_name == 'Adam':
        optimizer = tf.train.AdamOptimizer( \
            learning_rate=learning_rate)
    elif opt_name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer( \
            learning_rate=learning_rate, \
            momentum=momentum, \
            decay=grad_decay)
    elif opt_name == 'gd':
        optimizer = tf.train.GradientDescentOptimizer( \
            learning_rate=learning_rate)
    else:
        raise RuntimeError('Hyper-parameter optimizer is wrong!')
    
    return optimizer

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=initializer_func(params['initializer']))

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=initializer_func(params['initializer']))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,
                kernel_initializer=initializer_func(params['initializer']))
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

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
        optimizer = solver_func(params)
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

def main(hp, model_id, save_base_dir, data_dir, train_epochs, epoch_index):
    save_dir = save_base_dir + str(model_id)

    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        train_data = np.frombuffer(file.read(), np.uint8, offset=16).astype(np.float32).reshape(-1,28*28)
    with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
        train_labels = np.frombuffer(file.read(), np.uint8, offset=8).astype(np.int32)
    with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'), 'rb') as file:
        eval_data = np.frombuffer(file.read(), np.uint8, offset=16).astype(np.float32).reshape(-1,28*28)
    with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'), 'rb') as file:
        eval_labels = np.frombuffer(file.read(), np.uint8, offset=8).astype(np.int32)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(session_config=session_config)
    mnist_classifier = tf.estimator.Estimator(
                            model_fn=cnn_model_fn,
                            model_dir=save_dir,
                            config=run_config,
                            params=hp)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=hp['batch_size'],
        num_epochs=train_epochs,
        shuffle=True)

    results_to_log = []
    for i in range(train_epochs):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=10, # this is for debugging
            hooks=[logging_hook])

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        results_to_log.append((eval_results['global_step'], eval_results['accuracy'], hp['opt_case']['optimizer'], hp['opt_case']['lr']))

    filename = os.path.join(save_dir,'learning_curve.csv')
    file_exists = os.path.isfile(filename)
    fields=['global_step','eval_accuracy', 'optimizer', 'lr']
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        
        for i in results_to_log:
            writer.writerow({'global_step': epoch_index, 'eval_accuracy': i[1], 'optimizer':i[2], 'lr':i[3]})

    return eval_results['global_step'], eval_results['accuracy']

class MNISTModel(ModelBase):
    def __init__(self, cluster_id, hparams, save_base_dir):
        super(MNISTModel, self).__init__(cluster_id, hparams, save_base_dir)

        # prevent the NaN error
        #hparams['opt_case']['lr'] /= 1000.0
        #if cluster_id == 0:
        #    hparams['opt_case']['lr'] = 10000.0
    
    def train(self, epoches_to_train, total_epochs):
        data_dir = '/home/K8S/dataset/mnist'
        step, self.accuracy = \
            main(self.hparams, self.cluster_id, self.save_base_dir, data_dir, epoches_to_train, self.epoches_trained)
        self.epoches_trained += 1
