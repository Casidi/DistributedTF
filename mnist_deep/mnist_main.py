import tensorflow as tf
import numpy as np
import os
import csv
import gzip

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
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
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
        if params['opt_case']['optimizer'] == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['opt_case']['lr'])
        elif params['opt_case']['optimizer'] == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['opt_case']['lr'])
        elif params['opt_case']['optimizer'] == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=params['opt_case']['lr'])
        elif params['opt_case']['optimizer'] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=params['opt_case']['lr'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
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

def main(hp, model_id, save_base_dir, data_dir, train_epochs):
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
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=100, # this is for debugging
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    filename = os.path.join(save_dir,'learning_curve.csv')
    file_exists = os.path.isfile(filename)
    fields=['global_step','eval_accuracy']
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({'global_step': eval_results['global_step'], \
            'eval_accuracy': eval_results['accuracy']})

    return eval_results['global_step'], eval_results['accuracy']
