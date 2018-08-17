import tensorflow as tf
from mnist_dataset import get_train_batch, get_test_data
import math

class MNISTDeepModel:
    def __init__(self, cluster_id, hparams):
        self.cluster_id = cluster_id
        self.hparams = hparams
        self.train_step = 0
        self.need_explore = False

        self.build_graph_from_hparams()
        self.train_log = []

    def train(self, num_steps):
        train_images, train_labels = get_train_batch(100)
        for i in range(num_steps):
            self.train_log.append((self.train_step, self.get_accuracy()))
            self.sess.run([self.train_op], 
                        feed_dict={self.x: train_images,
                         self.y_: train_labels, 
                         self.is_training: True,
                         self.keep_prob: self.hparams['dropout']})            
            self.train_step += 1
            

    def perturb_hparams_and_update_graph(self):
        #TODO: implement this function to get the exploring working
        self.build_graph_from_hparams()
        return

    def get_accuracy(self):
        test_images, test_labels = get_test_data()
        batch_size = 100
        images_to_run = test_images.shape[0]
        num_batches = 0
        total_accuracy = 0.0
        while images_to_run > 0:
            begin = test_images.shape[0] - images_to_run
            images_to_run -= 100
            num_batches += 1
            if begin+100 <= test_images.shape[0]:
                images = test_images[begin:begin+100]
                labels = test_labels[begin:begin+100]
            else:
                images = test_images[begin:]
                labels = test_labels[begin:]
            total_accuracy += self.sess.run(self.accuracy,
                feed_dict={self.x: images,
                        self.y_: labels, 
                        self.is_training: False,
                        self.keep_prob: 1.0}) # Should turn off dropout when do evaluation
        return total_accuracy / num_batches

    def get_values(self):
        return [self.cluster_id, self.get_accuracy(), self.sess.run(self.trainable_vars), self.hparams]

    def set_values(self, values):
        self.hparams = values[3]
        self.build_graph_from_hparams()

        for i in range(len(self.trainable_vars)):
            self.trainable_vars[i].load(values[2][i], self.sess)
    
    def build_graph_from_hparams(self):
        self.tf_graph = tf.Graph()
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.1
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.tf_graph, config=config)
        
        with self.tf_graph.as_default():
            """
            x: an input tensor with the dimensions (N_examples, 784), where 784 is the
            number of pixels in a standard MNIST image.
            """
            self.x = tf.placeholder(tf.float32, shape=(None, 28*28))
            self.y_ = tf.placeholder(tf.int32, [None])
            one_hot_y_ = tf.one_hot(self.y_, 10)

            self.is_training = tf.placeholder(tf.bool, shape=())            
            initializer = self.initializer_func(self.hparams)            
            regularizer = self.regularizer_func(self.hparams)
            
            # Reshape to use within a convolutional neural net.
            # Last dimension is for "features" - there is only one here, since images are
            # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
            with tf.name_scope('reshape'):
                x_image = tf.reshape(self.x, [-1, 28, 28, 1])
            
            # First convolutional layer - maps one grayscale image to 32 feature maps.
            with tf.name_scope('conv1'):
                net = tf.layers.conv2d(
                        inputs=x_image,
                        filters=32,
                        kernel_size=5,
                        kernel_initializer=initializer,
                        padding='same',
                        activation=None,
                        kernel_regularizer=regularizer,
                        name='conv1')
                '''net = tf.layers.batch_normalization(
                        net, 
                        training = self.is_training)'''
                net = tf.nn.relu(net)
            with tf.name_scope('pool1'):
                net = tf.layers.max_pooling2d(
                        inputs=net,
                        pool_size=[2,2],
                        strides=2,
                        name='pool1')
            
            # Second convolutional layer -- maps 32 feature maps to 64.
            with tf.name_scope('conv2'):
                net = tf.layers.conv2d(
                        inputs=net,
                        filters=64,
                        kernel_size=5,
                        kernel_initializer=initializer,
                        padding='same', 
                        activation=None,
                        kernel_regularizer=regularizer,
                        name='conv2')
                '''net = tf.layers.batch_normalization(
                        net, 
                        training = self.is_training)'''
                net = tf.nn.relu(net)
            # Second pooling layer.
            with tf.name_scope('pool2'):
                net = tf.layers.max_pooling2d(
                        inputs=net,
                        pool_size=[2,2], 
                        strides=2, 
                        name='pool2')
            
            # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
            # is down to 7x7x64 feature maps -- maps this to 1024 features.
            with tf.name_scope('fc1'):
                net = tf.layers.flatten(net, name='flatten')
                net = tf.layers.dense(
                        inputs=net, 
                        units=1024, 
                        activation=tf.nn.relu,
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        name='fc1')
            
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            with tf.name_scope('dropout'):
                self.keep_prob = tf.placeholder(tf.float32)
                net = tf.layers.dropout(net, rate=self.keep_prob, training=self.is_training)                
            
            # Map the 1024 features to 10 classes, one for each digit
            with tf.name_scope('fc2'):
                y_conv = tf.layers.dense(
                        inputs=net, 
                        units=10,  
                        kernel_initializer=initializer,
                        kernel_regularizer=regularizer,
                        name='logits')

            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y_, logits=y_conv))

            self.train_op, learning_rate = self.solver_func(self.hparams, self.loss, 10)
            prediction = tf.argmax(y_conv, 1)
            equality = tf.equal(prediction, tf.argmax(one_hot_y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))         

            self.trainable_vars = tf.trainable_variables()
            self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def initializer_func(self, hparams):
        # Ref: https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
        # Noted: I like the normal myself because it allows more diversity in the training. 
        # But I have no empirical evidence or theoretical argument that it works better than uniform
        if hparams['initializer'] == 'glorot_normal':
            initializer = tf.glorot_normal_initializer()
        elif hparams['initializer'] == 'orthogonal':
            initializer = tf.orthogonal_initializer(gain=1.0)
        elif hparams['initializer'] == 'he_init':
            initializer = tf.keras.initializers.he_normal()
        else:
            initializer = None
            # Noted if None, tf.layers.conv2d and tf.layers.dense take glorot_uniform_initializer
        return initializer
        
        
    def regularizer_func(self, hparams):
        if hparams['regularizer'] == 'l1_regularizer':
            regularizer = tf.contrib.layers.l1_regularizer(scale=0.1)
        elif hparams['regularizer'] == 'l2_regularizer':
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        elif hparams['regularizer'] == 'l1_l2_regularizer':
            regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1=1.0, scale_l2=1.0,)
        else:
            regularizer = None
        return regularizer

    def solver_func(self, hparams, loss, training_iter):
        starting_lr = hparams['opt_case']['lr']
        decay_steps = int(math.ceil(training_iter * hparams['decay_steps'] / 100.0))
        # Decay lr at every 40 steps with a base of 0.96
        learning_rate = tf.train.exponential_decay(starting_lr, self.train_step, \
                        decay_steps, hparams['decay_rate'], staircase=True)
        
        # The variable training and update_ops are necessary for batch normalization
        # Noted: The operations which tf.layers.batch_normalization adds to update mean and variance 
        # don't automatically get added as dependencies of the train operation.
        # So tf.GraphKeys.UPDATE_OPS collection must be used here
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="resnet")
        with tf.control_dependencies(update_ops):
            if hparams['opt_case']['optimizer']=='Adadelta':
                train_op = tf.train.AdadeltaOptimizer( \
                    learning_rate=learning_rate \
                    ).minimize(loss, name="train_op")
            elif hparams['opt_case']['optimizer']=='Adagrad':
                train_op = tf.train.AdagradOptimizer( \
                    learning_rate=learning_rate \
                    ).minimize(loss, name="train_op")
            elif hparams['opt_case']['optimizer']=='Momentum':
                train_op = tf.train.MomentumOptimizer( \
                    learning_rate=learning_rate, \
                    momentum = hparams['opt_case']['momentum'] \
                    ).minimize(loss, name="train_op")
            elif hparams['opt_case']['optimizer']=='Adam':
                train_op = tf.train.AdamOptimizer( \
                    learning_rate=learning_rate \
                    ).minimize(loss, name="train_op")
            elif hparams['opt_case']['optimizer']=='RMSProp':
                train_op = tf.train.RMSPropOptimizer( \
                    learning_rate=learning_rate, \
                    momentum = hparams['opt_case']['momentum'], \
                    decay = hparams['opt_case']['grad_decay'] \
                    ).minimize(loss, name="train_op")
            elif hparams['opt_case']['optimizer']=='gd':
                train_op = tf.train.GradientDescentOptimizer( \
                    learning_rate=learning_rate \
                    ).minimize(loss, name="train_op")
            else:
                raise RuntimeError('Hyper-parameter optimizer is wrong!')
            
            return train_op, learning_rate