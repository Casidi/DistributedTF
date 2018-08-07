import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

rng = np.random

# Parameters
learning_rate = 0.0001
training_epochs = 1000
display_step = 50
use_bad_data = True

if use_bad_data:
    # Bad Training Data
    train_Y = np.asarray([  59.8000,   60.5000,   60.9000,   61.0000,   61.5000,   64.0000,   64.5000,
                            64.8000,   67.8000,   71.2000,   72.0000,   78.9000,   79.2000,   81.0000,
                            82.6000,   84.0000,   84.0000])
    train_X = np.asarray([600., 760., 802., 568., 679., 865., 1103., 865., 896., 1068.,
                            769., 1062., 1123., 1081., 1137., 1137., 1137.])
else:
    # Good Training Data
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                        7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                        2.827,3.465,1.65,2.904,2.42,2.94,1.3])

print(train_X.dtype)
print(train_Y.dtype)
print(str(train_X))
print(str(train_Y))

n_samples = train_X.shape[0]

print("Samples = %d" % n_samples)

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(1.0, name="weight")
b = tf.Variable(1.0, name="bias")

pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            #print("x=%f y=%f" % (x,y))
            op, c = sess.run([optimizer, cost], feed_dict={X: x, Y: y})

            # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run([cost], feed_dict={X: train_X, Y:train_Y})          
              
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{}".format(c), \
            "W=", sess.run(W), "b=", sess.run(b))
            if np.isnan(c):
                print 'this is bad'

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    if use_bad_data:
        # Bad Test Data
        test_Y = np.asarray([ 48.2000,  56.5000,  57.0000,  59.5000,  15.0000,  17.8000,  43.5000,  50.2000])
        test_X = np.asarray([ 549.,  710.,  568.,  825., 414.,  439.,  460.,  614. ])
    else:
        # Good Test Data
        test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
        test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])


    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))
