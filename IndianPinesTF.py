
# coding: utf-8

# Builds the __IndianPines__ network.
# ===================================
# Implements the _inference/loss/training pattern_ for model building.
# 1. inference() - Builds the model as far as is required for running the network forward to make predictions.
# 2. loss() - Adds to the inference model the layers required to generate loss.
# 3. training() - Adds to the loss model the Ops required to generate and apply gradients.
# 
# This file is used by the various "fully_connected_*.py" files and not meant to be run.

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import patch_size

# In[3]:

# The IndianPines dataset has 16 classes, representing different kinds of land-cover.
NUM_CLASSES = 16    # change to 16 in originaldata tell anirban

# We have chopped the IndianPines image into 28x28 pixels patches. 
# We will classify each patch
IMAGE_SIZE = patch_size.patch_size
KERNEL_SIZE = 3 #before it was 5 for 37x37
CHANNELS = 220
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS


# Build the IndianPines model up to where it may be used for inference.
# --------------------------------------------------
# Args:
# * images: Images placeholder, from inputs().
# * hidden1_units: Size of the first hidden layer.
# * hidden2_units: Size of the second hidden layer.
# 
# Returns:
# * softmax_linear: Output tensor with the computed logits.

# In[5]:

def inference(images, conv1_channels, conv2_channels, fc1_units, fc2_units):
    """Build the IndianPines model up to where it may be used for inference.
    Args:
    images: Images placeholder, from inputs().
    conv1_channels: Number of filters in the first convolutional layer.
    conv2_channels: Number of filters in the second convolutional layer.
    fc1_units = Number of units in the first fully connected hidden layer
    fc2_units = Number of units in the second fully connected hidden layer

    Returns:
    softmax_linear: Output tensor with the computed logits.
    """
    
    # Conv 1
    with tf.variable_scope('conv_1') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE, KERNEL_SIZE, 220, conv1_channels], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv1_channels], initializer=tf.constant_initializer(0.05))
        #print(biases)
        # Flattening the 3D image into a 1D array
        x_image = tf.reshape(images, [-1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS])
        z = tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='VALID')
        print (z)
        h_conv1 = tf.nn.relu(z+biases, name=scope.name)
    
    # Maxpool 1
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool1')
    
    # Conv2
    with tf.variable_scope('h_conv2') as scope:
        weights = tf.get_variable('weights', shape=[KERNEL_SIZE, KERNEL_SIZE, conv1_channels, conv2_channels], 
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=[conv2_channels], initializer=tf.constant_initializer(0.05))
        z = tf.nn.conv2d(h_pool1, weights, strides=[1, 1, 1, 1], padding='VALID')
        h_conv2 = tf.nn.relu(z+biases, name=scope.name)
        print (h_conv2)
        
    # Maxpool 2
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name='h_pool2')
    print (h_pool2)
    
    
    size_after_conv_and_pool_twice = int(math.ceil((math.ceil(float(IMAGE_SIZE-KERNEL_SIZE+1)/2)-KERNEL_SIZE+1)/2))
    
    
    #Reshape from 4D to 2D
    h_pool2_flat = tf.reshape(h_pool2, [-1, (size_after_conv_and_pool_twice**2)*conv2_channels])
    print (h_pool2_flat)
    size_after_flatten = int(h_pool2_flat.get_shape()[1])
    
    # FC 1
    with tf.variable_scope('h_FC1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([size_after_flatten, fc1_units],
                                stddev=1.0 / math.sqrt(float(size_after_flatten))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc1_units]),
                             name='biases')
        h_FC1 = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases, name=scope.name)
        
    # FC 2
    with tf.variable_scope('h_FC2'):
        weights = tf.Variable(
            tf.truncated_normal([fc1_units, fc2_units],
                                stddev=1.0 / math.sqrt(float(fc1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc2_units]),
                             name='biases')
        h_FC2 = tf.nn.relu(tf.matmul(h_FC1, weights) + biases, name=scope.name)
    
    # Linear
    with tf.variable_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([fc2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(h_FC2, weights) + biases
    
    
    return logits


# Define the loss function
# ------------------------

# In[6]:

def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


# Define the Training OP
# --------------------

# In[8]:

def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# Define the Evaluation OP
# ----------------------

# In[9]:

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

