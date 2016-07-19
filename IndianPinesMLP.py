
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
import patch_size
import tensorflow as tf


# In[3]:

# The IndianPines dataset has 16 classes, representing different kinds of land-cover.
NUM_CLASSES = 16    

# We will classify each patch
IMAGE_SIZE = patch_size.patch_size
CHANNELS = 220
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS


def inference(images, fc1_units, fc2_units, fc3_units):
    """Build the IndianPines model up to where it may be used for inference.
    Args:
    images: Images placeholder, from inputs().
    fc1_units = Number of units in the first fully connected hidden layer
    fc2_units = Number of units in the second fully connected hidden layer
    fc3_units = Number of units in the third fully connected hidden layer

    Returns:
    softmax_linear: Output tensor with the computed logits.
    """
    
    images_flat = tf.reshape(images, [-1, IMAGE_PIXELS])
    print (images_flat)
    size_after_flatten = int(images_flat.get_shape()[1])
    
    # FC 1
    with tf.variable_scope('h_FC1') as scope:
        weights = tf.Variable(
            tf.truncated_normal([size_after_flatten, fc1_units],
                                stddev=1.0 / math.sqrt(float(size_after_flatten))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc1_units]),
                             name='biases')
        h_FC1 = tf.nn.relu(tf.matmul(images_flat, weights) + biases, name=scope.name)
        
    # FC 2
    with tf.variable_scope('h_FC2'):
        weights = tf.Variable(
            tf.truncated_normal([fc1_units, fc2_units],
                                stddev=1.0 / math.sqrt(float(fc1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc2_units]),
                             name='biases')
        h_FC2 = tf.nn.relu(tf.matmul(h_FC1, weights) + biases, name=scope.name)
    
    # FC 3
    with tf.variable_scope('h_FC2'):
        weights = tf.Variable(
            tf.truncated_normal([fc2_units, fc3_units],
                                stddev=1.0 / math.sqrt(float(fc2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([fc3_units]),
                             name='biases')
        h_FC3 = tf.nn.relu(tf.matmul(h_FC2, weights) + biases, name=scope.name)
    
    
    # Linear
    with tf.variable_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([fc3_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(fc3_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(h_FC3, weights) + biases
    
    
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

