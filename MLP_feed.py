
# coding: utf-8

# Trains and Evaluates the IndianPines network using a feed dictionary
# ========================================

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import IndianPinesMLP 
import patch_size
# import IndianPines_data_set as input_data
import Spatial_dataset as input_data


# Declare model parameters as external flags
# -------------------------------------

# In[2]:

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 350, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 150, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('batch_size', 200, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
# flags.DEFINE_string('train_dir', '1.mat', 'Directory to put the training data.')


# In[3]:

learning_rate = 0.01
num_epochs = 20
max_steps = 5000
IMAGE_SIZE = patch_size.patch_size
fc1 = 500
fc2 = 350
fc3 = 150
batch_size = 200
TRAIN_FILES = 8
TEST_FILES = 4
DATA_PATH = os.path.join(os.getcwd(),"Data")


# In[4]:

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IndianPinesMLP
                                                           .IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


# In[5]:

def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }
    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


# In[6]:

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


# In[7]:

def add_DataSet(first,second):
    temp_image = np.concatenate((first.images,second.images),axis=0)
    temp_labels = np.concatenate((first.labels,second.labels),axis=0)
    temp_image = temp_image.reshape(temp_image.shape[0],IMAGE_SIZE,IMAGE_SIZE,220)
    temp_image = np.transpose(temp_image,(0,3,1,2))
    temp_labels = np.transpose(temp_labels)
    return input_data.DataSet(temp_image,temp_labels)


# In[ ]:

def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on IndianPines.
    
    """Concatenating all the training and test mat files"""
    for i in range(TRAIN_FILES):
        data_sets = input_data.read_data_sets(os.path.join(DATA_PATH, 'Train_'+str(IMAGE_SIZE)+'_'+str(i+1)+'.mat'), 'train')
        if(i==0):
            Training_data = data_sets
            continue
        else:
            Training_data = add_DataSet(Training_data,data_sets)
            
    for i in range(TEST_FILES):
        data_sets = input_data.read_data_sets(os.path.join(DATA_PATH, 'Test_'+str(IMAGE_SIZE)+'_'+str(i+1)+'.mat'),'test')
        if(i==0):
            Test_data = data_sets
            continue
        else:
            Test_data = add_DataSet(Test_data,data_sets)
        
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = IndianPinesMLP.inference(images_placeholder,
                                 FLAGS.hidden1,
                                 FLAGS.hidden2,
                                 FLAGS.hidden3)

        # Add to the Graph the Ops for loss calculation.
        loss = IndianPinesMLP.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = IndianPinesMLP.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = IndianPinesMLP.evaluation(logits, labels_placeholder)

        # Build the summary operation based on the TF collection of Summaries.
    #    summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
    #    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(Training_data,
                                     images_placeholder,
                                     labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                   feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 50 == 0:
            # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
    #             summary_str = sess.run(summary_op, feed_dict=feed_dict)
    #             summary_writer.add_summary(summary_str, step)
    #             summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, 'model-MLP-'+str(IMAGE_SIZE)+'X'+str(IMAGE_SIZE)+'.ckpt', global_step=step)

            # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        Training_data)
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        Test_data)


# In[ ]:

run_training()


# In[ ]:



