'''
The whole purpose of this experiment is try to load the network saved in
retrain.py, and run evaluation
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile

import tensorflow as tf
import numpy as np

from tensorflow.python.client import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile

from bottleneck import get_random_cached_bottlenecks

FLAGS = tf.app.flags.FLAGS

# Input and output file flags.
tf.app.flags.DEFINE_string('image_dir', '',
                           """Path to folders of labeled images.""")
#tf.app.flags.DEFINE_string('output_graph', '/tmp/output_graph.pb',
#                           """Where to save the trained graph.""")
#tf.app.flags.DEFINE_string('output_labels', '/tmp/output_labels.txt',
#                           """Where to save the trained graph's labels.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/tf_summary',
                           """Where to save the summary log files""")
tf.app.flags.DEFINE_integer(
    'testing_percentage', 10,
    """What percentage of images to use as a test set.""")
tf.app.flags.DEFINE_integer(
    'validation_percentage', 10,
    """What percentage of images to use as a validation set.""")
tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """Path to original.pb, """)
tf.app.flags.DEFINE_integer('test_batch_size', 500,
                            """How many images to test on at a time. This"""
                            """ test set is only used infrequently to verify"""
                            """ the overall accuracy of the model.""")
tf.app.flags.DEFINE_string(
    'bottleneck_dir', '/tmp/bottleneck',
    """Path to cache bottleneck layer values as files.""")

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_INPUT_NAME = 'BottleneckInputPlaceholder:0'
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
RESULT_TENSOR_NAME = 'final_result:0'

def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}

  # get a list of first level subdirs in image_dir
  sub_dirs = [x[0] for x in os.walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    #raw_input('press enter ...')
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      #use glob to generate a list of contained files smartly
      file_list.extend(glob.glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images, which may cause issues.')
    # get label name from dir name
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      #hash_name = re.sub(r'_nohash_.*$', '', file_name)
      '''
      CUSTOM: DONE: Looks like I need a special hashing function
      such that, images of the same video clip will turn out to have
      the same hash value
      '''
      hash_name = file_name[:-9]
      print(hash_name)
      #raw_input('paused...')

      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
      percentage_hash = (int(hash_name_hashed, 16) % (65536)) * (100 / 65535.0)
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result

def add_evaluation_step(result_tensor, ground_truth_tensor):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Nothing.
  """
  correct_prediction = tf.equal(
      tf.argmax(result_tensor, 1), tf.argmax(ground_truth_tensor, 1))
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, 'float'), name="accuracy")
  return evaluation_step

'''
load the network from the pb file
'''
def load_net():
  with tf.Session() as sess:
    model_filename = os.path.join(FLAGS.model_dir, 'original.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      (bottleneck_tensor, jpeg_data_tensor, bottleneck_input,
          resized_input_tensor, result_tensor) = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              BOTTLENECK_INPUT_NAME,
              RESIZED_INPUT_TENSOR_NAME, RESULT_TENSOR_NAME]))
  return (sess.graph, bottleneck_tensor, jpeg_data_tensor, bottleneck_input,
      resized_input_tensor, result_tensor)


'''
the main function.
'''
def main():
  '''
  1. load the network
  2. run the testing
  '''
  '''create image lists'''
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)

  '''clout classes'''
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    print('Only one valid folder of images found at ' + FLAGS.image_dir +
          ' - multiple classes are needed for classification.')
    return -1

  (graph, bottleneck_tensor, jpeg_data_tensor, bottleneck_input,
      resized_image_tensor, result_tensor) = load_net()

  '''create the ground truth input'''
  ground_truth_input = tf.placeholder(tf.float32,
                                      [None, class_count],
                                      name='GroundTruthInput')

  evaluation_step = add_evaluation_step(result_tensor, ground_truth_input)

  #raw_input('right before session')
  sess = tf.Session()
  #raw_input('right after session')

  ''' CUSTOM: summaries '''
  sw = tf.train.SummaryWriter(FLAGS.summary_dir, sess.graph)
  #raw_input('right after summary writer')
  ''' get a batch for testing'''
  test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
      sess, image_lists, FLAGS.test_batch_size, 'testing',
      FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
      bottleneck_tensor)
  #raw_input('right before evaluation')
  test_accuracy = sess.run(
      evaluation_step,
      feed_dict={bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth})
  print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
  main()
