import numpy as np
import os
import tensorflow as tf
import tempfile
from six.moves import urllib
import gzip
import shutil

all_train_images = None
all_train_labels = None
all_test_images = None
all_test_labels = None

def download(directory, filename):
  """Download (and unzip) a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
  _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
  print('Downloading %s to %s' % (url, zipped_filepath))
  urllib.request.urlretrieve(url, zipped_filepath)
  with gzip.open(zipped_filepath, 'rb') as f_in, \
      tf.gfile.Open(filepath, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
  os.remove(zipped_filepath)
  return filepath

def load_dataset():
  file_names = ('train-images-idx3-ubyte',
                  'train-labels-idx1-ubyte',
                  't10k-images-idx3-ubyte',
                  't10k-labels-idx1-ubyte')
  for n in file_names:
    download('MNIST_data', n)


  global all_train_images
  global all_train_labels
  global all_test_images
  global all_test_labels
  train_image_file = open(os.path.join('MNIST_data', file_names[0]), 'rb')
  train_label_file = open(os.path.join('MNIST_data', file_names[1]), 'rb')
  test_image_file = open(os.path.join('MNIST_data', file_names[2]), 'rb')
  test_label_file = open(os.path.join('MNIST_data', file_names[3]), 'rb')

  all_train_images = np.frombuffer(train_image_file.read(), np.uint8, offset=16).reshape(-1,28*28)
  all_train_labels = np.frombuffer(train_label_file.read(), np.uint8, offset=8)
  all_test_images = np.frombuffer(test_image_file.read(), np.uint8, offset=16).reshape(-1,28*28)
  all_test_labels = np.frombuffer(test_label_file.read(), np.uint8, offset=8)

  train_image_file.close()
  train_label_file.close()
  test_image_file.close()
  test_label_file.close()
  return

train_batch_start = 0
def get_train_batch(batch_size):
  global train_batch_start
  start = train_batch_start
  train_batch_start = (train_batch_start+batch_size) % len(all_train_images)
  if start + batch_size > len(all_train_images):
    train_images = all_train_images[start:len(all_train_images)] + all_train_images[0:train_batch_start]
    train_labels = all_train_labels[start:len(all_train_images)] + all_train_labels[0:train_batch_start]
  else:
    train_images = all_train_images[start:train_batch_start]
    train_labels = all_train_labels[start:train_batch_start]
  return train_images, train_labels

def get_test_data():
  return all_test_images, all_test_labels
