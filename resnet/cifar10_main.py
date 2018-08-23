# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from math import ceil # Xinyi add

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags._conventions import help_wrap # Xinyi add
from official.utils.flags import core as flags_core
from official.utils.logs import logger
import resnet_model
import resnet_run_loop

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

DATASET_NAME = 'CIFAR-10'


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record, is_training):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.
  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_NUM_IMAGES['train'],
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _HEIGHT, _WIDTH, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):
  """Model class with appropriate defaults for CIFAR-10 data."""

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """These are the parameters that work for CIFAR-10 data.
    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
      to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    Raises:
      ValueError: if invalid resnet_size is chosen
    """
    if resnet_size % 6 != 2:
      raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6

    super(Cifar10Model, self).__init__(
        resnet_size=resnet_size,
        bottleneck=False,
        num_classes=num_classes,
        num_filters=16,
        kernel_size=3,
        conv_stride=1,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=[num_blocks] * 3,
        block_strides=[1, 2, 2],
        final_size=64,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  features = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

  # Xinyi add, by default turnning off learning rate decay
  boundary_epochs=[250] # Default 250 epoch training
  decay_rates=[1,1]

  if flags.FLAGS.decay_steps !=0 and  flags.FLAGS.decay_steps != 100: # Overwrite
    boundary_epochs_length = int(ceil(100 / flags.FLAGS.decay_steps)) - 1
    boundary_epochs = []
    decay_rates = [1]
    decay_epochs = 250 * flags.FLAGS.decay_steps / 100.0
    for i in xrange(boundary_epochs_length):
      decay_rates.append(flags.FLAGS.decay_rate * decay_rates[i])
      boundary_epochs.append(decay_epochs*(i+1))
  
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=128,
      num_images=_NUM_IMAGES['train'], boundary_epochs=boundary_epochs,
      decay_rates=decay_rates) # Xinyi modified

  # We use a weight decay of 0.0002, which performs better
  # than the 0.0001 that was originally suggested.
  weight_decay = 2e-4 # Xinyi modified it inside resnet_model.py

  # Empirical testing showed that including batch_normalization variables
  # in the calculation of regularized loss helped validation accuracy
  # for the CIFAR-10 dataset, perhaps because the regularization prevents
  # overfitting on the small data set. We therefore include all vars when
  # regularizing and computing loss during training.
  def loss_filter_fn(_):
    return True

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=Cifar10Model,
      resnet_size=params['resnet_size'],
      weight_decay=weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=loss_filter_fn,
      dtype=params['dtype']
  )


def define_cifar_flags(hp, model_id, model_dir, data_dir, train_epochs, total_epochs, epoch_index): # Xinyi modified
  resnet_run_loop.define_resnet_flags()
  flags.adopt_module_key_flags(resnet_run_loop)
  
  # Xinyi add followings
  flags.DEFINE_string(
        name="optimizer", short_name="opt", default=hp['opt_case']['optimizer'],
        help=help_wrap("The name of optimizer type"))
  if hp['opt_case']['optimizer']=='Momentum' \
    or hp['opt_case']['optimizer']=='RMSProp':
      flags.DEFINE_float(
        name="momentum", short_name="mm",
        default=hp['opt_case']['momentum'],
        help=help_wrap("The momentum of Momentum SGD or RMSProp"))
  if hp['opt_case']['optimizer']=='RMSProp':
    flags.DEFINE_float(
        name="grad_decay", short_name="rmspd",
        default=hp['opt_case']['grad_decay'],
        help=help_wrap("The decay of RMSProp"))
  flags.DEFINE_float(
        name="learning_rate", short_name="lr",
        default=hp['opt_case']['lr'],
        help=help_wrap("The initial learning rate of optimizer"))
  flags.DEFINE_float(
        name="decay_rate", short_name="lrdr", default=hp['decay_rate'],
        help=help_wrap("The base term of learning rate decay function"))
  flags.DEFINE_integer(
        name="decay_steps", short_name="lrds", default=hp['decay_steps'],
        help=help_wrap("The power term of learning rate decay function"
            "This value is in percentage of train_epochs"
            "Zero value means turnning off decay"))
  flags.DEFINE_string(
        name="initializer", short_name="initn", default=hp['initializer'],
        help=help_wrap("The name of initialization method"
            "None value means glorot_uniform_initializer"))
  flags.DEFINE_string(
        name="regularizer", short_name="regn", default=hp['regularizer'],
        help=help_wrap("The name of regularization method"
            "None value means turnning off weight decay"))
  flags.DEFINE_float(
        name="weight_decay", short_name="wd", default=hp['weight_decay'],
        help=help_wrap("The amount of regularization"
            "If regularizer=None, the variable becomes useless"))
  flags.DEFINE_integer(
        name="model_id", short_name="mid", default=model_id,
        help=help_wrap("The index of model in the population"))
  flags.DEFINE_integer(
        name="total_epochs", short_name="ttep", default=train_epochs,
        help=help_wrap("The total epochs the model will be trained"))
  flags.DEFINE_integer(
        name="epoch_index", short_name="epi", default=epoch_index,
        help=help_wrap("The epoch index write to csv."))
  
  flags_core.set_defaults(data_dir=data_dir,
                          model_dir=model_dir,
                          resnet_size='50',
                          train_epochs=train_epochs,
                          epochs_between_evals=1,
                          batch_size=hp['batch_size'])


def run_cifar(flags_obj):
  """Run ResNet CIFAR-10 training and eval loop.
  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                    or input_fn)
  eval_accuracy = resnet_run_loop.resnet_main( # Xinyi modified
      flags_obj, cifar10_model_fn, input_function, DATASET_NAME,
      shape=[_HEIGHT, _WIDTH, _NUM_CHANNELS])
  
  return eval_accuracy  # Xinyi modified


def start(_): # Xinyi modified
  with logger.benchmark_context(flags.FLAGS):
    eval_accuracy = run_cifar(flags.FLAGS)
    
    return eval_accuracy, flags.FLAGS.model_id

import sys
def main(hp, model_id, save_base_dir, data_dir, train_epochs, total_epochs, epoch_index): # Xinyi modified
  tf.logging.set_verbosity(tf.logging.ERROR)
  model_dir = save_base_dir + str(model_id)

  for name in list(flags.FLAGS):
    delattr(flags.FLAGS, name)
  define_cifar_flags(hp, model_id, model_dir, data_dir, train_epochs, total_epochs, epoch_index)
  
  absl_app.parse_flags_with_usage(sys.argv)
  return start(0)

  
    
if __name__ == '__main__':
  hp = {  
            'opt_case': {'lr': 0.1, 'optimizer': 'Momentum', 'momentum': 0.9},
            'decay_steps': 20,
            'decay_rate': 0.1,
            'weight_decay': 2e-4,
            'regularizer': 'l2_regularizer',
            'initializer': 'he_init',
            'batch_size': 128}
            
  model_id = 0
  save_base_dir = './model_'
  data_dir = '/home/K8S/dataset/cifar10/'
  train_epochs = 1
  
  print('Return {}'.format( \
      main(hp, model_id, save_base_dir, data_dir, train_epochs))) # Xinyi modified
