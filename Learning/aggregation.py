# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange

import tensorflow.compat.v1 as tf


FLAGS=tf.flags.FLAGS

tf.flags.DEFINE_boolean('distributed_noise', False, 'Activate distributed noise mode')

tf.flags.DEFINE_integer('nb_successful_teachers', None, 'Number of teachers whose noise is kept secret (a.k.a. successful teachers)')

tf.flags.DEFINE_string('ohe_file','','Stores the one-hot encodings.')

tf.flags.DEFINE_string('lap_noises_file', '', 'Stores the laplace noises to send to the HE layer.')


def labels_from_probs(probs):
  """
  Helper function: computes argmax along last dimension of array to obtain
  labels (max prob or max logit value)
  :param probs: numpy array where probabilities or logits are on last dimension
  :return: array with same shape as input besides last dimension with shape 1
          now containing the labels
  """
  # Compute last axis index
  last_axis = len(np.shape(probs)) - 1

  # Label is argmax over last dimension
  labels = np.argmax(probs, axis=last_axis)

  # Return as np.int32
  return np.asarray(labels, dtype=np.int32)


def one_hot_encoded_from_labels(sample_labels):
  """
  param sample_labels: array that stores the labels the teachers assigned to one example
  return: one-hot encoded vector representing sample_labels
  """
  #sample_labels corresponds to the labels of one single sample
  nb_teachers = len(sample_labels)
  result = np.zeros((nb_teachers, 10))
  for teacher in xrange(nb_teachers):
    result[teacher, sample_labels[teacher]] = 1
  return result


def noisy_max(logits, lap_scale, return_clean_votes=False):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the noisy-max of
  the votes for candidate classes to select a label for each sample: it
  adds Laplace noise to label counts and returns the most frequent label.
  :param logits: logits or probabilities for each sample
  :param lap_scale: scale of the Laplace noise to be added to counts
  :param return_clean_votes: if set to True, also returns clean votes (without
                      Laplace noise). This can be used to perform the
                      privacy analysis of this aggregation mechanism.
  :return: pair of result and (if clean_votes is set to True) the clean counts
           for each class per sample and the original labels produced by
           the teachers.
  """

  # Compute labels from logits/probs and reshape array properly
  labels = labels_from_probs(logits)
  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))
  
  # Generate laplace noises to send to HE layer
  laplace_noises = np.random.laplace(loc=0.0, scale=float(lap_scale), size=(int(labels_shape[1]), 10))
  
  # Prepare filepath for dump of laplace noises
  filepath_lap = FLAGS.lap_noises_file + "_" + str(FLAGS.dataset) + '_lap_' + str(FLAGS.lap_scale) + '_nb_ex_' + str(int(labels_shape[1])) + '.txt'

  # Dump laplace noises array
  with open(filepath_lap, mode='w') as f:
    np.savetxt(f, laplace_noises, fmt='%f', delimiter=" ")

  # Initialize array to hold final labels
  result = np.zeros(int(labels_shape[1]))

  if return_clean_votes:
    # Initialize array to hold clean votes for each sample
    clean_votes = np.zeros((int(labels_shape[1]), 10))

  nb_successful_teachers = FLAGS.nb_successful_teachers
  if nb_successful_teachers == None:
    nb_successful_teachers = FLAGS.nb_teachers
  if nb_successful_teachers > FLAGS.nb_teachers:
    raise ValueError('The number of successful teachers must not be greater than the total number of teachers.')
  
  # Prepare filepath for dump of one-hot encodings
  filepath_ohe = FLAGS.ohe_file + "_" + str(FLAGS.dataset) + '_' + str(FLAGS.nb_teachers) + '_distributed_noise_nb_successful_' + str(nb_successful_teachers) + '_lap_' + str(FLAGS.lap_scale) + '_nb_ex_' + str(int(labels_shape[1])) + '.txt'
  
  np.random.seed(1)
  with open(filepath_ohe, mode='w') as f:
    # Parse each sample
    for i in xrange(int(labels_shape[1])):
      # Count number of votes assigned to each class
      label_counts = np.bincount(labels[:, i], minlength=10)
  
      if return_clean_votes:
        # Store vote counts for export
        clean_votes[i] = label_counts
      ohe = one_hot_encoded_from_labels(labels[:, i])
      if FLAGS.distributed_noise:
        print("\n\nlabels_shape[0] :", labels_shape[0], "\n\n")
        ohe[:nb_successful_teachers, :] += np.random.gamma(1/labels_shape[0], float(lap_scale), (nb_successful_teachers, 10)) \
                                          - np.random.gamma(1/labels_shape[0], float(lap_scale), (nb_successful_teachers, 10))
        # Save ohe of the current sample in file
        label_counts = np.sum(ohe, axis=0)
      else:
        # Cast in float32 to prepare before addition of Laplacian noise
        label_counts = np.asarray(label_counts, dtype=np.float32)
    
        # Sample independent Laplacian noise for each class
        for item in xrange(10):
          label_counts[item] += np.random.laplace(loc=0.0, scale=float(lap_scale))
			# Save the clean one-hot encodings for HBC (distributed_noise=False) and the noisy one-hot encodings for BHBC (distributed_noise=True)
      np.savetxt(f, ohe, delimiter=" ")
      f.write("\n")
      # Result is the most frequent label
      result[i] = np.argmax(label_counts)

  # Cast labels to np.int32 for compatibility with deep_cnn.py feed dictionaries
  result = np.asarray(result, dtype=np.int32)

  if return_clean_votes:
    # Returns several array, which are later saved:
    # result: labels obtained from the noisy aggregation
    # clean_votes: the number of teacher votes assigned to each sample and class
    # labels: the labels assigned by teachers (before the noisy aggregation)
    return result, clean_votes, labels
  else:
    # Only return labels resulting from noisy aggregation
    return result


def aggregation_most_frequent(logits):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the most frequent
  label. It is deterministic (no noise injection like noisy_max() above.
  :param logits: logits or probabilities for each sample
  :return:
  """
  # Compute labels from logits/probs and reshape array properly
  labels = labels_from_probs(logits)
  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))

  # Initialize array to hold final labels
  result = np.zeros(int(labels_shape[1]))

  # Parse each sample
  for i in xrange(int(labels_shape[1])):
    # Count number of votes assigned to each class
    label_counts = np.bincount(labels[:, i], minlength=10)

    label_counts = np.asarray(label_counts, dtype=np.int32)

    # Result is the most frequent label
    result[i] = np.argmax(label_counts)

  return np.asarray(result, dtype=np.int32)
