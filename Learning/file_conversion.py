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


"""
This script converts the one-hot encoded vectors output by the homomorphic argmax
in an array of labels (completed with dummy values) to feed the impoved GAN with
the expected structure
"""

import os
import tensorflow.compat.v1 as tf
import numpy as np
from random import randrange


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_teachers', 250, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 100,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 0.31,
                        'Scale of the Laplace noise added for privacy')
tf.flags.DEFINE_string('ohe_after_HE_argmax_dir','/ohe_after_argmax','Path and begining of the file where ohe after argmax have been stored.')
tf.flags.DEFINE_string('data_after_argmax_dir','/tmp/ohe_after_argmax_dir','Where .npy files will be stored.')
tf.flags.DEFINE_boolean('distributed_noise', False, 'Activate beyond honest but curious mode')
tf.flags.DEFINE_integer('nb_successful_teachers', None, 'Number of teachers who generate the individual noise properly')



def complete_labels_from_ohe(ohe, total_nb_ex):
  """
  Convert the one-hot encoded vectors coming from homomorphic argmax into labels
    param ohe: one-hot encoded vectors output by homomorphic argmax
    param total_nb_ex: total number of examples to respect the dimension of the array to pass to improved GAN
    return: array of the labels, completed by dummy values
  """
  nb_ex, nb_classes = ohe.shape
  dummy_value = -1
  # Initialize complete_labels array with dummy values which will remain for the last total_nb_ex - nb_ex examples
  complete_labels = np.full(total_nb_ex, dummy_value)
  for ex in range(nb_ex):
    argmax_failed = True
    # get label from one-hot encodings
    for c in range(nb_classes):
      if ohe[ex, c] == 1:
        complete_labels[ex] = c
        argmax_failed = False
    # if homomorphic argmax failed (there are only zeros in the one-hot encoding), chose a random label
    if argmax_failed:
      complete_labels[ex] = randrange(10)
  return complete_labels


def main(argv=None):
  dataset = FLAGS.dataset
  nb_teachers = FLAGS.nb_teachers
  nb_successful_teachers = FLAGS.nb_successful_teachers
  if FLAGS.nb_successful_teachers == None:
    nb_successful_teachers = nb_teachers
  if FLAGS.distributed_noise:
    filepath_ohe = os.path.join(FLAGS.ohe_after_HE_argmax_dir, "clean_ohe_" + str(dataset) + "_" + str(nb_teachers) + "_distributed_noise_nb_successful_" + str(nb_successful_teachers) + "_lap_" + str(FLAGS.lap_scale) + "_nb_ex_" + str(FLAGS.stdnt_share) + '.txt')
  else:
    filepath_ohe = os.path.join(FLAGS.ohe_after_HE_argmax_dir, "clean_ohe_" + str(dataset) + "_" + str(nb_teachers) + "_nb_ex_" + str(FLAGS.stdnt_share) + '.txt')
    
  # Get one-hot encodings from file
  with open(filepath_ohe, "r") as file:
    ohe = np.genfromtxt(file, delimiter=',')
  total_nb_ex = 10000
  complete_labels = complete_labels_from_ohe(ohe, total_nb_ex)
  data_dir = FLAGS.data_after_argmax_dir
  if FLAGS.distributed_noise:
    filepath_npy = os.path.join(data_dir, str(dataset) + '_' + str(nb_teachers) + '_student_labels_distributed_noise_lap_' + str(FLAGS.lap_scale) + '_nb_st_' + str(nb_successful_teachers) + '.npy')
  else:
    filepath_npy = os.path.join(data_dir, str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(FLAGS.lap_scale) + '.npy')
  # Save labels in npy file
  with tf.gfile.Open(filepath_npy, mode='w') as file_obj:
      np.save(file_obj, complete_labels)
if __name__ == '__main__':
  tf.app.run()
