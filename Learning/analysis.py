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
This script computes bounds on the privacy cost of training the
student model from noisy aggregation of labels predicted by teachers.
It should be used only after training the student (and therefore the
teachers as well).

The command that computes the epsilon bound associated
with the training of the MNIST student model (100 label queries) is:

python analysis.py
  --counts_file=mnist_250_teachers_labels.npy
  --indices_file=mnist_250_teachers_100_indices_used_by_student.npy 
  --distributed_noise=<True/False>
  --ratio_successful_teachers=<1/0.9/0.7>

The command that computes the epsilon bound associated
with the training of the SVHN student model (500 label queries) is:

python analysis.py
  --counts_file=svhn_250_teachers_labels.npy
  --max_examples=500  
  --distributed_noise=<True/False>
  --ratio_successful_teachers=<1/0.9/0.7>

"""
import os
import math
import numpy as np
from six.moves import xrange
import tensorflow.compat.v1 as tf

from scipy import special
from scipy.integrate import quad

from input import maybe_download

# These parameters can be changed to compute bounds for different failure rates
# or different model predictions.

tf.flags.DEFINE_integer("moments",25, "Number of moments")
tf.flags.DEFINE_float("noise_eps", 0.1, "Eps value for each call to noisymax.")
tf.flags.DEFINE_float("delta", 1e-5, "Target value of delta.")
tf.flags.DEFINE_string("counts_file","","Numpy matrix with raw counts")
tf.flags.DEFINE_string("indices_file","",
    "File containting a numpy matrix with indices used."
    "Optional. Use the first max_examples indices if this is not provided.")
tf.flags.DEFINE_integer("max_examples",1000,
    "Number of examples to use. We will use the first"
    " max_examples many examples from the counts_file"
    " or indices_file to do the privacy cost estimate")
tf.flags.DEFINE_bool("input_is_counts", False, "False if labels, True if counts")

tf.flags.DEFINE_boolean('distributed_noise', True, 'Activate distributed noise')
tf.flags.DEFINE_float('ratio_successful_teachers', 1.0, 'Ratio of teachers who did generate their individual noise.')

FLAGS = tf.flags.FLAGS


def compute_q_noisy_max(counts, noise_eps, tau):
  """returns ~ probability of [outcome != winner].
    param counts: a list of scores
    param noise_eps: noise inverse scale parameter
    return: upper bound of the probability that outcome is different from true winner.
  """
  # For the proof of the upper bound, see Section A.4 of the supplementary material

  winner = np.argmax(counts)
  counts_normalized = noise_eps * (counts - counts[winner])
  counts_rest = np.array(
      [counts_normalized[i] for i in xrange(len(counts)) if i != winner])
  q = 0.0

  if tau > 1/2:
    for c in counts_rest:
      gap = -c
      q += math.exp(-gap) * (1/2 + math.pow(gap, 2*tau-1)/(tau*math.pow(2, 4*tau-2)*math.pow(special.gamma(tau), 2)))
  else:
    for c in counts_rest:
      gap = -c
      q += math.exp(-gap) * (1/2 + math.pow(gap, tau/2)/(tau*math.pow(2, 5/2*tau-1)*math.pow(special.gamma(tau), 2))
                             * math.pow(3/2*tau, 3/2*tau) * math.pow(2/tau-3, 1-3/2*tau))

  return min(q, 1.0 - (1.0/len(counts)))


def logmgf_exact(q, priv_eps, l):
  """
  Computes the logmgf value given q and privacy eps.
  The bound used is the min of three terms (see Theorem 3 of the paper).
  param q: upper bound of the probability that outcome is different from true winner
  param priv_eps: DP guarantee per query
  param l: order of the moment generating function
  return: upper bound of the moment generating function
  """
  if q < (math.exp(priv_eps)-1)/(math.exp(2*priv_eps)-1):
    t_one = (1-q) * math.pow((1-q) / (1 - math.exp(priv_eps) * q), l)
    t_two = q * math.exp(priv_eps * l)
    t = t_one + t_two
    log_t = math.log(t)
    return min(priv_eps * l, 0.5 * priv_eps * priv_eps * l * (l + 1), log_t)
  else:
    return min(priv_eps * l, 0.5 * priv_eps * priv_eps * l * (l + 1))

def priv_eps_per_query(noise_eps, tau):
  """
    Computes the privacy guarantee per query
      param noise_eps: noise inverse scale parameter
      return: priv_eps: DP guarantee per query
  """
  if tau != 1:

    I = lambda z: quad(lambda x: math.pow((x+z)*x, tau-1) * math.exp(-2*x), 0, math.inf)[0]

    L = noise_eps/math.pow(special.gamma(tau), 2)

    F = lambda t: L * quad(lambda u: math.exp(-noise_eps*abs(u))*I(noise_eps * abs(u)), -math.inf, t)[0]

    f = lambda t: L * math.exp(-noise_eps * abs(t)) * I(noise_eps * abs(t))

    g = lambda t: (1 - F(t)) / (1-F(t+2))

    g_dif = lambda t: ((1-F(t))*f(t+2) - (1-F(t+2))*f(t))/math.pow(1-F(t+2), 2)

    if tau > 1 / 2:
      priv_eps = min(math.log(1 + (F(1) - F(-1)) / (1-F(2))), math.log(g(0) - g_dif(0)))
    else:
      priv_eps = math.log(1 + (F(1) - F(-1)) / (1-F(2)))

    return priv_eps
  else:
    return 2 * noise_eps

def logmgf_from_counts(counts, noise_eps, tau, priv_eps, l):
  q = compute_q_noisy_max(counts, noise_eps, tau)
  return logmgf_exact(q, priv_eps, l)


def main(unused_argv):
  # Binaries for MNIST results
  paper_binaries_mnist = \
    ["https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/mnist_250_teachers_labels.npy?raw=true",
    "https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/mnist_250_teachers_100_indices_used_by_student.npy?raw=true"]
  if FLAGS.counts_file == "mnist_250_teachers_labels.npy" \
    or FLAGS.indices_file == "mnist_250_teachers_100_indices_used_by_student.npy":
    maybe_download(paper_binaries_mnist, os.getcwd())

  # Binaries for SVHN results
  paper_binaries_svhn = ["https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/svhn_250_teachers_labels.npy?raw=true"]
  if FLAGS.counts_file == "svhn_250_teachers_labels.npy":
    maybe_download(paper_binaries_svhn, os.getcwd())

  input_mat = np.load(FLAGS.counts_file)
  if FLAGS.input_is_counts:
    counts_mat = input_mat
  else:
    # In this case, the input is the raw predictions. Transform
    num_teachers, n = input_mat.shape
    counts_mat = np.zeros((n, 10)).astype(np.int32)
    for i in range(n):
      for j in range(num_teachers):
        counts_mat[i, int(input_mat[j, i])] += 1
  n = counts_mat.shape[0]
  num_examples = min(n, FLAGS.max_examples)

  if not FLAGS.indices_file:
    indices = np.array(range(num_examples))
  else:
    index_list = np.load(FLAGS.indices_file)
    indices = index_list[:num_examples]
  
  l_list = 1.0 + np.array(xrange(FLAGS.moments))
  total_log_mgf_nm = np.array([0.0 for _ in l_list])
  noise_eps = FLAGS.noise_eps

  tau = 1
  if FLAGS.distributed_noise:
    tau = FLAGS.ratio_successful_teachers
    if tau > 1:
      raise Exception("The ratio of successful teachers must be lower than 1.")
  priv_eps = priv_eps_per_query(noise_eps, tau)

  for i in indices:
    total_log_mgf_nm += np.array(
        [logmgf_from_counts(counts_mat[i], noise_eps, tau, priv_eps, l)
         for l in l_list])

  delta = FLAGS.delta

  # We want delta = exp(alpha - eps l).
  # Solving gives eps = (alpha - ln (delta))/l
  eps_list_nm = (total_log_mgf_nm - math.log(delta)) / l_list

  print("Epsilons (Noisy Max): " + str(eps_list_nm))

  print("Epsilon = " + str(min(eps_list_nm)) + ".")
  if min(eps_list_nm) == eps_list_nm[-1]:
    print("Warning: May not have used enough values of l")

  return


if __name__ == "__main__":
  tf.app.run()
