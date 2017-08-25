"""Recovering normal distribution parameters in pystan."""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import (
    Empirical,
    Normal,
    NormalWithSoftplusScale,
)

MU = 0.0
SIGMA = 1.5
T = 100000

def sample_data(sample_size):
    return np.random.normal(MU, SIGMA, sample_size)

# Data
y_train = sample_data(T)

# Params (defined as 1d priors)
mu = NormalWithSoftplusScale(loc=[0.0], scale=[5.0])
sigma = NormalWithSoftplusScale(loc=[0.0], scale=[5.0])

# Model (defined as vector over full dataset)
y = NormalWithSoftplusScale(loc=tf.tile(mu, [T]), scale=tf.tile(sigma, [T]))

# Posterior distribution families
q_mu = NormalWithSoftplusScale(
    loc = tf.Variable([0.0]),
    scale = tf.Variable([5.0]),
)
q_sigma = NormalWithSoftplusScale(
    loc = tf.Variable([0.0]),
    scale = tf.Variable([5.0]),
)

# Inference arguments
latent_vars = {mu: q_mu, sigma: q_sigma}
data = {y: y_train}

# Inference
inference = ed.KLqp(latent_vars, data)
inference.run(n_it
