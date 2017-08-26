"""Recovering normal distribution parameters in Edward."""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import (
    Empirical,
    Normal,
    NormalWithSoftplusScale,
)

MU = 6.0
SIGMA = 1.5
N = 1000

# Data
y_train = np.random.normal(MU, SIGMA, [N, 1])

# Params (defined as 1d priors)
mu = Normal(loc=[0.0], scale=[5.0])
inv_softplus_sigma = Normal(loc=[0.0], scale=[1.0])

# Model (defined as vector over full dataset)
y = NormalWithSoftplusScale(loc=mu, scale=inv_softplus_sigma, sample_shape=N)


## Variational Inference
print('Variational based approach')

# Posterior distribution families
q_mu = Normal(
    loc = tf.Variable([0.0]),
    scale = tf.Variable([5.0]),
)
q_inv_softplus_sigma = Normal(
    loc = tf.Variable([0.0]),
    scale = tf.Variable([1.0]),
)

# Inference arguments
latent_vars = {mu: q_mu, inv_softplus_sigma: q_inv_softplus_sigma}
data = {y: y_train}

# Inference
inference = ed.KLqp(latent_vars, data)
inference.run(n_samples=5, n_iter=2500)

print(q_mu.mean().eval())
print(q_inv_softplus_sigma.mean().eval())


## Sampling Methods
print('Sampling based approach')

# Posterior distribution families
q_mu = Empirical(params=tf.Variable(tf.random_normal([1000, 1])))
q_inv_softplus_sigma = Empirical(params=tf.Variable(tf.random_normal([1000, 1])))

# Inference arguments
latent_vars = {mu: q_mu, inv_softplus_sigma: q_inv_softplus_sigma}
data = {y: y_train}

# Inference
inference = ed.HMC(latent_vars, data)
inference.run()

print(q_mu.mean().eval())
print(q_inv_softplus_sigma.mean().eval())
