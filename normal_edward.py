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
y_train = np.random.normal(MU, SIGMA, [N])

# Model
mu = Normal(loc=0.0, scale=5.0)
inv_softplus_sigma = Normal(loc=0.0, scale=1.0)
y = NormalWithSoftplusScale(loc=mu, scale=inv_softplus_sigma, sample_shape=N)

## Variational Model with VI
q_mu = Normal(
    loc = tf.Variable(0.0),
    scale = tf.Variable(5.0),
)
q_inv_softplus_sigma = Normal(
    loc = tf.Variable(0.0),
    scale = tf.Variable(1.0),
)

# Inference arguments
latent_vars = {mu: q_mu, inv_softplus_sigma: q_inv_softplus_sigma}
data = {y: y_train}

# Inference
inference = ed.KLqp(latent_vars, data)
inference.run(n_samples=5, n_iter=2500)

print(q_mu.mean().eval())
print(q_inv_softplus_sigma.mean().eval())


# Empirical Model with Sampler

# Posterior distribution families
q_mu = Empirical(params=tf.Variable(tf.random_normal([2000])))
q_inv_softplus_sigma = Empirical(params=tf.Variable(tf.random_normal([2000])))

# Inference arguments
latent_vars = {mu: q_mu, inv_softplus_sigma: q_inv_softplus_sigma}
data = {y: y_train}

# Inference
inference = ed.HMC(latent_vars, data)
inference.run(step_size=0.003, n_steps=5)

print(tf.reduce_mean(q_mu.params[1000:]).eval())
print(tf.nn.softplus(tf.reduce_mean(q_inv_softplus_sigma.params[1000:])).eval())
