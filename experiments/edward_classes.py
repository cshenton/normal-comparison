"""Some initial tests with a reusable edward model."""
import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import (
    Empirical,
    Normal,
    NormalWithSoftplusScale,
)

class NormalModel():
    """A normal distribution model."""
    def __init__(self):
        self.is_fit = False

        self.mu = Normal(loc=0.0, scale=5.0)
        self.inv_softplus_sigma = Normal(loc=0.0, scale=1.0)

    def fit(self, y_train):
        """Fits the normal distribution to the data in numpy array y.

        Once this method is run, the variational model for the parameters
        will be adjusted to match the posterior distribution.
        """
        if self.is_fit:
            msg = 'model already fit'
            raise ValueError(msg)

        self.q_mu = Normal(
            loc = tf.Variable(0.0),
            scale = tf.Variable(5.0),
        )
        self.q_inv_softplus_sigma = Normal(
            loc = tf.Variable(0.0),
            scale = tf.Variable(1.0),
        )
        self.y = NormalWithSoftplusScale(
            loc=self.mu,
            scale=self.inv_softplus_sigma,
            sample_shape=y_train.shape,
        )
        data = {self.y: y_train}
        params = {
            self.mu: self.q_mu,
            self.inv_softplus_sigma: self.q_inv_softplus_sigma
        }
        inference = ed.KLqp(params, data)
        inference.run(n_samples=5, n_iter=2500)

        self.is_fit = True

    def fit_map(self, y_train):
        """Fits the model's maximum a posterior."""
        if self.is_fit:
            msg = 'model already fit'
            raise ValueError(msg)

        self.y = NormalWithSoftplusScale(
            loc=self.mu,
            scale=self.inv_softplus_sigma,
            sample_shape=y_train.shape,
        )
        data = {self.y: y_train}
        inference = ed.MAP([self.mu, self.inv_softplus_sigma], data)
        inference.run(n_iter=1000)

        self.is_fit = True


    def sample(self, sample_shape=1):
        """Samples from the fit model.

        Requires no data be input, since this is a purely generative model.
        """
        if not self.is_fit:
            msg = 'Can not sample from unfit model.'
            raise ValueError(msg)
        return self.y.sample(sample_shape).eval()





