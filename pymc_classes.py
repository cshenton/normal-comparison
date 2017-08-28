"""Attempt at implementing a reusable API for a pymc model."""
import numpy as np
import pymc3 as pm
import theano.tensor as tt

class NormalModel():
    """A normal distribution model."""
    def __init__(self):
        self.is_fit = False

        with pm.Model() as self.model:
            self.mu = pm.Normal('mu', mu=0.0, sd=5.0)
            self.inv_softplus_sigma = pm.Normal(
                'inv_softplus_sigma', mu=0.0, sd=1.0)

    def fit(self, y_train):
        """Fits the normal distribution to the data in numpy array y.

        Once this method is run, the variational model for the parameters
        will be adjusted to match the posterior distribution.
        """
        if self.is_fit:
            msg = 'model already fit'
            raise ValueError(msg)

        self.is_fit = True

    def fit_map(self, y_train):
        """Fits the model's maximum a posterior."""
        if self.is_fit:
            msg = 'model already fit'
            raise ValueError(msg)
        with self.model:
            self.y = pm.Normal(
                'y',
                mu=self.mu,
                sd=tt.nnet.softplus(self.inv_softplus_sigma),
                observed=y_train
            )
            params = pm.find_MAP(model=self.model)

        print(params)
        self.is_fit = True

    def sample(self, sample_shape=1):
        """Samples from the fit model.

        Requires no data be input, since this is a purely generative model.
        """
        if not self.is_fit:
            msg = 'Can not sample from unfit model.'
            raise ValueError(msg)
        return self.y.sample(sample_shape).eval()
