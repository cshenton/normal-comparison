"""Recovering normal distribution parameters in pymc3."""
import numpy as np
import pymc3 as pm

MU = 0.0
SIGMA = 1.5
N = 1000

def sample_data(sample_size):
    return np.random.normal(MU, SIGMA, sample_size)

# Data
y_train = sample_data(T)

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    mu = pm.Normal('alpha', mu=0, sd=5)
    sigma = pm.HalfNormal('sigma', sd=2.5)

    # Likelihood (sampling distribution) of observations
    y = pm.Normal('y', mu=mu, sd=sigma, observed=y_train)
