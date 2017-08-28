"""Recovering normal distribution parameters in pymc3."""
import numpy as np
import pymc3 as pm
import theano.tensor as tt

MU = 6.0
SIGMA = 1.5 # sigma = softplus(inv_softplus_sigma)
N = 1000

# Data
y_train = np.random.normal(MU, SIGMA, N)

basic_model = pm.Model()

with basic_model:
    # Priors
    mu = pm.Normal('mu', mu=0.0, sd=5.0)
    inv_softplus_sigma = pm.Normal('inv_softplus_sigma', mu=0.0, sd=1.0)

    # Model
    y = pm.Normal(
        'y', mu=mu, sd=tt.nnet.softplus(inv_softplus_sigma), observed=y_train
    )

    print('Sampling based approach')
    nuts_trace = pm.sample(2000)
    pm.summary(nuts_trace)

    print('Variational based approach')
    result = pm.fit(150000, method='advi')
    advi_trace = result.sample(2000)
    pm.summary(advi_trace)

basic_model = pm.Model()

with basic_model:
    # Priors
    mu = pm.Normal('mu', mu=0.0, sd=5.0)
    inv_softplus_sigma = pm.Normal('inv_softplus_sigma', mu=0.0, sd=1.0)

    # Model
    y = pm.Normal(
        'y', mu=mu, sd=tt.nnet.softplus(inv_softplus_sigma), observed=data
    )

    print('Sampling based approach')
    params = pm.find_MAP(model=basic_model)
    print(params)
