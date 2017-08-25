"""Recovering normal distribution parameters in pystan."""
import pystan
import numpy as np

MU = 6.0
SIGMA = 1.5 # sigma = softplus(inv_softplus_sigma)
N = 1000

model = """
data {
    int<lower=0> N;
    real y[N];
}
parameters {
    real mu;
    real<lower=0> inv_softplus_sigma;
}
transformed parameters {
    real<lower=0> sigma;
    sigma = log(1+exp(inv_softplus_sigma));
}
model {
    mu ~ normal(0.0, 5.0);
    inv_softplus_sigma ~ normal(0.0, 1.0);
    y ~ normal(mu, sigma);
}
"""

# Data
y_train = np.random.normal(MU, SIGMA, N)
data = {'y': y_train, 'N': N}

# Model
sm = pystan.StanModel(model_code=model)

print('Sampling based approach')
fit = sm.sampling(data=data, iter=2000, chains=4)
print(fit)

print('Variational based approach')
fit = sm.vb(data=data, iter=150000)
print(fit['mean_pars'])
