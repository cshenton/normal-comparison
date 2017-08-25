# Python PPL comparison

This is a comparison of modelling shape and scale parameter uncertainty
with the following probabilistic programming libraries in python:

- `edward`
- `pymc3`
- `pystan`

Each example generates samples from a normal distribution, then attempts
to recover their values, accounting for uncertainty in the mean and std
deviation of the distribution.

## Sampling and Variational Inference

For each library, I record both a sampling based approach and an ADVI based
approach. By default, the libraries use different samplers and ADVI algorithms,
but *a priori* I don't expect these to make much of a difference for such
a simple example.

## Prior Distributions and Model

Edward is seriously lacking in available distribution. In particular, there
are no Half Normal or Half Cauchy distributions, which are typical priors
for a normal scale parameter. In addition, there are known issues in edward
with [using an inverse gamma distribution](https://discourse.edwardlib.org/t/a-toy-normal-model-failed-klqp-and-why/253/2) when doing variational inference.

We'll note this as a downside to using edward, but in the interest of a
fair comparison, we adopt the following functional forms and prior values
for the latent variables and model.

```python
# priors
mu ~ normal(0.0, 5.0)
inv_softplus_sigma ~ normal(0.0, 1.0)

# model
y ~ normal(mu, softplus(inv_softplus_sigma))
```

And where using VI, I set the family of posterior distributions for `mu`
and `inv_softplus_sigma` to be the family of normal distributions.

In particular, when using variational inference, I initialise the distributions
at the vaules of the priors. This is not done in the edward tutorials, and
seems to have the effect of providing unintentionally informative priors.

## Installation

Just run

```bash
pip install -r requirements.txt
```

## Running

To run a particular example, just run (for example):

```bash
python normal_edward.py
```

from the root of the repository.
