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

Edward has fewer available distributions that the alternatives. In particular, there are no Half Normal or Half Cauchy distributions, which are typical priors
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
and `inv_softplus_sigma` to be the family of normal distributions. In
particular, when using variational inference, I initialise the distributions
at the values of the priors. This is not done in the edward tutorials, and
seems to have the effect of providing unintentionally informative priors,
which is the motivation for this comparison.

## Installation

Just run

```bash
git clone git@github.com:cshenton/normal-comparison.git
cd normal-comparison
pip install -r requirements.txt
```

## Running

To run a particular example, just run (for example):

```bash
python normal_edward.py
```

from the root of the repository.

# Results

## PyMC3

- No surprises
- Both sampling and ADVI recover parameters
- Good balance of features for sampling and ADVI methods

PyMC3 provides a pretty pain-free API for implementing both sampling and
ADVI based approaches. In particular, for this simple example, sensible
defaults mean we don't have to manually specify a family of posterior
distributions for ADVI.

It also easily recovers the distribution parameters in both cases. It's
a shame that PyMC3 doesn't use tensorflow for the backend, since compatibility
with tooling like Keras and Tensorboard is one of the biggest pluses for
Edward.

## PyStan

- Painful compile times slow debugging
- Both sampling and VI recover parameters
- VI is currently experimental, no nice output format
- Model code is both terse and readable

Stan's advantage is portability. Model specifications can work it the command
line, python, R, and Julia front ends. This is great for research, but a
pain for writing unit tested code that's going to run on a server.

In addition, it is the most focussed on sampling based approaches, and as
a result its Variational Bayes estimator is currently experimental.

I'll probably not be investing time into writing stan code, because the
Theano and Tensorflow backends of PyMC3 and Edward come with great tooling,
and benefit from the externality of a wider community than just people doing
bayesian inference.

## Edward

- Great API
- Posterior Definition is a little verbose
- VI Strugges to converge on scale parameter
- Sampling methods don't seem to work

Edward has by far my favourite API of the three. So it was unfortunate that
the estimated scale parameter varied so much from run to run. This is as compared to the ADVI routine in PyMC3, which produced mean estimates of
the scale parameter within 0.01 of eachother from run to run.

In addition, I could not get the sampling methods to work. This should
serve more as an indication of how thin the documentation is on sampling
methods in Edward, rather than the underlying library's correctness.
