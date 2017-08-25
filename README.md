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

Edward has fewer available distributions than the alternatives. In particular, there are no Half Normal or Half Cauchy distributions, which are typical priors
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

And where using variational inference, I set the family of posterior distributions for `mu` and `inv_softplus_sigma` to be the family of normal distributions, or whatever the software default is otherwise.

If the package requires explicitly initialising the posterior family, I
initialise it at the value of the priors.

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

from the root of the repository. The output format is not consistent between
the packages, but rather than coercing them into a similar format, I leave
them as they are, as an illustration of how easy it is to inspect results
in each package.

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

- Compile times slow down the debugging process
- Both sampling and VI recover parameters
- VI is currently experimental, no nice output format
- Model code is both terse and readable

Stan's advantage is portability. Model specifications can work with the command
line, python, R, and Julia front ends. This is great for research, but a
pain for writing unit tested code that's going to run on a server.

Out of the three, it is the most focussed on sampling based approaches,
and as a result its Variational Bayes estimator is currently experimental.
This will likely change in the next major release, but means it's harder
to be agnostic to your method of inference within the framework.

However, stan still feels best suited to research, and lacks the production
tooling that that tensorflow and theano backends give edward and pymc3.

## Edward

- Posterior Definition is a little verbose
- VI Strugges to converge on scale parameter
- Sampling methods don't seem to work

I really like the high level desin of edward's API. So it was unfortunate
that the estimated scale parameter varied so much from run to run. This
is as compared to the ADVI routine in PyMC3, which produced mean estimates
of the scale parameter within 0.01 of eachother from run to run.

The linear and mixed effects regression examples on the project's website
simply omit uncertainty in the scale parameter, so do not have this issue.
This was actually the motivation for writing this comparison in the first
place. Either the KLqp algorithm has difficulty dealing with scale parameter
uncertainty, or I've implemented things incorrectly.

In addition, I could not get the sampling methods to work. This should
serve more as an indication of how thin the documentation is on sampling
methods in Edward, rather than the underlying library's correctness.
