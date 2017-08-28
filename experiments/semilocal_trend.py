"""Attempt at implementing a reusable API for a pymc model."""
import numpy as np
import pymc3 as pm
import theano.tensor as tt

from pymc3.distributions import distribution

class SemiLocalTrend(distribution.Continuous):
    """Distribution of semi local hidden trends.

    Describes distributions of sequences delta, where delta evolves according
    to the equation:

        delta_t+1 = trend + phi(delta_t - trend) + eta_t
        delta_t+1 = (1-phi)*trend + phi*delta_t + eta_t
    OR
        beta_t+1 = phi*beta_t + eta_t
        delta_t = beta_t + trend*t (factor out deterministic trend)

    Where trend describes a long term trend, phi is an AR1 coefficient, and
    eta_t is a gaussian error term.

    Attributes:
        trend (float): The long term trend.
        phi (float): Effect of lagged value on current value.
        eta_sd (float): The standard deviation of the error term eta.
    """
    def __init__(self, trend, phi, eta_sd, *args, **kwargs):
        super(SemiLocalTrend, self).__init__(*args, **kwargs)
        self.trend = tt.as_tensor_variable(trend)
        self.phi = tt.as_tensor_variable(phi)
        self.eta_sd = tt.as_tensor_variable(eta_sd)

    def logp(self, x):
        """The log probability of observing sequence x."""
        x_init = x[0]
        x_now = x[1:]
        x_prev = x[:-1]

        # Log probability of the first observation
        logp_initial = pm.Normal.dist(mu=0.0, sd=eta_sd).logp(x_init)

        # Log probability of the subsequent sequence
        location = self.trend + self.phi*(x_prev - self.trend)
        logp_sequence = pm.Normal.dist(mu=location, sd=eta_sd).logp(x_now)

        logp_total = logp_initial + tt.sum(logp_sequence)

        return logp_total


# semilocal linear trend

# level is a random walk
# slope is an ar1 process centered around D.

# This imposes some stability on the long term series, still allows for
# sustained periods of above average growth. More complex models could allow
# D to slowly change.

# level_t+1 = level_t + slope_t + epsilon_t
# slope_t+1 = D + phi*(slope_t - D) + eta_t
# Define in terms of (mean 0) cumulative slope deviation
# slope_t = slope_dev_t + D
# slope_dev_t+1 = phi*slope_dev_t + eta_t
# So level_t+1 = level_t + slope_dev_t + D + epsilon_t
# level sequence = random walk + cumsum(trend_dev) + cumsum(trend_mean) + noise
# let intercept / start be absorbed by seasonals / holiday

# So in general
# level = random process + cumsum(random process) + cumsum(random proc) + ...

# where we eventually just set one of those to bottom out as deterministic
# so here
# - deviations around the trend are random
# - the trend itself it random
# - the trend of the trend is deterministic
# and deviations themselves are just the cumsum of a gaussian error.


# Then we could estimate with y_t = level_t + err_t for example
# This allows for local linera trend (phi=1) or local level (phi=D=0)

# How can we fix these distributions to have a stable mean?


class AR1(distribution.Continuous):
    """An Autoregressive distribution order 1.

    A distribution of sequences of AR1 processes specified by the equations.

        theta_t+t = phi*theta_t + epsilon_t
        theta_0 = epsilon_0
        epsilon_t ~ normal(0, eps_sd)

    The generating process supports generating and computing log probabilities
    of sequences of arbitrary length. Also holds a custom numpy ufunc that
    allows for discounted cumulative sums with factor phi.

    Attributes:
        phi (float): The persistence term.
        eps_sd (float): The standard deviation of the noise term.
    """
    def __init__(self, phi, eps_sd, *args, **kwargs):
        super(AR1, self).__init__(*args, **kwargs)
        self.phi = tt.as_tensor_variable(phi)
        self.eps_sd = tt.as_tensor_variable(eps_sd)
        self.disc_add = np.frompyfunc(lambda x1, x2: x1*phi + x2, 2, 1)

    def logp(self, x):
        """The log probability of a single obserbed sequence x."""
        x_init = x[0]
        x_now = x[1:]
        x_prev = x[:-1]

        # Log probability of the first observation
        logp_initial = pm.Normal.dist(mu=0.0, sd=self.eps_sd).logp(x_init)

        # Log probability of the subsequent sequence
        location = self.phi * x_prev
        logp_sequence = pm.Normal.dist(mu=location, sd=self.eps_sd).logp(x_now)

        logp_total = logp_initial + tt.sum(logp_sequence)

        return logp_total

    def random(self, length, size):
        """Generates draws of sequences from the distribution.

        Achieves this by drawing sequences of gaussian noise using the sd
        of the distribution, and taking the discounted cumulative sum of
        the noise over the length of the sequence.

        Args:
            length (int): The length of sequences to draw.
            shape (int or list of ints): The sample shape to draw.

        Returns:
            numpy.ndarray: An array of dimensions shape * length.
        """
        shape = list(size) + [length]
        x_base = pm.Normal.dist(mu=0.0, sd=self.eps_sd).random(size=shape)
        # Type casting necessary since we accumulate an autogenerated ufunc
        result = self.disc_add.accumulate(
            x_base, axis=-1, dtype=np.object).astype(np.float)
        return result

class NestedAR1(distribution.Continuous):
    """A Recursive Autoregressive distribution order 1.

    This is equivalent to the locally semilinear trend distribution specified
    in R's bsts and python's statsmodels. It is a  distribution of discounted
    cumulative sums of sequences of AR1 processes specified by the equations:

        theta_t+t = phi_1*theta_t + epsilon_t
        theta_0 = epsilon_0
        epsilon_t ~ ar1(phi_2, eps_sd)

    Attributes:
        rho (float): The persistence term of the outer distribution.
        phi (float): The persistence term of the inner distribution.
        eps_sd (float): The standard deviation of the noise term of the
            inner distribution.
        inner (AR1): The inner distribution.
    """
    def __init__(self, rho, phi, eps_sd, *args, **kwargs):
        super(NestedAR1, self).__init__(*args, **kwargs)
        self.rho = tt.as_tensor_variable(rho)
        self.inner = AR1.dist(phi=phi, eps_sd=eps_sd)
        self.disc_add = np.frompyfunc(lambda x1, x2: x1*rho + x2, 2, 1)

    def logp(self, x):
        """The log probability of a single obserbed sequence x."""
        x_init = x[0]
        x_now = x[1:]
        x_prev = x[:-1]

        # Log probability of the first observation
        logp_initial = pm.Normal.dist(mu=0.0, sd=self.eps_sd).logp(x_init)

        # Log probability of the subsequent sequence
        diffs = x_now - rho*x_prev
        logp_sequence = self.inner.logp(diffs)

        logp_total = logp_initial + tt.sum(logp_sequence)
        return logp_total

    def random(self, length, size):
        """Generates draws of sequences from the distribution.

        Achieves this by drawing sequences of gaussian noise using the sd
        of the distribution, and taking the discounted cumulative sum of
        the noise over the length of the sequence.

        Args:
            length (int): The length of sequences to draw.
            shape (int or list of ints): The sample shape to draw.

        Returns:
            numpy.ndarray: An array of dimensions shape * length.
        """
        x_base = self.inner.random(length=length, size=size)
        # Type casting necessary since we accumulate an autogenerated ufunc
        result = self.disc_add.accumulate(
            x_base, axis=-1, dtype=np.object).astype(np.float)
        return result

# nest_t+1 = nest_t + eps_t
# where eps_t ~ ar1(phi, eps_sd)

# outer_t+1 = phi*outer_t + inner_t + epsilon_t
# inner_t+1 = rho*inner_t + eta_t
# outer_t+1 = phi*outer_t + rho*inner_t-1 + eta_t-1 + epsilon_t

# so outer_t+1 is normally distributed with loc phi*outer + inner and var sd_ep



class SemiLocalModel():
    """A Bayesian Structural Semi-local model.

    The equations for the model are:

        y_t+1 = y_t + delta_t + epsilon_t
        delta_t+1 = D + phi(delta_t - D) + eta_t

    Where epsilon and eta are uncorrelated gaussian error terms, D is the
    long run slope, and phi is the AR1 parameter, which should be stationary.
    Given data series Y, we would like to estimate D, phi, and the variances
    of the two error terms.
    """
    def __init__(self):
        with pm.Model() as self.model:
            # priors
            slope = pm.Normal(0, 5)
            phi = pm.Normal(0, 0.3)
            scale_eps = pm.Normal(0, 1)
            scale_eta = pm.Normal(0, 1)

            # now we need to describe the likelihood of a series y
            # y can be specified in vector form as a function of delta
            # delta can be specified sans data as an AR process

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
