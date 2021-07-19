""" This script generates the figures used in the implementation example.
"""

import os
import scipy.stats as ss
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import timeit
import numpy as np
import elfi

import matplotlib
matplotlib.rcParams['text.usetex'] = True

prepath = './../images/'
savefig = True

np.random.seed(21)
class Prior:
    def rvs(self, size=None, random_state=None):
        # size from (BS,) -> (BS,1)
        if size is not None:
            size = np.concatenate((size, [1]))
        return ss.uniform(loc=-2.5, scale=5).rvs(size=size, random_state=random_state)

    def pdf(self, theta):
        return ss.uniform(loc=-2.5, scale=5).pdf(theta)

    def logpdf(self, theta):
        return ss.uniform(loc=-2.5, scale=5).logpdf(theta)


class Likelihood:
    r"""Implements the distribution
    P(x|theta) = N(x; theta^4, 1)         if theta in [-0.5, 0.5]
                 N(x; theta + 0.5 + 0.5^4 if theta > 0.5
                 N(x; theta - 0.5 + 0.5^4 if theta < 0.5
    """

    def rvs(self, theta, seed=None):
        """Vectorized sampling from likelihood.

        """
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(np.float)
        samples = np.empty_like(theta)

        c = 0.5 - 0.5 ** 4

        tmp_theta = theta[theta <= -0.5]
        samples[theta <= -
                0.5] = ss.norm(loc=-tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta <= -0.5] = np.inf

        tmp_theta = theta[theta <= 0.5]
        samples[theta <= 0.5] = ss.norm(
            loc=tmp_theta**4, scale=1).rvs(random_state=seed)
        theta[theta <= 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(
            loc=tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta < np.inf] = np.inf

        assert np.allclose(theta, np.inf)
        return samples

    def pdf(self, x, theta):
        """Vectorised pdf evaluation.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 1
        assert x.ndim == 1

        BS = theta.shape[0]
        N = x.shape[0]
        theta = theta.astype(np.float)

        pdf_eval = np.zeros((BS))
        c = 0.5 - 0.5 ** 4

        def help_func(lim, mode):
            tmp_theta = theta[theta <= lim]
            tmp_theta = np.expand_dims(tmp_theta, -1)
            scale = np.ones_like(tmp_theta)
            if mode == 1:
                pdf_eval[theta <= lim] = np.prod(
                    ss.norm(loc=-tmp_theta - c, scale=scale).pdf(x), 1)
            elif mode == 2:
                pdf_eval[theta <= lim] = np.prod(
                    ss.norm(loc=tmp_theta**4, scale=scale).pdf(x), 1)
            elif mode == 3:
                pdf_eval[theta <= lim] = np.prod(
                    ss.norm(loc=tmp_theta - c, scale=scale).pdf(x), 1)
            theta[theta <= lim] = np.inf

        big_M = 10**7
        help_func(lim=-0.5, mode=1)
        help_func(lim=0.5, mode=2)
        help_func(lim=big_M, mode=3)
        assert np.allclose(theta, np.inf)
        return pdf_eval


def summary(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return np.prod(x, 1)


def create_factor(x):
    """Creates the function g(theta) = L(theta)*prior(theta).

    """
    lik = Likelihood()
    pr = Prior()

    def tmp_func(theta):
        return float(lik.pdf(x, np.array([theta])) * pr.pdf(theta))
    return tmp_func


def approximate_Z(func, a, b):
    """Approximates the partition function with exhaustive integration.
    """
    return integrate.quad(func, a, b)[0]


def create_gt_posterior(likelihood, prior, data, Z):
    """Returns a function that computes the gt posterior
    """
    def tmp_func(theta):
        return likelihood.pdf(data, np.array([theta])) * prior.pdf(np.array([theta])) / Z
    return tmp_func


data = np.array([0.])
dim = data.shape[0]
a = -2.5  # integration left limit
b = 2.5   # integration right limit

likelihood = Likelihood()
prior = Prior()

factor = create_factor(x=data)
Z = approximate_Z(factor, a, b)
gt_posterior_pdf = create_gt_posterior(likelihood, prior, data, Z)


############# PLOT GROUND TRUTH ##################
plt.figure()
plt.title("Ground-truth PDFs")
plt.xlim(-3, 3)
plt.xlabel(r'$\theta$')
plt.ylabel(r'density')
plt.ylim(0, .6)

# plot prior
theta = np.linspace(-3, 3, 200)
y = prior.pdf(theta)
plt.plot(theta, y, 'b-.', label=r'Prior: $p(\theta))$')

# plot likelihood
y = likelihood.pdf(x=data, theta=theta)
plt.plot(theta, y, 'g-.', label=r'Likelihood: $p(y_0|\theta))$')

# plot posterior
y = np.squeeze(np.array([gt_posterior_pdf(th) for th in theta]))
plt.plot(theta, y, 'r-.', label=r'Posterior: $p(\theta|y_0)$')

plt.legend()
if savefig:
    plt.savefig(os.path.join(
        prepath, "chapter3/example_gt.png"), bbox_inches='tight')
plt.show(block=False)


############# DEFINE ELFI MODEL ##################
def simulator(theta, dim, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    theta = np.repeat(theta, dim, -1)
    return likelihood.rvs(theta, seed=random_state)


elfi.new_model("1D_example")
elfi_prior = elfi.Prior(Prior(), name="theta")
elfi_simulator = elfi.Simulator(
    simulator, elfi_prior, dim, observed=np.expand_dims(data, 0), name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

# left_lim = np.array([-2.5])
# right_lim = np.array([2.5])
bounds = [(-2.5, 2.5)]
dim = data.shape[-1]

# Defines the ROMC inference method
romc = elfi.ROMC(dist, bounds)

############# TRAINING ###################
n1 = 500
seed = 21
romc.solve_problems(n1=n1, seed=seed, use_bo=False, optimizer_args=None)
if savefig:
    romc.distance_hist(bins=100,
                       savefig=os.path.join(prepath, "chapter3/example_theta_dist.png"))
else:
    romc.distance_hist(bins=100)
    
eps = .75
romc.estimate_regions(eps_filter=eps, use_surrogate=False,
                      region_args=None, fit_models=False)
if savefig:
    romc.visualize_region(0,
                         savefig=os.path.join(prepath, "chapter3/example_region.png"))
else:
    romc.visualize_region(0)

    
############# INFERENCE ##################
n2 = 50
tmp = romc.sample(n2=n2)
if savefig:
    romc.visualize_region(i=0, savefig=os.path.join(
        prepath, "chapter3/example_region_samples.png"))
else:
    romc.visualize_region(i=0)

romc.result.summary()

# compute expectation
print("Expected value   : %.3f" %
      romc.compute_expectation(h=lambda x: np.squeeze(x)))
print("Expected variance: %.3f" %
      romc.compute_expectation(h=lambda x: np.squeeze(x)**2))


############# PLOT HISTOGRAM OF SAMPLES  ##################
plt.figure()
plt.title("Histogram of the samples drawn")
plt.hist(x=romc.result.samples_array,
         weights=romc.result.weights,
         bins=70, density=True, range=(-3, 3))
theta = np.linspace(-3, 3, 60)
y = np.squeeze(np.array([gt_posterior_pdf(th) for th in theta]))
plt.plot(theta, y, 'r-.', label="True Posterior")
plt.xlabel(r'$\theta$')
plt.ylabel(r'density')
plt.ylim([0, .6])
if savefig:
    plt.savefig(os.path.join(
        prepath, "chapter3/example_marginal.png"), bbox_inches='tight')
plt.show(block=False)


############# PLOT POSTERIOR ##################
plt.figure()
plt.title("Approximate Posterior")
plt.xlim(-3, 3)
plt.xlabel(r'$\theta$')
plt.ylabel("density")
plt.ylim(0, .6)

# plot histogram of samples
plt.hist(x=romc.result.samples_array,
         weights=romc.result.weights,
         bins=50, density=True, range=(-3, 3),
         facecolor='y', alpha=.5, label="samples histogram")

# plot prior
theta = np.linspace(-3, 3, 150)
y = prior.pdf(theta)
plt.plot(theta, y, 'b-.', label='Prior')

# plot likelihood
y = likelihood.pdf(x=data, theta=theta)
plt.plot(theta, y, 'g-.', label='Likelihood')

# plot ROMC posterior
y = [romc.eval_posterior(np.array([[th]])) for th in theta]
tmp = np.squeeze(np.array(y))
plt.plot(theta, tmp, '-.o', color="navy", label="ROMC Posterior")

# plot posterior
y = np.squeeze(np.array([gt_posterior_pdf(th) for th in theta]))
plt.plot(theta, y, 'r-.', label="True Posterior")


plt.legend()
if savefig:
    plt.savefig(os.path.join(
        prepath, "chapter3/example_posterior.png"), bbox_inches='tight')
plt.show(block=False)


# Evaluation
# compute divergence
def wrapper(x):
    """gt_posterior_pdf with batching.
    """
    res = []
    for i in range(x.shape[0]):
        tmp = x[i]
        res.append(gt_posterior_pdf(float(tmp)))
    return np.array(res)


print(romc.compute_divergence(wrapper, distance="Jensen-Shannon"))

# compute ESS
print("Nof Samples: %d, ESS: %.3f" %
      (len(romc.result.weights), romc.compute_ess()))
