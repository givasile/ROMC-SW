from elfi.methods.inference.romc import OptimisationProblem
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from functools import partial

import os
import timeit
import time
import scipy.stats as ss
import scipy.integrate as integrate
import logging
import matplotlib.pyplot as plt
from elfi.examples import ma2
import numpy as np
import elfi
import matplotlib
import random

random.seed(21)

# matplotlib.rcParams['text.usetex'] = True
logging.basicConfig(level=logging.CRITICAL)
savefig = False

######################################################################
# Part 1. Generates the subfigures used in the running example in the#
# presentation of our implementation i.e. figure 3                   #
######################################################################

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
        """Vectorised sampling from likelihood.

        """
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(float)
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
        theta = theta.astype(float)

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

prepath = './images/'
np.random.seed(21)
data = np.array([0.])
dim = data.shape[0]
a = -2.5  # integration left limit
b = 2.5   # integration right limit

likelihood = Likelihood()
prior = Prior()

factor = create_factor(x=data)
Z = approximate_Z(factor, a, b)
gt_posterior_pdf = create_gt_posterior(likelihood, prior, data, Z)


# plot ground-truth prior, likelihood, posterior
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
        prepath, "chapter3/example_gt.pdf"), bbox_inches='tight')
plt.show(block=False)


# define elfi model
def simulator(theta, dim, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    theta = np.repeat(theta, dim, -1)
    return likelihood.rvs(theta, seed=random_state)

elfi.new_model("1D_example")
elfi_prior = elfi.Prior(Prior(), name="theta")
elfi_simulator = elfi.Simulator(
    simulator, elfi_prior, dim, observed=np.expand_dims(data, 0), name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")
bounds = [(-2.5, 2.5)]
dim = data.shape[-1]

# define romc inference
romc = elfi.ROMC(dist, bounds)

print("\n######################################################################")
print("PART 1 - Generates all and outputs until page 19 (running example of the paper)\n")

# Example - Training part
# Training (fitting) part
n1 = 500
seed = 21
use_bo = False

# Training step-by-step
romc.solve_problems(n1=n1, seed=seed, use_bo=use_bo, optimizer_args=None)

eps = .75
romc.estimate_regions(eps_filter=eps, use_surrogate=False,
                      region_args=None, fit_models=False)

romc.visualize_region(i=2)

# Inference part
seed = 21
n2 = 50
romc.sample(n2=n2, seed=seed)

# visualize_region, adding the samples now
romc.visualize_region(i=2)

# visualize marginal (built-in ELFI tool)
weights = romc.result.weights
romc.result.plot_marginals(weights=weights, bins=100, range=(-3, 3))
plt.show(block=False)

# summarize the samples (built-in ElFI tool)
romc.result.summary()
# Method: ROMC
# Number of samples: 19300
# Parameter                Mean               2.5%              97.5%
# theta:                 -0.012             -1.985              1.987

# compute expectation
exp_eval = romc.compute_expectation(h=lambda x: np.squeeze(x))
print("Expected value   : %.3f" % exp_eval)
# Expected value: -0.012

exp_var = romc.compute_expectation(h=lambda x: np.squeeze(x)**2)
print("Expected variance: %.3f" % exp_var)
# Expected variance: 1.120

# eval unnorm posterior
print("%.3f" % romc.eval_unnorm_posterior(theta = np.array([[0]])))
# 58.800

# check eval posterior
print("%.3f" % romc.eval_posterior(theta = np.array([[0]])))
# 0.289

# plot histogram of samples along with the ground-truth prior, likelihood, posterior
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
        prepath, "chapter3/example_posterior.pdf"), bbox_inches='tight')
plt.show(block=False)


# Evaluating the approximate posterior, comparing with the ground truth
def wrapper(x):
    """Wrapper for enabling batching in the ground truth posterior.
    """
    res = []
    for i in range(x.shape[0]):
        tmp = x[i]
        res.append(gt_posterior_pdf(float(tmp)))
    return np.array(res)

# 4.3 Evaluation part
res = romc.compute_divergence(wrapper, distance="Jensen-Shannon")
print("Jensen-Shannon divergence: %.3f" % res)
# Jensen-Shannon divergence: 0.035

# compute ESS
nof_samples = len(romc.result.weights)
ess = romc.compute_ess()
print("Nof Samples: %d, ESS: %.3f" % (nof_samples, ess))
# Nof Samples: 19300, ESS: 16196.214

print("######################################################################\n")

######################################################################
# Part 2. Generates the subfigures used in the gradient-based        #
# optimisation version of the example at chapter 5, i.e. figures 5, 6#
######################################################################

print("\n##############################################################")
print("PARTS 2,3,4 - Generate all figures and outputs for Section 5\n")


# set the path for storing images
prepath = './images/chapter4/'

# helping functions for ploting the marginal histogram of the samples
def plot_marginal(samples, weights, mean, std, title, xlabel, ylabel, bins, range, ylim, savepath):
    plt.figure()
    plt.title(title)
    plt.hist(x=samples,
             weights=weights,
             bins=bins, density=True, range=range)
    plt.axvline(mean, 0, 1,
                color="r", linestyle="--", label=r"$\mu = %.3f$" % (mean))
    plt.axhline(1,
                (mean-std-range[0])/(range[1] - range[0]),
                (mean+std-range[0])/(range[1] - range[0]),
                color="k",
                linestyle="--", label=r"$\sigma = %.3f$" % (std))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
        # tplt.save(savepath[:-3] + "tex")
    plt.show(block=False)

# helping functions for ploting the 2D posterior pdf
def plot_romc_posterior(title, posterior, nof_points, savefig):
    plt.figure()
    th1 = np.linspace(bounds[0][0], bounds[0][1], nof_points)
    th2 = np.linspace(bounds[1][0], bounds[1][1], nof_points)
    X, Y = np.meshgrid(th1, th2)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    th = np.stack((x_flat, y_flat), -1)
    z_flat = posterior(th)
    Z = z_flat.reshape(nof_points, nof_points)

    plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.title(title)
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.colorbar()
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
        # tplt.save(savefig[:-3] + "tex")
    plt.show(block=False)

# Set seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

n1 = 100
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .01
vis_ind_1 = 1

print("\n######################################################################")
print("PART 2 - Fit the gradient-based ROMC for Section 5\n")

# define model
romc = elfi.ROMC(model, bounds=bounds, discrepancy_name="d")

# fitting part
romc.solve_problems(n1=n1, seed=seed)
romc.estimate_regions(eps_filter=eps, fit_models=True)

# sampling part
tmp = romc.sample(n2=n2, seed=seed)
if savefig:
    romc.visualize_region(vis_ind_1, savefig=os.path.join(
        prepath, "ma2_region_1.pdf"))
else:
    romc.visualize_region(vis_ind_1)

# plot/save the marginal posterior distribution
name = "t1"
if savefig:
    plot_marginal(romc.result.samples["t1"], romc.result.weights,
                  romc.result.sample_means_array[0],
                  np.sqrt(romc.result.samples_cov()[0, 0]),
                  r"ROMC (gradient-based) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc.pdf"))
else:
    plot_marginal(romc.result.samples["t1"], romc.result.weights,
                  romc.result.sample_means_array[0],
                  np.sqrt(romc.result.samples_cov()[0, 0]),
                  r"ROMC (gradient-based) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), savepath=savefig)

name = "t2"
if savefig:
    plot_marginal(romc.result.samples["t2"], romc.result.weights,
                  romc.result.sample_means_array[1],
                  np.sqrt(romc.result.samples_cov()[1, 1]),
                  r"ROMC (gradient-based) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc.pdf"))
else:
    plot_marginal(romc.result.samples["t2"], romc.result.weights,
                  romc.result.sample_means_array[1],
                  np.sqrt(romc.result.samples_cov()[1, 1]),
                  r"ROMC (gradient-based) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)


# plot/save the romc posterior
if savefig:
    plot_romc_posterior('ROMC (gradient-based)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=os.path.join(prepath, "mae2_romc_posterior.pdf"))
else:
    plot_romc_posterior('ROMC (gradient-based)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=False)

# save ROMC (gradient-based) results
romc_gb_mu_th1, romc_gb_mu_th2 = romc.result.sample_means_array
romc_gb_sigma_th1 = np.sqrt(romc.result.samples_cov()[0, 0])
romc_gb_sigma_th2 = np.sqrt(romc.result.samples_cov()[1, 1])

print("######################################################################\n")

######################################################################
# Part 3. Generates the subfigures used in the Bayesian optimization #
# version of the example at chapter 5, i.e. figures 5, 6             #
######################################################################

print("\n######################################################################")
print("PART 3 - Fit the Bayesian Optimization ROMC and the Rejection Sampling for Section 5\n")

# seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

x = model.generate(1000, outputs=["t1", "t2"])

n1 = 100
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .03
vis_ind_1 = 1
vis_ind_2 = 3
vis_ind_3 = 12

# Define Model
romc1 = elfi.ROMC(model, bounds=bounds, discrepancy_name="d")

# fitting part
romc1.solve_problems(n1=n1, seed=seed, use_bo=True)
romc1.estimate_regions(eps_filter=eps, use_surrogate=False, fit_models=True)

# sampling part
tmp = romc1.sample(n2=n2, seed=seed)
if savefig:
    romc1.visualize_region(vis_ind_1, savefig=os.path.join(
        prepath, "ma2_region_1_bo.pdf"))
else:
    romc1.visualize_region(vis_ind_1)

# plot/save the marginal posterior distribution
name = "t1"
if savefig:
    plot_marginal(romc1.result.samples["t1"], romc1.result.weights,
                  romc1.result.sample_means_array[0],
                  np.sqrt(romc1.result.samples_cov()[0, 0]),
                  r"ROMC (BO) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc_bo.pdf"))
else:
    plot_marginal(romc1.result.samples["t1"], romc1.result.weights,
                  romc1.result.sample_means_array[0],
                  np.sqrt(romc1.result.samples_cov()[0, 0]),
                  r"ROMC (BO) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), savepath=savefig)

name = "t2"
if savefig:
    plot_marginal(romc1.result.samples["t2"], romc1.result.weights,
                  romc1.result.sample_means_array[1],
                  np.sqrt(romc1.result.samples_cov()[1, 1]),
                  r"ROMC (BO) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc_bo.pdf"))
else:
    plot_marginal(romc1.result.samples["t2"], romc1.result.weights,
                  romc1.result.sample_means_array[1],
                  np.sqrt(romc1.result.samples_cov()[1, 1]),
                  r"ROMC (BO) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)

# plot/save the romc posterior with BO
if savefig:
    plot_romc_posterior('ROMC (BO)',
                        romc1.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=os.path.join(prepath, "mae2_romc_posterior_bo.pdf"))
else:
    plot_romc_posterior('ROMC (BO)',
                        romc1.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=False)

# save ROMC (Bayesian Optimization) results
romc_bo_mu_th1, romc_bo_mu_th2 = romc1.result.sample_means_array
romc_bo_sigma_th1 = np.sqrt(romc1.result.samples_cov()[0, 0])
romc_bo_sigma_th2 = np.sqrt(romc1.result.samples_cov()[1, 1])


# Perform the inference using rejection sampling
N = 10000
rej = elfi.Rejection(model, discrepancy_name="d", batch_size=10000, seed=seed)
# vis = dict(xlim=[-2, 2], ylim=[-1, 1])
result = rej.sample(N, threshold=.1, vis=None)

# plot/save the marginal posterior distribution using Rejection sampling
name = "t1"
if savefig:
    plot_marginal(result.samples["t1"], None,
                  result.sample_means_array[0],
                  np.std(result.samples_array[:, 0]),
                  r"Rejection ABC - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_rejection.pdf"))
else:
    plot_marginal(result.samples["t1"], None,
                  result.sample_means_array[0],
                  np.std(result.samples_array[:, 0]),
                  r"Rejection ABC - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), savepath=savefig)

name = "t2"
if savefig:
    plot_marginal(result.samples["t2"], None,
                  result.sample_means_array[1],
                  np.std(result.samples_array[:, 1]),
                  r"Rejection ABC - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_rejection.pdf"))

else:
    plot_marginal(result.samples["t2"], None,
                  result.sample_means_array[1],
                  np.std(result.samples_array[:, 1]),
                  r"Rejection ABC - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)

# save Rejection ABC results
romc_rej_mu_th1 = result.sample_means_array[0]
romc_rej_mu_th2 = result.sample_means_array[1] 
romc_rej_sigma_th1 = np.std(result.samples_array[:, 0])
romc_rej_sigma_th2 = np.std(result.samples_array[:, 1])

print("######################################################################\n")

######################################################################
# Part 4. Generates the subfigures used in the Neural Network        #
# extension example of chapter 5, i.e. figures 5, 6                  #
######################################################################

print("\n######################################################################")
print("PART 4 - Fit the Neural Network ROMC for Section 5\n")

# seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

# Define a custom optimization class
dim = 2
n1 = 100
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .01

# the custom optimization class
class CustomOptim(OptimisationProblem):
    def __init__(self, **kwargs):
        super(CustomOptim, self).__init__(**kwargs)

    def fit_local_surrogate(self, **kwargs):
        nof_samples = 500
        objective = self.objective

        # helper function
        def local_surrogate(theta, model_scikit):
            assert theta.ndim == 1
            theta = np.expand_dims(theta, 0)
            return float(model_scikit.predict(theta))

        # create local surrogate model as a function of theta
        def create_local_surrogate(model):
            return partial(local_surrogate, model_scikit=model)

        local_surrogates = []
        for i in range(len(self.regions)):
            # prepare dataset
            x = self.regions[i].sample(nof_samples)
            y = np.array([objective(ii) for ii in x])

            # train Neural Network
            mlp = MLPRegressor(hidden_layer_sizes=(10,10), solver='adam')
            model = Pipeline([('linear', mlp)])
            model = model.fit(x, y)
            local_surrogates.append(create_local_surrogate(model))

        self.local_surrogates = local_surrogates
        self.state["local_surrogates"] = True

    # @staticmethod
    # def create_local_surrogate(model):
    #     def _local_surrogate(th):
    #         th = np.expand_dims(th, 0)
    #         return float(model.predict(th))
    #     return _local_surrogate

# initiate ROMC with custom_optim_class
romc = elfi.ROMC(model, bounds=bounds, discrepancy_name="d",
                 custom_optim_class=CustomOptim)

# fitting part
romc.solve_problems(n1=n1, seed=seed)
romc.estimate_regions(eps_filter=eps, fit_models=True)

# sampling part
romc.sample(n2=n2, seed=seed)

# plot/save the marginal posterior distribution
name = "t1"
if savefig:
    plot_marginal(romc.result.samples["t1"], romc.result.weights,
                  romc.result.sample_means_array[0],
                  np.sqrt(romc.result.samples_cov()[0, 0]),
                  r"ROMC (Neural Network) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc_nn.pdf"))
else:
    plot_marginal(romc.result.samples["t1"], romc.result.weights,
                  romc.result.sample_means_array[0],
                  np.sqrt(romc.result.samples_cov()[0, 0]),
                  r"ROMC (Neural Network) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), savepath=savefig)

name = "t2"
if savefig:
    plot_marginal(romc.result.samples["t2"], romc.result.weights,
                  romc.result.sample_means_array[1],
                  np.sqrt(romc.result.samples_cov()[1, 1]),
                  r"ROMC (Neural Network) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc_nn.pdf"))

else:
    plot_marginal(romc.result.samples["t2"], romc.result.weights,
                  romc.result.sample_means_array[1],
                  np.sqrt(romc.result.samples_cov()[1, 1]),
                  r"ROMC (Neural Network) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)

# plot/save the romc posterior
if savefig:
    plot_romc_posterior('ROMC (Neural Network)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=os.path.join(prepath, "mae2_romc_posterior_nn.pdf"))
else:
    plot_romc_posterior('ROMC (Neural Network)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=False)

# save ROMC (Neural Network) results
romc_nn_mu_th1, romc_nn_mu_th2 = romc.result.sample_means_array
romc_nn_sigma_th1 = np.sqrt(romc.result.samples_cov()[0, 0])
romc_nn_sigma_th2 = np.sqrt(romc.result.samples_cov()[1, 1])

print("######################################################################\n")


# print results of Table 1
print("\n\n\n")
print("########################## Table 1 ###########################")
print("Rejection ABC                | %.3f | %.3f | %.3f | %.3f |" % (romc_rej_mu_th1, romc_rej_sigma_th1, romc_rej_mu_th2, romc_rej_sigma_th2))
print("ROMC (gradient-based)        | %.3f | %.3f | %.3f | %.3f |" % (romc_gb_mu_th1, romc_gb_sigma_th1, romc_gb_mu_th2, romc_gb_sigma_th2))
print("ROMC (Bayesian Optimization) | %.3f | %.3f | %.3f | %.3f |" % (romc_bo_mu_th1, romc_bo_sigma_th1, romc_bo_mu_th2, romc_bo_sigma_th2))
print("ROMC (Neural Network)        | %.3f | %.3f | %.3f | %.3f |" % (romc_nn_mu_th1, romc_nn_sigma_th1, romc_nn_mu_th2, romc_nn_sigma_th2))
print("##############################################################")

print("\n##########################################################")


######################################################################
# Part 5; Generates the figures for measuring the execution time.    #
# i.e. figure 7                                                      #
######################################################################

print("\n######################################################################")
print("PART 5 - Generate the figures that measure the execution time\n")

np.random.seed(21)
seed = 21
prepath = './images/chapter4'
model = ma2.get_model(seed_obs=seed)
bounds = [(-2., 2.), (-2., 2.)]


# Execution time using the sequential scheme
eps = .1
n1 = [1, 50, 100, 150, 200]
solve_grad = []
estimate_regions = []
for i, n in enumerate(n1):
    romc = elfi.ROMC(model, bounds, discrepancy_name="d", parallelize=False)

    tic = timeit.default_timer()
    romc.solve_problems(n1=int(n), seed=seed, use_bo=False)
    toc = timeit.default_timer()
    solve_grad.append(toc-tic)

    tic = timeit.default_timer()
    romc.estimate_regions(eps_filter=eps, use_surrogate=False)
    toc = timeit.default_timer()
    estimate_regions.append(toc-tic)

# Execution time using the parallel scheme
estimate_regions_parallel = []
solve_grad_parallel = []
for i, n in enumerate(n1):
    time.sleep(2)
    romc1 = elfi.ROMC(model, bounds, discrepancy_name="d", parallelize=True)

    tic = timeit.default_timer()
    romc1.solve_problems(n1=int(n), seed=seed, use_bo=False)
    toc = timeit.default_timer()
    solve_grad_parallel.append(toc-tic)

    tic = timeit.default_timer()
    romc1.estimate_regions(eps_filter=eps, use_surrogate=False, fit_models=False)
    toc = timeit.default_timer()
    estimate_regions_parallel.append(toc-tic)

# Figure with the exec times for solving the optimisation problems
plt.figure()
plt.title("Optimisation problems")
plt.plot(n1, solve_grad, "bo--", label="sequential")
plt.plot(n1, solve_grad_parallel, "ro--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
if savefig:
    plt.savefig(os.path.join(prepath, "exec_time_solve.pdf"), bbox_inches="tight")
    # tplt.save(os.path.join(prepath, "exec_time_solve.tex"))
plt.show(block=False)

# Figure with the exec times for estimating the proposal regions
plt.figure()
plt.title("Proposal Regions")
plt.plot(n1, estimate_regions, "bo--", label="sequential")
plt.plot(n1, estimate_regions_parallel, "ro--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
if savefig:
    plt.savefig(os.path.join(prepath, "exec_time_regions.pdf"), bbox_inches="tight")
    # tplt.save(os.path.join(prepath, "exec_time_regions.tex"))
plt.show(block=False)

print("\n######################################################################")
