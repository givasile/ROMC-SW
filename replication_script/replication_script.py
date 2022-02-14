# Importing all
from elfi.methods.inference.romc import OptimisationProblem
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from functools import partial

import os
import timeit
import scipy.stats as ss
import scipy.integrate as integrate
import logging
import matplotlib.pyplot as plt
from elfi.examples import ma2
import numpy as np
import elfi
import tikzplotlib as tplt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
logging.basicConfig(level=logging.INFO)
prepath = './images/'
savefig = True

np.random.seed(21)


######################################################################
# This part generates the figures used in the implementation example.#
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
    tplt.clean_figure()
    tplt.save(os.path.join(prepath, "chapter3/example_gt.tex"))
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
    # tplt.clean_figure()
    tplt.save(os.path.join(prepath, "chapter3/example_marginal.tex"))
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
    # tplt.clean_figure()
    tplt.save(os.path.join(prepath, "chapter3/example_post.tex"))
plt.show(block=False)


# Evaluation
def gt_posterior_pdf_with_batching(x):
    """Wrapper for enabling batching in the ground truth posterior.
    """
    res = []
    for i in range(x.shape[0]):
        tmp = x[i]
        res.append(gt_posterior_pdf(float(tmp)))
    return np.array(res)


# compute divergence between romc posterior vs gt_posterior_pdf
print(romc.compute_divergence(gt_posterior_pdf_with_batching, distance="Jensen-Shannon"))

# compute ESS
print("Nof Samples: %d, ESS: %.3f" % (len(romc.result.weights), romc.compute_ess()))


######################################################################
# This part generates the figures used in the gradient-based optimizer
# version of chapter 5 - Use case illustration.
######################################################################
prepath = './images/chapter4/'


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
        tplt.save(savepath[:-3] + "tex")
    plt.show(block=False)

# Set seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

# plot prior samples
x = model.generate(1000, outputs=["t1", "t2"])
plt.figure()
plt.title("Samples from the prior")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.plot(x["t1"], x["t2"], "bo")
if savefig:
    plt.savefig(os.path.join(prepath, "mae2_prior_samples.png"),
                bbox_inches="tight")
    tplt.clean_figure()
    tplt.save(os.path.join(prepath, "ma2_prior_samples.tex"))
plt.show(block=False)

####### ROMC with gradients ################
n1 = 100
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .01
vis_ind_1 = 1
vis_ind_2 = 3
vis_ind_3 = 12

romc = elfi.ROMC(model, bounds=bounds, discrepancy_name="d")
romc.solve_problems(n1=n1, seed=seed)
if savefig:
    romc.distance_hist(savefig=os.path.join(
        prepath, "ma2_distance_hist.png"))
else:
    romc.distance_hist()


romc.estimate_regions(eps_filter=eps, fit_models=True)

tmp = romc.sample(n2=n2, seed=seed)
if savefig:
    romc.visualize_region(vis_ind_1, savefig=os.path.join(
        prepath, "ma2_region_1.png"))
else:
    romc.visualize_region(vis_ind_1)

if savefig:
    romc.visualize_region(vis_ind_2, savefig=os.path.join(
        prepath, "ma2_region_2.png"))
else:
    romc.visualize_region(vis_ind_2)

if savefig:
    romc.visualize_region(vis_ind_3, savefig=os.path.join(
        prepath, "ma2_region_3.png"))
else:
    romc.visualize_region(vis_ind_3)

print(romc.result.summary())
print(romc.result.samples_cov())

name = "t1"
if savefig:
    plot_marginal(romc.result.samples["t1"], romc.result.weights,
                  romc.result.sample_means_array[0],
                  np.sqrt(romc.result.samples_cov()[0, 0]),
                  r"ROMC (gradient-based) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc.png"))
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
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc.png"))
else:
    plot_marginal(romc.result.samples["t2"], romc.result.weights,
                  romc.result.sample_means_array[1],
                  np.sqrt(romc.result.samples_cov()[1, 1]),
                  r"ROMC (gradient-based) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)


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
        tplt.save(savefig[:-3] + "tex")
    plt.show(block=False)


if savefig:
    plot_romc_posterior('ROMC (gradient-based)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=os.path.join(prepath, "mae2_romc_posterior.png"))
else:
    plot_romc_posterior('ROMC (gradient-based)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=False)


######################################################################
# This part generates the figures used in the Bayesian Optimization
# version of chapter 5 - Use case illustration.
######################################################################
# Set seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

# plot prior samples

x = model.generate(1000, outputs=["t1", "t2"])
plt.figure()
plt.title("Samples from the prior")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.plot(x["t1"], x["t2"], "bo")
if savefig:
    plt.savefig(os.path.join(prepath, "mae2_prior_samples.png"),
                bbox_inches="tight")
    tplt.clean_figure()
    tplt.save(os.path.join(prepath, "ma2_prior_samples.tex"))
plt.show(block=False)

n1 = 100
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .03
vis_ind_1 = 1
vis_ind_2 = 3
vis_ind_3 = 12

use_bo = True
romc1 = elfi.ROMC(model, bounds=bounds, discrepancy_name="d")
romc1.solve_problems(n1=n1, seed=seed, use_bo=use_bo)
if savefig:
    romc1.distance_hist(savefig=os.path.join(
        prepath, "ma2_distance_hist_bo.png"))
else:
    romc1.distance_hist()

romc1.estimate_regions(eps_filter=eps, use_surrogate=False, fit_models=True)

tmp = romc1.sample(n2=n2)
if savefig:
    romc1.visualize_region(vis_ind_1, savefig=os.path.join(
        prepath, "ma2_region_1_bo.png"))
else:
    romc1.visualize_region(vis_ind_1)

if savefig:
    romc1.visualize_region(vis_ind_2, savefig=os.path.join(
        prepath, "ma2_region_2_bo.png"))
else:
    romc1.visualize_region(vis_ind_2)

if savefig:
    romc1.visualize_region(vis_ind_3, savefig=os.path.join(
        prepath, "ma2_region_3_bo.png"))
else:
    romc1.visualize_region(vis_ind_3)

print(romc1.result.summary())
print(romc1.result.samples_cov())

name = "t1"
if savefig:
    plot_marginal(romc1.result.samples["t1"], romc1.result.weights,
                  romc1.result.sample_means_array[0],
                  np.sqrt(romc1.result.samples_cov()[0, 0]),
                  r"ROMC (BO) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc_bo.png"))
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
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc_bo.png"))
else:
    plot_marginal(romc1.result.samples["t2"], romc1.result.weights,
                  romc1.result.sample_means_array[1],
                  np.sqrt(romc1.result.samples_cov()[1, 1]),
                  r"ROMC (BO) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)


# rejection
N = 10000
rej = elfi.Rejection(model, discrepancy_name="d", batch_size=10000, seed=seed)
vis = dict(xlim=[-2, 2], ylim=[-1, 1])
result = rej.sample(N, threshold=.1, vis=vis)

name = "t1"
if savefig:
    plot_marginal(result.samples["t1"], None,
                  result.sample_means_array[0],
                  np.std(result.samples_array[:, 0]),
                  r"Rejection ABC - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_rejection.png"))
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
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_rejection.png"))
else:
    plot_marginal(result.samples["t2"], None,
                  result.sample_means_array[1],
                  np.std(result.samples_array[:, 1]),
                  r"Rejection ABC - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)

print(result.summary())
print("%.3f, %.3f" % (np.std(result.samples_array[:, 0]), np.std(result.samples_array[:, 1])))

if savefig:
    plot_romc_posterior('ROMC (BO)',
                        romc1.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=os.path.join(prepath, "mae2_romc_posterior_bo.png"))
else:
    plot_romc_posterior('ROMC (BO)',
                        romc1.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=False)

######################################################################
# This part generates the figures used in the Neural Network extension
# example of chapter 5 - Use case illustration.
######################################################################

# Set seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

# plot prior samples

x = model.generate(1000, outputs=["t1", "t2"])
plt.figure()
plt.title("Samples from the prior")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.plot(x["t1"], x["t2"], "bo")
if savefig:
    plt.savefig(os.path.join(prepath, "mae2_prior_samples_nn.png"),
                bbox_inches="tight")
    tplt.clean_figure()
    tplt.save(os.path.join(prepath, "ma2_prior_samples_nn.tex"))
plt.show(block=False)


####### Define a custom optimization class ################

dim = 2
n1 = 100
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .01
vis_ind_1 = 1
vis_ind_2 = 3
vis_ind_3 = 12


class CustomOptim(OptimisationProblem):
    def __init__(self, **kwargs):
        super(CustomOptim, self).__init__(**kwargs)

    def fit_local_surrogate(self, **kwargs):
        nof_samples = 500
        objective = self.objective

        def local_surrogate(theta, model_scikit):
            assert theta.ndim == 1
            theta = np.expand_dims(theta, 0)
            return float(model_scikit.predict(theta))

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

    @staticmethod
    def create_local_surrogate(model):
        def _local_surrogate(th):
            th = np.expand_dims(th, 0)
            return float(model.predict(th))
        return _local_surrogate


romc = elfi.ROMC(model, bounds=bounds, discrepancy_name="d", custom_optim_class=CustomOptim)
romc.solve_problems(n1=n1, seed=seed)
if savefig:
    romc.distance_hist(savefig=os.path.join(
        prepath, "ma2_distance_hist_nn.png"))
else:
    romc.distance_hist()


romc.estimate_regions(eps_filter=eps, fit_models=True)

romc.sample(n2=n2, seed=seed)
if savefig:
    romc.visualize_region(vis_ind_1, savefig=os.path.join(prepath, "ma2_region_1_nn.png"))
else:
    romc.visualize_region(vis_ind_1)

if savefig:
    romc.visualize_region(vis_ind_2, savefig=os.path.join(prepath, "ma2_region_2_nn.png"))
else:
    romc.visualize_region(vis_ind_2)

if savefig:
    romc.visualize_region(vis_ind_3, savefig=os.path.join(prepath, "ma2_region_3_nn.png"))
else:
    romc.visualize_region(vis_ind_3)

print(romc.result.summary())
print(romc.result.samples_cov())

name = "t1"
if savefig:
    plot_marginal(romc.result.samples["t1"], romc.result.weights,
                  romc.result.sample_means_array[0],
                  np.sqrt(romc.result.samples_cov()[0, 0]),
                  r"ROMC (Neural Network) - $\theta_1$",
                  r"$\theta_1$",
                  r"density",
                  60,
                  (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc_nn.png"))
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
                  (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc_nn.png"))
else:
    plot_marginal(romc.result.samples["t2"], romc.result.weights,
                  romc.result.sample_means_array[1],
                  np.sqrt(romc.result.samples_cov()[1, 1]),
                  r"ROMC (Neural Network) - $\theta_2$",
                  r"$\theta_2$",
                  r"density",
                  60,
                  (-0.5, 1), (0, 3), savepath=savefig)


if savefig:
    plot_romc_posterior('ROMC (Neural Network)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=os.path.join(prepath, "mae2_romc_posterior_nn.png"))
else:
    plot_romc_posterior('ROMC (Neural Network)',
                        romc.eval_unnorm_posterior,
                        nof_points=50,
                        savefig=False)


######################################################################
# This part generates the figures used for the execution time.       #
######################################################################
np.random.seed(21)
prepath = './images/chapter4'

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
    def rvs(self, th1, th2, seed=None):
        assert isinstance(th1, np.ndarray)
        assert isinstance(th2, np.ndarray)
        assert th1.ndim == 2
        assert th2.ndim == 2
        assert np.allclose(th1.shape, th2.shape)

        x = []
        for i in range(th1.shape[0]):
            cur_th = np.concatenate((th1[i], th2[i]))
            x.append(ss.multivariate_normal(
                mean=cur_th, cov=1).rvs(random_state=seed))
        return np.array(x)

    def pdf(self, x, th1, th2):
        assert isinstance(th1, float)
        assert isinstance(th2, float)
        assert isinstance(x, np.ndarray)

        th = np.stack((th1, th2))
        rv = ss.multivariate_normal(mean=th, cov=1)
        nof_points = x.shape[0]
        prod = 1
        for i in range(nof_points):
            prod *= rv.pdf(x[i])
        return prod


def create_factor(x):
    lik = Likelihood()
    pr = Prior()

    def tmp_func(th1, th2):
        return lik.pdf(x, th1, th2) * pr.pdf(th1) * pr.pdf(th2)
    return tmp_func


def approximate_Z(func):
    return integrate.dblquad(func, -2.5, 2.5, lambda x: -2.5, lambda x: 2.5)[0]


def create_gt_posterior(factor, Z):
    def tmp_func(th1, th2):
        return factor(th1, th2) / Z
    return tmp_func


def plot_gt_posterior(posterior, nof_points):
    plt.figure()
    x = np.linspace(-4, 4, nof_points)
    y = np.linspace(-4, 4, nof_points)

    x, y = np.meshgrid(x, y)

    tmp = []
    for i in range(x.shape[0]):
        tmp.append([])
        for j in range(x.shape[1]):
            tmp[i].append(posterior(x[i, j], y[i, j]))
    z = np.array(tmp)
    plt.contourf(x, y, z, 40)
    plt.title('Ground-Truth Posterior PDF')
    plt.colorbar()
    plt.xlabel("th_1")
    plt.ylabel("th_2")
    plt.show(block=False)


def create_gt_marginal_1(factor, Z):
    def tmp_func(th1):
        def integrand(th2):
            return factor(th1, th2)
        return integrate.quad(integrand, -2.5, 2.5)[0] / Z
    return tmp_func


def create_gt_marginal_2(factor, Z):
    def tmp_func(th2):
        def integrand(th1):
            return factor(th1, th2)
        return integrate.quad(integrand, -2.5, 2.5)[0] / Z
    return tmp_func


def plot_gt_marginals(marg_1, marg_2, nof_points):
    plt.figure()
    x = np.linspace(-3, 3, nof_points)
    y = [marg_1(tmp_x) for tmp_x in x]
    plt.plot(x, y, "r--", label=r"$p(\theta_1|y_0)$")
    y = [marg_2(tmp_x) for tmp_x in x]
    plt.plot(x, y, "b--", label=r"$p(\theta_2|y_0)$")
    plt.title("Ground-truth marginals")
    plt.xlabel(r"$\theta_i$")
    plt.ylabel(r"$p(\theta_i|\mathbf{y_0})$")
    plt.legend()
    plt.show(block=False)


def compute_gt_mean(gt_marginal):
    def h(x):
        return gt_marginal(x)*x
    return integrate.quad(h, -2.5, 2.5)


def compute_gt_std(gt_marginal, m):
    def h(x):
        return gt_marginal(x)*((x-m)**2)
    return np.sqrt(integrate.quad(h, -2.5, 2.5))


def plot_marginal(samples, weights, mean, std, marg, title, xlabel, ylabel, bins,
                  range, ylim):
    plt.figure()
    plt.title(title)
    plt.hist(x=samples,
             weights=weights,
             bins=bins, density=True, range=range)
    # plt.axvline(mean, 0, 1,
    #             color="r", linestyle="--", label=r"$\mu = %.3f$" % (mean))
    # plt.axhline(.2,
    #             (mean-std-range[0])/(range[1] - range[0]),
    #             (mean+std-range[0])/(range[1] - range[0]),
    #             color="k",
    #             linestyle="--", label=r"$\sigma = %.3f$" % (std))
    x = np.linspace(-3, 3, 40)
    y = [marg(tmp_x) for tmp_x in x]
    plt.plot(x, y, "r--")
    plt.plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.show(block=False)


data = np.array([[-.5, .5]])
dim = data.shape[-1]
factor = create_factor(data)
Z = approximate_Z(factor)
gt_posterior = create_gt_posterior(factor, Z)

# gt_marg_1 = create_gt_marginal_1(factor, Z)
# gt_marg_2 = create_gt_marginal_2(factor, Z)
# gt_mean_th1 = compute_gt_mean(gt_marg_1)[0]
# gt_std_th1 = compute_gt_std(gt_marg_1, gt_mean_th1)[0]
# gt_mean_th2 = compute_gt_mean(gt_marg_2)[0]
# gt_std_th2 = compute_gt_std(gt_marg_2, gt_mean_th2)[0]
# plot_gt_posterior(gt_posterior, nof_points=50)
# plot_gt_marginals(gt_marg_1, gt_marg_2, nof_points=100)


def simulate_data(th1, th2, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    return likelihood.rvs(th1, th2, seed=random_state)


def summarize(x):
    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    return np.prod(x, axis=-1)


elfi.new_model("2D_example")
elfi_th1 = elfi.Prior(Prior(), name="th1")
elfi_th2 = elfi.Prior(Prior(), name="th2")
elfi_simulator = elfi.Simulator(
    simulate_data, elfi_th1, elfi_th2, observed=data, name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")
summary = elfi.Summary(summarize, dist, name="summary")


bounds = [(-2.5, 2.5), (-2.5, 2.5)]


# LINEAR
seed = 21
eps = 1
n1 = np.linspace(1, 501, 5)
solve_grad = []
estimate_regions = []
sample = []
eval_post = []
for i, n in enumerate(n1):
    romc = elfi.ROMC(summary, bounds, parallelize=False)

    tic = timeit.default_timer()
    romc.solve_problems(n1=int(n), seed=seed,
                        use_bo=False)
    toc = timeit.default_timer()
    solve_grad.append(toc-tic)

    tic = timeit.default_timer()
    romc.estimate_regions(eps_filter=eps, use_surrogate=False)
    toc = timeit.default_timer()
    estimate_regions.append(toc-tic)

    tic = timeit.default_timer()
    romc.sample(n2=50)
    toc = timeit.default_timer()
    sample.append(toc-tic)

    tic = timeit.default_timer()
    romc.eval_unnorm_posterior(np.zeros((50, 2)))
    toc = timeit.default_timer()
    eval_post.append(toc-tic)

# PARALLEL
estimate_regions_parallel = []
solve_grad_parallel = []
sample_parallel = []
eval_post_parallel = []
for i, n in enumerate(n1):
    time.sleep(2)
    romc1 = elfi.ROMC(summary, bounds, parallelize=True)

    tic = timeit.default_timer()
    romc1.solve_problems(n1=int(n), seed=seed,
                         use_bo=False)
    toc = timeit.default_timer()
    solve_grad_parallel.append(toc-tic)

    tic = timeit.default_timer()
    romc1.estimate_regions(eps_filter=eps, use_surrogate=False)
    toc = timeit.default_timer()
    estimate_regions_parallel.append(toc-tic)

    tic = timeit.default_timer()
    romc1.sample(n2=50)
    toc = timeit.default_timer()
    sample_parallel.append(toc-tic)

    tic = timeit.default_timer()
    romc1.eval_unnorm_posterior(np.zeros((50, 2)))
    toc = timeit.default_timer()
    eval_post_parallel.append(toc-tic)


plt.figure()
plt.title("Optimisation problems: Sequential vs Parallel Execution")
plt.plot(n1, solve_grad, "bo--", label="linear")
plt.plot(n1, solve_grad_parallel, "go--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
if savefig:
    plt.savefig(os.path.join(prepath, "solve_problems_parallel"),
                bbox_inches="tight")
plt.show(block=False)


plt.figure()
plt.title("Acceptance Regions: Sequential vs Parallel Execution")
plt.plot(n1, estimate_regions, "bo--", label="linear")
plt.plot(n1, estimate_regions_parallel, "go--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
if savefig:
    plt.savefig(os.path.join(prepath, "estimate_regions_parallel"),
                bbox_inches="tight")
plt.show(block=False)


plt.figure()
plt.title("Sampling: Sequential vs Parallel execution")
plt.plot(n1, sample, "bo--", label="linear")
plt.plot(n1, sample_parallel, "go--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
if savefig:
    plt.savefig(os.path.join(prepath, "sample_parallel"),
                bbox_inches="tight")
plt.show(block=False)


plt.figure()
plt.title("Posterior Evaluation: Sequential vs Parallel execution")
plt.plot(n1, eval_post, "bo--", label="linear")
plt.plot(n1, eval_post_parallel, "go--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
if savefig:
    plt.savefig(os.path.join(prepath, "eval_post_parallel"),
                bbox_inches="tight")
plt.show(block=False)
