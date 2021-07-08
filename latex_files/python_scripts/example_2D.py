import logging
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from elfi.methods.parameter_inference import ROMC
from elfi.examples import ma2
import elfi
import numpy as np
import scipy.stats as ss
import matplotlib
import os
matplotlib.rcParams['text.usetex'] = True

logging.basicConfig(level=logging.INFO)
prepath = '/home/givasile/ORwDS/edinburgh-thesis/Thesis/tmp_images/chapter4/'


def plot_marginal(samples, weights, mean, std, marg, title, xlabel, ylabel, bins,
                  range, ylim, savepath):
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
    plt.plot(x, y, "r--", label=r"$p(\theta_1|y_0)$")
    plt.plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show(block=False)


# Set seed for reproducibility
seed = 21
np.random.seed(seed)


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
    plt.title('Ground-truth Posterior')
    plt.colorbar()
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.savefig(os.path.join(prepath, "ex2D_gt_posterior.png"),
                bbox_inches="tight")
    plt.show(block=False)


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
    plt.savefig(os.path.join(prepath, "ex2D_gt_marginals.png"),
                bbox_inches="tight")
    plt.show(block=False)


def compute_gt_mean(gt_marginal):
    def h(x):
        return gt_marginal(x)*x
    return integrate.quad(h, -2.5, 2.5)


def compute_gt_std(gt_marginal, m):
    def h(x):
        return gt_marginal(x)*((x-m)**2)
    return np.sqrt(integrate.quad(h, -2.5, 2.5))


data = np.array([[-.5, .5]])
dim = data.shape[-1]
factor = create_factor(data)
Z = approximate_Z(factor)
gt_posterior = create_gt_posterior(factor, Z)
gt_marg_1 = create_gt_marginal_1(factor, Z)
gt_marg_2 = create_gt_marginal_2(factor, Z)
gt_mean_th1 = compute_gt_mean(gt_marg_1)[0]
gt_std_th1 = compute_gt_std(gt_marg_1, gt_mean_th1)[0]
gt_mean_th2 = compute_gt_mean(gt_marg_2)[0]
gt_std_th2 = compute_gt_std(gt_marg_2, gt_mean_th2)[0]
plot_gt_posterior(gt_posterior, nof_points=50)
plot_gt_marginals(gt_marg_1, gt_marg_2, nof_points=100)


def simulate_data(th1, th2, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    return likelihood.rvs(th1, th2, seed=random_state)


def summarize(x):
    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    return np.prod(x, axis=-1)


elfi.new_model("2D_example")
elfi_th1 = elfi.Prior(Prior(), name="t1")
elfi_th2 = elfi.Prior(Prior(), name="t2")
elfi_simulator = elfi.Simulator(simulate_data, elfi_th1, elfi_th2,
                                observed=data, name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")
summary = elfi.Summary(summarize, dist, name="d")
model = summary


####### ROMC with gradients ################
n1 = 500
n2 = 30
bounds = [(-2.5, 2.5), (-2.5, 2.5)]
eps = .75
vis_ind_1 = 1

romc = ROMC(model, bounds=bounds, discrepancy_name="d")
romc.solve_problems(n1=n1, seed=seed)
romc.distance_hist(savefig=os.path.join(
    prepath, "ex2D_distance_hist.png"))

romc.estimate_regions(eps_filter=eps, fit_models=True)

tmp = romc.sample(n2=n2)
romc.visualize_region(vis_ind_1, savefig=os.path.join(
    prepath, "ex2D_region_1.png"))

print(romc.result.summary())


name = "t1"
plot_marginal(romc.result.samples["t1"], romc.result.weights,
              romc.result.sample_means_array[0],
              np.sqrt(romc.result.samples_cov()[0, 0]),
              gt_marg_1,
              r"ROMC (gradient-based) - Histogram of the parameter $\theta_1$",
              r"$\theta_1$",
              r"density",
              50,
              bounds[0], (0, .6), os.path.join(prepath, "ex2D_hist_t1_romc.png"))

name = "t2"
plot_marginal(romc.result.samples["t2"], romc.result.weights,
              romc.result.sample_means_array[1],
              np.sqrt(romc.result.samples_cov()[1, 1]),
              gt_marg_2,
              r"ROMC (gradient-based) - Histogram of the parameter $\theta_2$",
              r"$\theta_2$",
              r"density",
              50,
              bounds[1], (0, .6), os.path.join(prepath, "ex2D_hist_t2_romc.png"))


####### ROMC with BO ###########
use_bo = True
romc1 = ROMC(model, bounds=bounds, discrepancy_name="d")
romc1.solve_problems(n1=n1, seed=seed, use_bo=use_bo)
romc1.distance_hist(savefig=os.path.join(
    prepath, "ex2D_distance_hist_bo.png"))

romc1.estimate_regions(eps_filter=eps, use_surrogate=False, fit_models=True)

tmp = romc1.sample(n2=n2)
romc1.visualize_region(vis_ind_1, savefig=os.path.join(
    prepath, "ex2D_region_1_bo.png"))

print(romc1.result.summary())


name = "t1"
plot_marginal(romc1.result.samples["t1"], romc1.result.weights,
              romc1.result.sample_means_array[0],
              np.sqrt(romc1.result.samples_cov()[0, 0]),
              gt_marg_1,
              r"ROMC (Bayesian Optimisation) - Histogram of the parameter $\theta_1$",
              r"$\theta_1$",
              r"density",
              50,
              bounds[0], (0, .6), os.path.join(prepath, "ex2D_hist_t1_romc_bo.png"))

name = "t2"
plot_marginal(romc1.result.samples["t2"], romc1.result.weights,
              romc1.result.sample_means_array[1],
              np.sqrt(romc1.result.samples_cov()[1, 1]),
              gt_marg_2,
              r"ROMC (Bayesian Optimisation) - Histogram of the parameter $\theta_2$",
              r"$\theta_2$",
              r"density",
              50,
              bounds[1], (0, .6), os.path.join(prepath, "ex2D_hist_t2_romc_bo.png"))


def plot_romc_posterior(title, posterior, nof_points, savefig):
    plt.figure()
    th1 = np.linspace(-4, 4, nof_points)
    th2 = np.linspace(-4, 4, nof_points)
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
    plt.savefig(savefig, bbox_inches='tight')
    plt.show(block=False)


romc.posterior.partition = None
romc.posterior.eps_cutoff = .75
plot_romc_posterior('ROMC Posterior (gradient-based)',
                    romc.eval_posterior,
                    nof_points=40,
                    savefig=os.path.join(prepath, "ex2D_romc_posterior.png"))


plot_romc_posterior('ROMC Posterior (Bayesian optimisation)',
                    romc1.eval_posterior,
                    nof_points=20,
                    savefig=os.path.join(prepath, "ex2D_romc_posterior_bo.png"))


def wrapper(x):
    res = []
    for i in range(x.shape[0]):
        res.append(gt_posterior(x[i][0], x[i][1]))
    return np.array(res)


romc.compute_divergence(wrapper, step=.2)
