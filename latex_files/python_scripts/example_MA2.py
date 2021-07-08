import logging
import matplotlib.pyplot as plt
from elfi.methods.parameter_inference import ROMC
from elfi.examples import ma2
import elfi
import numpy as np
import scipy.stats
import matplotlib
import os
matplotlib.rcParams['text.usetex'] = True

logging.basicConfig(level=logging.INFO)
prepath = '/home/givasile/ORwDS/edinburgh-thesis/Thesis/tmp_images/chapter4/'


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
    plt.savefig(savepath, bbox_inches='tight')
    plt.show(block=False)


# Set seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

# plot prior samples

x = model.generate(1000)
plt.figure()
plt.title("Samples from the prior")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.plot(x["t1"], x["t2"], "bo")
plt.savefig(os.path.join(prepath, "mae2_prior_samples.png"),
            bbox_inches="tight")
plt.show(block=False)

####### ROMC with gradients ################
n1 = 1000
n2 = 20
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .01
vis_ind_1 = 12
vis_ind_2 = 1
vis_ind_3 = 2

romc = ROMC(model, bounds=bounds, discrepancy_name="d")
romc.solve_problems(n1=n1, seed=seed)
romc.distance_hist(savefig=os.path.join(
    prepath, "ma2_distance_hist.png"))

romc.estimate_regions(eps_filter=eps, fit_models=True)

tmp = romc.sample(n2=n2)
romc.visualize_region(vis_ind_1, savefig=os.path.join(
    prepath, "ma2_region_1.png"))

romc.visualize_region(vis_ind_2, savefig=os.path.join(
    prepath, "ma2_region_2.png"))

romc.visualize_region(vis_ind_3, savefig=os.path.join(
    prepath, "ma2_region_3.png"))

print(romc.result.summary())


name = "t1"
plot_marginal(romc.result.samples["t1"], romc.result.weights,
              romc.result.sample_means_array[0],
              np.sqrt(romc.result.samples_cov()[0, 0]),
              r"ROMC (gradient-based) - Histogram of the parameter $\theta_1$",
              r"$\theta_1$",
              r"density",
              60,
              (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc.png"))

name = "t2"
plot_marginal(romc.result.samples["t2"], romc.result.weights,
              romc.result.sample_means_array[1],
              np.sqrt(romc.result.samples_cov()[1, 1]),
              r"ROMC (gradient-based) - Histogram of the parameter $\theta_2$",
              r"$\theta_2$",
              r"density",
              60,
              (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc.png"))


####### ROMC with BO ###########
use_bo = True
romc1 = ROMC(model, bounds=bounds, discrepancy_name="d")
romc1.solve_problems(n1=n1, seed=seed, use_bo=use_bo)
romc1.distance_hist(savefig=os.path.join(
    prepath, "ma2_distance_hist_bo.png"))

romc1.estimate_regions(eps_filter=eps, use_surrogate=False, fit_models=True)

tmp = romc1.sample(n2=n2)
romc1.visualize_region(vis_ind_1, savefig=os.path.join(
    prepath, "ma2_region_1_bo.png"))
romc1.visualize_region(vis_ind_2, savefig=os.path.join(
    prepath, "ma2_region_2_bo.png"))
romc1.visualize_region(vis_ind_3, savefig=os.path.join(
    prepath, "ma2_region_3_bo.png"))

print(romc1.result.summary())


name = "t1"
plot_marginal(romc1.result.samples["t1"], romc1.result.weights,
              romc1.result.sample_means_array[0],
              np.sqrt(romc1.result.samples_cov()[0, 0]),
              r"ROMC (Bayesian Optimisation) - Histogram of the parameter $\theta_1$",
              r"$\theta_1$",
              r"density",
              60,
              (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_romc_bo.png"))

name = "t2"
plot_marginal(romc1.result.samples["t2"], romc1.result.weights,
              romc1.result.sample_means_array[1],
              np.sqrt(romc1.result.samples_cov()[1, 1]),
              r"ROMC (Bayesian Optimisation) - Histogram of the parameter $\theta_2$",
              r"$\theta_2$",
              r"density",
              60,
              (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_romc_bo.png"))


# rejection
N = 10000
rej = elfi.Rejection(model, discrepancy_name="d", batch_size=10000, seed=seed)
vis = dict(xlim=[-2, 2], ylim=[-1, 1])
result = rej.sample(N, threshold=.1, vis=vis)

name = "t1"
plot_marginal(result.samples["t1"], None,
              result.sample_means_array[0],
              np.std(result.samples_array[:, 0]),
              r"Rejection ABC - Histogram of the parameter $\theta_1$",
              r"$\theta_1$",
              r"density",
              60,
              (0.3, 1.2), (0, 3.5), os.path.join(prepath, "mae2_hist_t1_rejection.png"))

name = "t2"
plot_marginal(result.samples["t2"], None,
              result.sample_means_array[1],
              np.std(result.samples_array[:, 1]),
              r"Rejection ABC - Histogram of the parameter $\theta_2$",
              r"$\theta_2$",
              r"density",
              60,
              (-0.5, 1), (0, 3), os.path.join(prepath, "mae2_hist_t2_rejection.png"))


print(result.summary())


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
    plt.savefig(savefig, bbox_inches='tight')
    plt.show(block=False)


plot_romc_posterior('MAE2, ROMC Unnorm Posterior (gradient-based)',
                    romc.eval_unnorm_posterior,
                    nof_points=50,
                    savefig=os.path.join(prepath, "mae2_romc_posterior.png"))


plot_romc_posterior('MAE2, ROMC Unnorm Posterior (Bayesian optimisation)',
                    romc1.eval_unnorm_posterior,
                    nof_points=50,
                    savefig=os.path.join(prepath, "mae2_romc_posterior_bo.png"))
