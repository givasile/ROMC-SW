import logging
import matplotlib.pyplot as plt
from elfi.examples import ma2
import elfi
import numpy as np
import scipy.stats
import matplotlib
import os
import tikzplotlib as tplt
matplotlib.rcParams['text.usetex'] = True

logging.basicConfig(level=logging.INFO)
prepath = './../images/chapter4/'
savefig = False

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
        # tplt.clean_figure()
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

# from elfi.methods.parameter_inference import ROMC
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
