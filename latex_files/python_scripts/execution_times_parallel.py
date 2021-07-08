"""Testing the execution times using parallelisation."""

import timeit
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.ndimage as ndimage
import scipy.stats as ss
import os
import elfi
import matplotlib
matplotlib.rcParams['text.usetex'] = True

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
n1 = np.linspace(1, 501, 10)
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


prepath = '/home/givasile/ORwDS/edinburgh-thesis/Thesis/tmp_images/chapter4'


plt.figure()
plt.title("Solve optimisation problems: sequential vs parallel")
plt.plot(n1, solve_grad, "bo--", label="linear")
plt.plot(n1, solve_grad_parallel, "ro--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
plt.savefig(os.path.join(prepath, "solve_problems_parallel"),
            bbox_inches="tight")
plt.show(block=False)


plt.figure()
plt.title("Construct bounding boxes: sequential vs parallel")
plt.plot(n1, estimate_regions, "bo--", label="linear")
plt.plot(n1, estimate_regions_parallel, "ro--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
plt.savefig(os.path.join(prepath, "estimate_regions_parallel"),
            bbox_inches="tight")
plt.show(block=False)


plt.figure()
plt.title("Sampling: sequential vs parallel")
plt.plot(n1, sample, "bo--", label="linear")
plt.plot(n1, sample_parallel, "ro--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
plt.savefig(os.path.join(prepath, "sample_parallel"),
            bbox_inches="tight")
plt.show(block=False)


plt.figure()
plt.title("Evaluate the approximate posterior: sequential vs parallel")
plt.plot(n1, eval_post, "bo--", label="linear")
plt.plot(n1, eval_post_parallel, "ro--", label="parallel")
plt.xlabel(r"$n_1$")
plt.ylabel("time (sec)")
plt.legend()
plt.savefig(os.path.join(prepath, "eval_post_parallel"),
            bbox_inches="tight")
plt.show(block=False)
