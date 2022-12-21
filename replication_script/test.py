import numpy as np
import elfi

# Set seed for reproducibility
seed = 1
np.random.seed(seed)
model = ma2.get_model(seed_obs=seed)

# Hyperparameters
n1 = 5
n2 = 200
bounds = [(-2, 2), (-1.25, 1.25)]
eps = .01

# define model
romc = elfi.ROMC(model, bounds=bounds, parallelize=False, discrepancy_name="d")

# fitting part
romc.solve_problems(n1=n1, use_bo=True, seed=seed)
romc.estimate_regions(eps_filter=eps, fit_models=False)

# sampling part
tmp = romc.sample(n2=n2, seed=seed)
romc.visualize_region(vis_ind_1)

# print summary statistics
print(romc.result.summary())
print(romc.result.samples_cov())
