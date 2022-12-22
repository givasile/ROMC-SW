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

# ROMC
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


# BOLFI
bolfi = elfi.BOLFI(model, target_name = "d",
                   batch_size=1,
                   initial_evidence=20,
                   update_interval=10,
                   bounds={'t1':(-2, 2), 't2':(-1.25, 1.25)},
                   acq_noise_var=0.1,
                   seed=seed)
