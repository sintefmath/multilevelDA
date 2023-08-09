## PARAMETERS 
# Collecting all relevant parameter for the experiments
# in this separate file

import numpy as np

#TODO: Allow for different parameter sets

start_basis = 1
num_bases = 7
sim_model_error_basis_args = {
                            "basis_x_start": start_basis+1, 
                            "basis_x_end": start_basis+1+num_bases,
                            "basis_y_start": start_basis,
                            "basis_y_end": start_basis+num_bases,

                            "kl_decay": 0.9,
                            "kl_scaling": 0.001,
                            }

sim_model_error_timestep=60.0

# Parameters for time
T_spinup = 3*24*3600
T_da = 7*24*3600
T_forecast = 3*24*3600

# Parameters for truth obervation
xdim = 1332000.0
ydim = 666000.0

xs = np.linspace(0, xdim, 10, endpoint=False)
ys = np.linspace(0, ydim, 5, endpoint=False)

[obs_xs, obs_ys] = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2).T

R = [0.05, 5.0, 5.0]

# Parameters for assimilation
obs_var = slice(1,3)
r = 5e4
relax_factor = 0.5
min_location_level = 0

da_timestep = 900

# Drifters
