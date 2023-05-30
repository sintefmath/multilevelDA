## PARAMETERS 
# Collecting all relevant parameter for the experiments
# in this separate file

#TODO: Allow for different parameter sets

# Parameters for the simulation initialisation
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }

steady_state_bump_a = 3
steady_state_bump_fractal_dist = 7

# Parameters for perturbations
init_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 6,
    "basis_y_start": 2,
    "basis_y_end": 7,

    "kl_decay": 1.25,
    "kl_scaling": 0.18,
}

sim_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 7,
    "basis_y_start": 2,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.004,
}

sim_model_error_timestep=60.0

# Parameters for time
T_da = 6*3600
T_forecast = 6*3600

# Parameters for truth obervation
obs_xs = [40040.0]
obs_ys = [80040.0]

R = [0.05, 1.0, 1.0]

# Parameters for assimilation
obs_var = slice(1,3)
r = 2.5e4
relax_factor = 0.1
min_location_level = 0

da_timestep = 900

