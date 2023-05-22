# %% [markdown]
# # Multi Level Analysis

# %% [markdown]
# ### Classes and modules

# %%

#Import packages we need
import numpy as np
import sys, os

#For plotting
import matplotlib
from matplotlib import pyplot as plt

import pycuda.driver as cuda

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "VarianceLevels/Basin/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %%
ls = [6, 7, 8, 9, 10]

# %% 
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }


# %%
model_error_args_list = []

for l in ls:
    lvl_grid_args = initGridSpecs(l)
    model_error_args_list.append( {
        "nx": lvl_grid_args["nx"],
        "ny": lvl_grid_args["ny"],
        "dx": lvl_grid_args["dx"],
        "dy": lvl_grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        } )


# %% 
init_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 6,
    "basis_y_start": 2,
    "basis_y_end": 7,

    "kl_decay": 1.25,
    "kl_scaling": 0.18,
}

# %% 
init_mekls = []
for l_idx in range(len(ls)): 
    init_mekls.append( ModelErrorKL.ModelErrorKL(**model_error_args_list[l_idx], **init_model_error_basis_args) 
                )

# %% 
sim_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 7,
    "basis_y_start": 2,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.004,
}

# %% [markdown]
## Set-up statisitcs

# %% 
from utils.VarianceStatistics import * 

# %% 
welford_vars = []
welford_diff_vars = []
for l_idx, l in enumerate(ls):
    welford_vars.append( WelfordsVariance3((init_mekls[l_idx].ny, init_mekls[l_idx].nx)) )
    if l_idx > 0:
        welford_diff_vars.append( WelfordsVariance3((init_mekls[l_idx].ny, init_mekls[l_idx].nx)) )

# %% 
Ts = [0, 15*60, 3600, 6*3600, 12*3600]

# %% 
welford_vars_Ts = []
welford_diff_vars_Ts = [] 
for T in Ts:
    welford_vars_Ts.append( copy.deepcopy(welford_vars))
    welford_diff_vars_Ts.append( copy.deepcopy(welford_diff_vars))

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--initSteadyState', type=int, default=1, choices=[0,1])
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=60.0) 


args = parser.parse_args()

init_steady_state = bool(args.initSteadyState)
if init_steady_state:
    make_data_args = make_init_steady_state
else:
    make_data_args = make_init_fields

init_model_error = bool(args.init_error)
sim_model_error = bool(args.sim_error)
sim_model_error_timestep = args.sim_error_timestep

N_var = args.N

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

data_args = initGridSpecs(ls[-1])
log.write("nx = " + str(data_args["nx"]) + ", ny = " + str(data_args["ny"])+"\n")
log.write("dx = " + str(data_args["dx"]) + ", dy = " + str(data_args["dy"])+"\n")
log.write("T = " + ", ".join([str(T) for T in Ts]) +"\n\n")

log.write("Init State\n")
if init_steady_state: 
    log.write("Double-bump steady state\n\n")
else:
    log.write("Lake-at-rest\n\n")

log.write("Init Perturbation\n")
if init_model_error:
    log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")
else: 
    log.write("False\n\n")

log.write("Temporal Perturbation\n")
if sim_model_error:
    log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
    log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")
else:
    log.write("False\n\n")

log.write("Statistics\n")
log.write("N = " + str(N_var) + "\n")

log.close()

# %%
for l_idx in range(len(ls)):  # loop over levels
    ## Init fields
    data_args = make_data_args(model_error_args_list[l_idx])
    if l_idx > 0:
        coarse_data_args = make_data_args(model_error_args_list[l_idx-1])
    
    for i in range(N_var): # loop over samples
        print(l_idx, i)
        ## INIT SIM
        sim = make_sim(model_error_args_list[l_idx], sample_args=sample_args, init_fields=data_args)
        if init_model_error:
            init_mekls[l_idx].perturbSim(sim)
        if sim_model_error:
            sim.model_time_step = sim_model_error_timestep
            sim.setKLModelError(**sim_model_error_basis_args)

        coarse_sim = None
        if l_idx > 0:
            coarse_sim = make_sim(model_error_args_list[l_idx-1], sample_args=sample_args, init_fields=coarse_data_args)
            if init_model_error:
                init_mekls[l_idx-1].perturbSimSimilarAs(coarse_sim, modelError=init_mekls[l_idx])
            if sim_model_error:
                coarse_sim.model_time_step = sim_model_error_timestep
                coarse_sim.setKLModelError(**sim_model_error_basis_args)

        ## EVOLVE SIMS
        for t_idx, T in enumerate(Ts): 
            if T > 0: 
                sim.dataAssimilationStep(T, otherSim=coarse_sim)

            eta, hu, hv = sim.download(interior_domain_only=True)
            welford_vars_Ts[t_idx][l_idx].update(eta, hu, hv)

            if l_idx > 0:
                coarse_eta, coarse_hu, coarse_hv = coarse_sim.download(interior_domain_only=True)
                welford_diff_vars_Ts[t_idx][l_idx-1].update(eta - coarse_eta.repeat(2,0).repeat(2,1), 
                                                hu - coarse_hu.repeat(2,0).repeat(2,1), 
                                                hv - coarse_hv.repeat(2,0).repeat(2,1))

# %% 
for t_idx, T in enumerate(Ts):
    vars = np.array([np.sqrt(np.average(np.array(wv.finalize())**2, axis=(1,2))) for wv in welford_vars_Ts[t_idx]])
    diff_vars = np.array([np.sqrt(np.average(np.array(wv.finalize())**2, axis=(1,2))) for wv in welford_diff_vars_Ts[t_idx]])
    
    np.save(output_path+"/vars_"+str(T), vars)
    np.save(output_path+"/diff_vars_"+str(T), diff_vars)
