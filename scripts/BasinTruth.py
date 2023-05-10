# %% [markdown]
# # Multi Level Statistics

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

output_path = "DataAssimilation/BasinSL/"+timestamp 
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
from utils.BasinSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
L = 10

# %% 
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }


# %% 
init_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 6,
    "basis_y_start": 2,
    "basis_y_end": 7,

    "kl_decay": 1.25,
    "kl_scaling": 0.05,
}

# %% 
sim_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 7,
    "basis_y_start": 2,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.0025,
}


# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Tda', type=float, default=6*3600)
parser.add_argument('--Tforecast', type=float, default=6*3600)
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=5*60) 

pargs = parser.parse_args()

T_da = pargs.Tda
T_forecast = pargs.Tforecast
init_model_error = bool(pargs.init_error)
sim_model_error = bool(pargs.sim_error)
sim_model_error_timestep = pargs.sim_error_timestep

# T_da = 6*3600
# T_forecast = 6*3600
# init_model_error = False
# sim_model_error = True
# sim_model_error_timestep = 60.0

# %%
# Book keeping
log.write("L = " + str(L) + "\n")

grid_args = initGridSpecs(L)
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

log.write("Init State\n")
log.write("Double Bump\n\n")

log.write("Init Perturbation\n")
if init_model_error:
    log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")
else: 
    init_model_error_basis_args = None
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
    sim_model_error_basis_args = None
    log.write("False\n\n")

log.close()

# %% 
def write2file(T, mode=""):
    print("Saving ", mode, " at time ", T)
    true_state = truth.download(interior_domain_only=True)
    np.save(output_path+"/truth_"+str(T)+".npy", np.array(true_state))
    

# %%
args = {
    "nx": grid_args["nx"],
    "ny": grid_args["ny"],
    "dx": grid_args["dx"],
    "dy": grid_args["dy"],
    "gpu_ctx": gpu_ctx,
    "gpu_stream": gpu_stream,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
    }

data_args = make_init_steady_state(args)

# %% 
# Truth
if init_model_error:
    init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)

if sim_model_error:
    sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)

sim_args = {
    "gpu_ctx" : args["gpu_ctx"],
    "nx" : args["nx"],
    "ny" : args["ny"],
    "dx" : args["dx"],
    "dy" : args["dy"],
    "f"  : sample_args["f"],
    "g"  : sample_args["g"],
    "r"  : 0,
    "dt" : 0,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
    "eta0" : data_args["eta"],
    "hu0"  : data_args["hu"],
    "hv0"  : data_args["hv"],
    "H"    : data_args["Hi"],
}

truth = CDKLM16.CDKLM16(**sim_args) 
if init_model_error:
    init_mekl.perturbSim(truth)
if sim_model_error:
    truth.model_error = sim_mekl
    truth.model_time_step = sim_model_error_timestep



# %% 
# DA period
write2file(int(truth.t), "")

while truth.t < T_da + T_forecast:
    # Forward step
    truth.dataAssimilationStep(truth.t+300)

    # DA step
    write2file(int(truth.t), "")


# %%
