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

output_path = "Testing/Basin/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(os.path.realpath(os.path.dirname(__file__)), search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")


# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *
from utils.BasinSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
L = 10

# %% 
from utils.BasinParameters import *

# %%
grid_args = initGridSpecs(L)
args = {
    "nx": grid_args["nx"],
    "ny": grid_args["ny"],
    "dx": grid_args["dx"],
    "dy": grid_args["dy"],
    "gpu_ctx": gpu_ctx,
    "gpu_stream": gpu_stream,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
    }

data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

# %% 
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n\n")

# %%

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


sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)

# %%
import nvidia_smi
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

log.write("GPU " + str(nvidia_smi.nvmlDeviceGetName(gpu_handle)) + "on " + str(os.uname()[1]) + "\n\n")

SL_ensemble = []
for e in range(10000):
    try:
        sim = CDKLM16.CDKLM16(**sim_args) 
        sim.model_error = sim_mekl
        sim.model_time_step = sim_model_error_timestep
        SL_ensemble.append( sim )

        gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
        log.write("Sim " + str(e) + ": " + "{:.2f}".format(100*gpu_info.used/gpu_info.total) + " GPU mem used\n")
    except:
        nvidia_smi.nvmlShutdown()
        log.write("Failed to init sim " + str(e) + "\n")
        log.close()
        sys.exit(0)

nvidia_smi.nvmlShutdown()
