# %% [markdown]
# # Full simulation in Basin

# %% [markdown]
# ### Classes and modules

# %%
#Lets have matplotlib "inline"
import os
import sys

#Import packages we need
import numpy as np
import datetime
from IPython.display import display
import copy

#For plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"

import pycuda.driver as cuda

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import  Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %%
gpu_ctx = Common.CUDAContext()

# %%
gpu_stream = cuda.Stream()

# %% [markdown]
# Utils

# %%
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
from utils.BasinPlot import * 

# %% [markdown]
# ### Collecting Perturbations

# %%
from utils.BasinParameters import *

# %%
L = 9

# %%
grid_args = initGridSpecs(L)
args =  {
    "nx": grid_args["nx"],
    "ny": grid_args["ny"],
    "dx": grid_args["dx"],
    "dy": grid_args["dy"],
    "gpu_ctx": gpu_ctx,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
    }

# %%
init_mekl =  ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args, gpu_stream=gpu_stream)


# %%
data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

# %% [markdown]
# ### Monte Carlo Drifters

# %%
from gpuocean.dataassimilation import DataAssimilationUtils as dautils
observation_args = {'observation_type': dautils.ObservationType.UnderlyingFlow,
                'nx': grid_args["nx"], 'ny': grid_args["ny"],
                'domain_size_x': grid_args["nx"]*grid_args["dx"],
                'domain_size_y': grid_args["ny"]*grid_args["dy"],
               }


# %%
init_positions = np.array([[500*80, 1000*80],  #[[x1, y1],
                           [550*80, 1000*80],  # [x2, y2],
                           [450*80, 1000*80],
                           [500*80, 1050*80],
                           [550*80, 1050*80],
                           [450*80, 1050*80],
                           [500*80, 1100*80],
                           [550*80, 1100*80],
                           [450*80, 1100*80]
                           ]) 

# %%
from gpuocean.utils import Observation

# %%
num_drifters = len(init_positions)
from gpuocean.drifters import GPUDrifterCollection


# %% 
# scales = np.array([[ 0.25, 0.0], [ 0.1, 0.0], [ 0.05, 0.0], 
#                    [ 0.0, 0.01],  [ 0.0, 0.005],  [ 0.0, 0.0025]])

# for scale in scales:
#     print(scale)
#     init_model_error_basis_args["kl_scaling"] = scale[0]
#     sim_model_error_basis_args["kl_scaling"] = scale[1]

# Bookkeeping
forecasts = [] 
init_states = np.zeros((10, 3, grid_args["ny"], grid_args["nx"]))
final_states = np.zeros((10, 3, grid_args["ny"], grid_args["nx"]))

for n in range(10):
    print("Experiment ", n)
    forecast = Observation.Observation(**observation_args)
    drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                                boundaryConditions = args["boundary_conditions"],
                                                domain_size_x = forecast.domain_size_x,
                                                domain_size_y = forecast.domain_size_y)

    # Set up sim
    sim = make_sim(args, sample_args, data_args)
    init_mekl.perturbSim(sim)
    
    init_states[n] = np.array(sim.download(interior_domain_only=True))
    
    sim.setKLModelError(**sim_model_error_basis_args)
    sim.model_time_step = sim_model_error_timestep

    # DA period
    sim.dataAssimilationStep(6*3600)

    # Forecast period with drifters
    drifters.setDrifterPositions(init_positions)
    sim.attachDrifters(drifters)
    forecast.add_observation_from_sim(sim)

    for hours in range(6):
        for obses in range(4):
            sim.dataAssimilationStep(sim.t + 900)
            forecast.add_observation_from_sim(sim)

    forecasts.append(forecast)

    final_states[n] = np.array(sim.download(interior_domain_only=True))

# Save MC results
init_states = np.moveaxis(init_states, 0, -1)
final_states = np.moveaxis(final_states, 0, -1)
# np.save("MC/MCinitStates_init_"+"{:.4f}".format(scale[0])[2:]+"_sim_"+"{:.4f}".format(scale[1])[2:]+".npy", init_states)
np.save("MC/MCinitStates.npy", init_states)
# np.save("MC/MCfinalStates_init_"+"{:.4f}".format(scale[0])[2:]+"_sim_"+"{:.4f}".format(scale[1])[2:]+".npy", final_states)
np.save("MC/MCfinalStates.npy", final_states)

fig, ax = plt.subplots(1,1, figsize=(10,10))
domain_extent = [0, sim.nx*sim.dx/1000, 0, sim.ny*sim.dy/1000]

ax.imshow(np.zeros((grid_args["ny"], grid_args["nx"])), interpolation="none", origin='lower', 
            cmap=plt.cm.Oranges, extent=domain_extent, zorder=-10)

for forecast in forecasts:
    for drifter_id in range(len(init_positions)): 
        path = forecast.get_drifter_path(drifter_id, 0,  sim.t, in_km = True)[0]
        ax.plot(path[:,0], path[:,1], color="C"+str(drifter_id), ls="-", zorder=-3)

# plt.savefig("MC/MCdrift_init_"+"{:.4f}".format(scale[0])[2:]+"_sim_"+"{:.4f}".format(scale[1])[2:]+".pdf", bbox_inches="tight")
plt.savefig("MC/MCdrift.pdf", bbox_inches="tight")
