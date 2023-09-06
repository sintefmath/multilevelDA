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

output_path = "MC/SL/"+timestamp
os.makedirs(output_path, exist_ok=True)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")


# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *
from utils.BasinSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
L = 9

# %% 
from utils.BasinParameters import *

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Ne', type=int, default=50)

pargs = parser.parse_args()

Ne = pargs.Ne

# %%
# Book keeping
log.write("L = " + str(L) + "\n")
log.write("Ne = " + str(Ne) + "\n\n")

grid_args = initGridSpecs(L)
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

log.write("Init State\n")
log.write("Double Bump\n")
log.write("Bump size [m]: " + str(steady_state_bump_a) +"\n")
log.write("Bump dist [fractal]: " + str(steady_state_bump_fractal_dist) + "\n\n")


log.write("Init Perturbation\n")
log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")


log.write("Temporal Perturbation\n")
log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")

log.close()


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

data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

def makePlots():
    # mean
    SL_mean = SLestimate(SL_ensemble, np.mean)
    fig, axs = imshow3(SL_mean, eta_vlim=steady_state_bump_a, huv_vlim=100)
    plt.savefig(output_path+"/MCmean_"+str(int(SL_ensemble[0].t))+".pdf")

    # var
    SL_var = SLestimate(SL_ensemble, np.var)
    fig, axs = imshow3var(SL_var, eta_vlim=0.025, huv_vlim=100)
    plt.savefig(output_path+"/MCvar_"+str(int(SL_ensemble[0].t))+".pdf")

    plt.close('all')


# %%
# Ensemble
SL_ensemble = initSLensemble(Ne, args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)

# %% 
# DA period
makePlots()

while SL_ensemble[0].t < T_da:
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + 3600)
    print("Plotting at ", SL_ensemble[0].t)
    makePlots()


# %% 
# Prepare drifters
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.utils import Observation
from gpuocean.dataassimilation import DataAssimilationUtils as dautils
observation_args = {'observation_type': dautils.ObservationType.UnderlyingFlow,
                'nx': grid_args["nx"], 'ny': grid_args["ny"],
                'domain_size_x': grid_args["nx"]*grid_args["dx"],
                'domain_size_y': grid_args["ny"]*grid_args["dy"],
               }

num_drifters = len(init_positions)

forecasts = []
for e in range(len(SL_ensemble)):
    forecast = Observation.Observation(**observation_args)
    drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                            boundaryConditions = args["boundary_conditions"],
                                            domain_size_x = forecast.domain_size_x,
                                            domain_size_y = forecast.domain_size_y)
    drifters.setDrifterPositions(init_positions)
    SL_ensemble[e].attachDrifters(drifters)
    forecast.add_observation_from_sim(SL_ensemble[0])
    forecasts.append(forecast)


# %%
# Forecast period
while SL_ensemble[0].t < T_da + T_forecast:
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + 300.0)
    for e in range(len(SL_ensemble)):
        forecasts[e].add_observation_from_sim(SL_ensemble[e])
    if SL_ensemble[0].t % 3600 < 0.1:
        print("Plotting at ", SL_ensemble[0].t)
        makePlots()


drifter_folder = os.path.join(output_path, 'sldrifters')
os.makedirs(drifter_folder)
for e in range(len(SL_ensemble)):
    forecasts[e].to_pickle(os.path.join(drifter_folder,"sldrifters_"+str(e).zfill(4)+".pickle"))



