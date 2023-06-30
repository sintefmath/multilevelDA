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
coarse_grid_args = initGridSpecs(L-1)

coarse_args = {
    "nx": coarse_grid_args["nx"],
    "ny": coarse_grid_args["ny"],
    "dx": coarse_grid_args["dx"],
    "dy": coarse_grid_args["dy"],
    "gpu_ctx": gpu_ctx,
    "gpu_stream": gpu_stream,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
    }

coarse_data_args = make_init_steady_state(coarse_args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)


# %%
# Book keeping
log.write("L = " + str(L) + "\n")

log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T  = 900\n")


# %%
Yvar = 1
Yloc = [1024,512]

log.write("Yvar = " + str(Yvar) + "\n")
log.write("Yloc = " + str(Yloc[0]) + ", " + str(Yloc[1])+"\n")

# %% 
def covsXY(Ne, Yvar, Yloc):
    init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)
    coarse_init_mekl = ModelErrorKL.ModelErrorKL(**coarse_args, **init_model_error_basis_args)

    sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)
    coarse_sim_mekl = ModelErrorKL.ModelErrorKL(**coarse_args, **sim_model_error_basis_args)

    ensemble = []
    coarse_ensemble = []

    for e in range(Ne):
        sim = make_sim(args, sample_args, init_fields=data_args)
        coarse_sim = make_sim(coarse_args, sample_args, init_fields=coarse_data_args)
        
        if init_model_error_basis_args is not None:
            init_mekl.perturbSim(sim)
            coarse_init_mekl.perturbSimSimilarAs(coarse_sim, modelError=init_mekl)

        if sim_model_error_basis_args is not None:
            sim.model_error = sim_mekl
            coarse_sim.model_error = coarse_sim_mekl

        sim.model_time_step = sim_model_error_timestep
        sim.model_time_step = sim_model_error_timestep

        ensemble.append(sim)
        coarse_ensemble.append(coarse_sim)

    for e in range(Ne):
        print(e)
        ensemble[e].dataAssimilationStep(900, otherSim=coarse_ensemble[e])


    state = SLdownload(ensemble)
    coarse_state = SLdownload(coarse_ensemble)

    covXY = np.mean((state - np.mean(state, axis=-1)[:,:,:,np.newaxis]) * (state[Yvar,Yloc[0],Yloc[1],:]-np.mean(state[Yvar,Yloc[0],Yloc[1],:])), axis=-1)
    coarse_covXY = np.mean((coarse_state - np.mean(coarse_state, axis=-1)[:,:,:,np.newaxis]) * (coarse_state[Yvar,int(Yloc[0]/2),int(Yloc[1]/2),:]-np.mean(state[Yvar,int(Yloc[0]/2),int(Yloc[1]/2),:])), axis=-1).repeat(2,1).repeat(2,2)

    return covXY, coarse_covXY



# %%
ref_Ne = 200
log.write("ref_Ne: " + str(ref_Ne) + "\n")

ref_covXY, ref_coarse_covXY = covsXY(ref_Ne, Yvar, Yloc)

# %%
imshow3var(ref_covXY, eta_vlim=1, huv_vlim=50)

imshow3var(ref_coarse_covXY, eta_vlim=1, huv_vlim=50)

# %%
ref_diff_covXY = ref_covXY - ref_coarse_covXY
imshow3(ref_diff_covXY, eta_vlim=0.05, huv_vlim=5)

# %%
np.abs(ref_diff_covXY).max()

# %%
#####################################
#####################################
#####################################
Nes = [5, 10, 15, 20, 25]

log.write("Nes: " + ", ".join([str(Ne) for Ne in Nes]) + "\n")

err_abs = np.zeros((len(Nes),3))
err_rel = np.zeros((len(Nes),3))

for n, Ne in enumerate(Nes):
    N_test = 5
    err_abs_n = np.zeros(3)
    err_rel_n = np.zeros(3)
    for i in range(N_test):
        covXY, coarse_covXY = covsXY(Ne, Yvar, Yloc)
        diff_covXY = covXY - coarse_covXY

        err_abs_n += 1/N_test * np.max(np.abs(diff_covXY-ref_diff_covXY), axis=(1,2))
        err_rel_n += 1/N_test * np.max(np.abs(diff_covXY-ref_diff_covXY), axis=(1,2))/np.max(np.abs(ref_covXY), axis=(1,2))

    err_abs[n] = err_abs_n
    err_rel[n] = err_rel_n

# %%
np.save(output_path+"/err_abs.npy", err_abs)
np.save(output_path+"/err_rel.npy", err_rel)
