# %% [markdown]
# # Multi Level Analysis

# %% [markdown]
# ### Classes and modules

# %%

#Import packages we need
import numpy as np
import sys, os
import time

#For plotting
import matplotlib
from matplotlib import pyplot as plt

import pycuda.driver as cuda

# %%
# import time
# print("Gonna sleep now!")
# time.sleep(3*3600)

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "PracticalCost/Basin/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")

log.write("Script " + str(os.path.basename(__file__))+ "\n\n")

import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

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
ls = [6, 7, 8, 9, 10]


# %% 
from utils.BasinParameters import *

# %% [markdown]
## Set-up statisitcs

# %% 
T_spinup = 3600
T_test = 6*3600

N_test = 25

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

grid_args = initGridSpecs(ls[-1])
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T_spinup = " + str(T_spinup) +"\n")
log.write("T_test = " + str(T_spinup) +"\n\n")

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


log.write("\nStatistics\n")
log.write("N = " +str(N_test)+"\n")
log.close()

# %% [markdown]
## PLAIN SIMULATIONS PER LEVEL

# %% 
costsPure = np.zeros((len(ls), N_test))

# %% 
for l_idx in range(len(ls)):
    print("Level ", l_idx)
    # Prepare level data
    grid_args = initGridSpecs(ls[l_idx])
    args= {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        }
    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) 

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
    
    for n in range(N_test):
        print(l_idx, n)

        sim = CDKLM16.CDKLM16(**sim_args)
        sim.setKLModelError(**sim_model_error_basis_args)
        sim.model_time_step = sim_model_error_timestep

        sim.dataAssimilationStep(T_spinup)

        gpu_ctx.synchronize()
        tic = time.time()

        sim.dataAssimilationStep(T_spinup + T_test)

        gpu_ctx.synchronize()
        toc = time.time()

        costsPure[l_idx, n] = toc-tic

        sim.cleanUp()

np.save(output_path+"/costsPure.npy", costsPure)

# %% [markdown]
## SIMULATIONS WITH PARTNERS PER LEVEL

# %% 
costsPartnered = np.zeros((len(ls), N_test))

# %% 
for l_idx in range(len(ls)):
    print("Level ", l_idx)
    # Prepare level data
    grid_args = initGridSpecs(ls[l_idx])
    args= {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        }
    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) 

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


    # Prepare coarse data
    coarse_grid_args = initGridSpecs(ls[l_idx]-1)
    coarse_args= {
        "nx": coarse_grid_args["nx"],
        "ny": coarse_grid_args["ny"],
        "dx": coarse_grid_args["dx"],
        "dy": coarse_grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        }
    coarse_data_args = make_init_steady_state(coarse_args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) 

    coarse_sim_args = {
        "gpu_ctx" : coarse_args["gpu_ctx"],
        "nx" : coarse_args["nx"],
        "ny" : coarse_args["ny"],
        "dx" : coarse_args["dx"],
        "dy" : args["dy"],
        "f"  : sample_args["f"],
        "g"  : sample_args["g"],
        "r"  : 0,
        "dt" : 0,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
        "eta0" : coarse_data_args["eta"],
        "hu0"  : coarse_data_args["hu"],
        "hv0"  : coarse_data_args["hv"],
        "H"    : coarse_data_args["Hi"],
    }

    
    for n in range(N_test):
        print(l_idx, n)

        sim = CDKLM16.CDKLM16(**sim_args)
        sim.setKLModelError(**sim_model_error_basis_args)
        sim.model_time_step = sim_model_error_timestep

        coarse_sim = CDKLM16.CDKLM16(**coarse_sim_args)
        coarse_sim.setKLModelErrorSimilarAs(sim)
        coarse_sim.model_time_step = sim_model_error_timestep

        sim.dataAssimilationStep(T_spinup, otherSim=coarse_sim)

        gpu_ctx.synchronize()
        tic = time.time()

        sim.dataAssimilationStep(T_spinup + T_test, otherSim=coarse_sim)

        gpu_ctx.synchronize()
        toc = time.time()

        costsPartnered[l_idx, n] = toc-tic

        sim.cleanUp()
        coarse_sim.cleanUp()

np.save(output_path+"/costsPartnered.npy", costsPartnered)

# %% [markdown]
## PLAIN ENSEMBLE SIMULATIONS PER LEVEL

# Stepping DA-steps 

# %% 
Ne = 50

log = open(output_path+"/log.txt", 'w')
log.write("\nNe = " + str(Ne) + "\n")
log.close()

# %% 
costsPureEnsemble = np.zeros((len(ls), N_test))

# %% 
for l_idx in range(len(ls)):
    print("Level ", l_idx)
    # Prepare level data
    grid_args = initGridSpecs(ls[l_idx])
    args= {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        }
    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) 

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
    
    for n in range(N_test):
        print(l_idx, n)

        sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)
        ensemble = []
        for e in range(Ne):
            sim = CDKLM16.CDKLM16(**sim_args)
            sim.model_error = sim_mekl
            sim.model_time_step = sim_model_error_timestep
            ensemble.append(sim)

        for e in range(Ne):
            ensemble[e].dataAssimilationStep(T_spinup)
        

        gpu_ctx.synchronize()
        tic = time.time()

        while ensemble[0].t < T_spinup + T_test:
            t_step = np.minimum(da_timestep, T_spinup + T_test - ensemble[0].t)
            for e in range(Ne):
                ensemble[e].dataAssimilationStep(ensemble[e].t + t_step)

        gpu_ctx.synchronize()
        toc = time.time()

        costsPureEnsemble[l_idx, n] = (toc-tic)/Ne

        sim_mekl.cleanUp()
        for e in range(Ne):
            ensemble[e].cleanUp()

np.save(output_path+"/costsPureEnsemble.npy", costsPureEnsemble)

# %% [markdown]
## ENSEMBLE SIMULATIONS WITH PARTNERS PER LEVEL

# %% 
costsPartneredEnsemble = np.zeros((len(ls), N_test))

# %% 
for l_idx in range(len(ls)):
    print("Level ", l_idx)
    # Prepare level data
    grid_args = initGridSpecs(ls[l_idx])
    args= {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        }
    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) 

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
    

    # Prepare coarse data
    coarse_grid_args = initGridSpecs(ls[l_idx]-1)
    coarse_args= {
        "nx": coarse_grid_args["nx"],
        "ny": coarse_grid_args["ny"],
        "dx": coarse_grid_args["dx"],
        "dy": coarse_grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        }
    coarse_data_args = make_init_steady_state(coarse_args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) 

    coarse_sim_args = {
        "gpu_ctx" : coarse_args["gpu_ctx"],
        "nx" : coarse_args["nx"],
        "ny" : coarse_args["ny"],
        "dx" : coarse_args["dx"],
        "dy" : args["dy"],
        "f"  : sample_args["f"],
        "g"  : sample_args["g"],
        "r"  : 0,
        "dt" : 0,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
        "eta0" : coarse_data_args["eta"],
        "hu0"  : coarse_data_args["hu"],
        "hv0"  : coarse_data_args["hv"],
        "H"    : coarse_data_args["Hi"],
    }


    for n in range(N_test):
        print(l_idx, n)

        sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)
        coarse_sim_mekl = ModelErrorKL.ModelErrorKL(**coarse_args, **sim_model_error_basis_args)

        ensemble = []
        coarse_ensemble = []
        for e in range(Ne):
            sim = CDKLM16.CDKLM16(**sim_args)
            sim.model_error = sim_mekl
            sim.model_time_step = sim_model_error_timestep

            coarse_sim = CDKLM16.CDKLM16(**coarse_sim_args)
            coarse_sim.model_error = coarse_sim_mekl
            coarse_sim.model_time_step = sim_model_error_timestep

            ensemble.append(sim)
            coarse_ensemble.append(coarse_sim)

        for e in range(Ne):
            ensemble[e].dataAssimilationStep(T_spinup, otherSim=coarse_ensemble[e])

        gpu_ctx.synchronize()
        tic = time.time()

        while ensemble[0].t < T_spinup + T_test:
            t_step = np.minimum(da_timestep, T_spinup + T_test - ensemble[0].t)
            for e in range(Ne):
                ensemble[e].dataAssimilationStep(ensemble[e].t + t_step, otherSim=coarse_ensemble[e])

        gpu_ctx.synchronize()
        toc = time.time()

        costsPartneredEnsemble[l_idx, n] = (toc-tic)/Ne

        sim_mekl.cleanUp()
        for e in range(Ne):
            ensemble[e].cleanUp()
            coarse_ensemble[e].cleanUp()

np.save(output_path+"/costsPartneredEnsemble.npy", costsPartneredEnsemble)

