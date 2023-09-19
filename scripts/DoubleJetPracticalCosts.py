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

output_path = "PracticalCost/"+timestamp 
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
from gpuocean.SWEsimulators import CDKLM16


# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %%
ls = [6, 7, 8, 9]


# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.DoubleJetParametersReplication import * 

# %%
from gpuocean.utils import DoubleJetCase

args_list = []
init_list = []

for l in ls:
    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=2**l, nx=2**(l+1))
    doubleJetCase_args, doubleJetCase_init, _ = doubleJetCase.getInitConditions()

    args_list.append(doubleJetCase_args)
    init_list.append(doubleJetCase_init)


# %% [markdown]
## Set-up statisitcs

# %% 
T_spinup = 12*3600
T_test = 24*3600

N_test = 25

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
log.write("T_spinup = " + str(T_spinup) +"\n")
log.write("T_test = " + str(T_spinup) +"\n\n")

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

    for n in range(N_test):
        print(l_idx, n)

        sim = CDKLM16.CDKLM16(**args_list[l_idx], **init_list[l_idx])
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
for l_idx in range(1,len(ls)):
    print("Level ", l_idx)

    for n in range(N_test):
        print(l_idx, n)

        sim = CDKLM16.CDKLM16(**args_list[l_idx], **init_list[l_idx])
        sim.setKLModelError(**sim_model_error_basis_args)
        sim.model_time_step = sim_model_error_timestep

        coarse_sim = CDKLM16.CDKLM16(**args_list[l_idx-1], **init_list[l_idx-1])
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

