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

from skimage.measure import block_reduce

import pycuda.driver as cuda

# %%
# import time
# print("Gonna sleep now!")
# time.sleep(3*3600)

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),"VarianceLevelsMC/"+timestamp)
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(os.path.realpath(os.path.dirname(__file__)), search_parent_directories=True)
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../')))
from utils.DoubleJetSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %%
ls = [6, 7, 8]

# %% 
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
Ts = [3*24*3600, 4*24*3600, 5*24*3600, 6*24*3600, 7*24*3600, 8*24*3600, 9*24*3600, 10*24*3600]

#debug:
# Ts = [0, 15*60, 3600, 2*3600]
# T_da = 3600
# T_forecast = 3600

# %% 
Ne = 100

# %%
localisation = True


# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
log.write("T = " + ", ".join([str(T) for T in Ts]) +"\n\n")

log.write("Temporal Perturbation\n")
log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")

log.write("\nStatistics\n")
log.write("Ne = " +str(Ne)+"\n")
# log.write("g(u) = (u-E[u])^2\n")
# log.write("norm = "+ str(norm.__name__) + "\n")
log.close()


# %%
grid_list = []
for l_idx in range(len(ls)):
    grid_list.append( {key: args_list[l_idx][key] for key in ('nx', 'ny', 'dx', 'dy', 'gpu_ctx', 'boundary_conditions')} )

mekls = [ModelErrorKL.ModelErrorKL(gpu_stream=gpu_stream, **grid_list[l_idx], **sim_model_error_basis_args) for l_idx in range(len(ls))]


# %%
# Ensemble
SL_ensembles = []
SL_ensembles.append([])
for l_idx in range(1,len(ls)):
    SL_ensembles.append([])

for e in range(Ne):
    sim = CDKLM16.CDKLM16(**args_list[0], **init_list[0]) 
    SL_ensembles[0].append( sim )

    for l_idx in range(1, len(ls)):    
        sim = CDKLM16.CDKLM16(**args_list[l_idx], **init_list[l_idx]) 
        SL_ensembles[l_idx].append( sim )

# %% 
# Spin up
numTsteps = int((T_spinup)/sim_model_error_timestep) 
for Tstep in range(numTsteps):
    for e in range(Ne):
        SL_ensembles[0][e].step(sim_model_error_timestep, apply_stochastic_term=False)
        mekls[0].perturbSim(SL_ensembles[0][e])
        for l_idx in range(1, len(ls)):
            SL_ensembles[l_idx][e].step(sim_model_error_timestep, apply_stochastic_term=False)
            mekls[l_idx].perturbSimSimilarAs(SL_ensembles[l_idx][e], modelError=mekls[0])


for l_idx in range(len(ls)):
    for e in range(l_idx):
        assert np.abs(SL_ensembles[l_idx][e].t - T_spinup) < 0.1, "Ensembles is not at spin up time"

# loop over time
t_now = T_spinup
for t_idx, T in enumerate(Ts):

    numDAsteps = int((np.minimum(T, T_spinup + T_da)-t_now)/da_timestep)
    for DAstep in range(numDAsteps):
        
        # Forward step
        numTsteps = int(da_timestep/sim_model_error_timestep) 
        for Tstep in range(numTsteps):
            for e in range(Ne):
                SL_ensembles[0][e].step(sim_model_error_timestep, apply_stochastic_term=False)
                mekls[0].perturbSim(SL_ensembles[0][e])
                for l_idx in range(1, len(ls)):
                    SL_ensembles[l_idx][e].step(sim_model_error_timestep, apply_stochastic_term=False)
                    mekls[l_idx].perturbSimSimilarAs(SL_ensembles[l_idx][e], modelError=mekls[0])

            t_now = t_now + sim_model_error_timestep
            print(datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), ": ", t_now)

    # Save 
    for l_idx in range(len(ls)):
        np.save(output_path+"/SLensemble_"+str(T)+"_"+str(l_idx)+".npy", SLdownload(SL_ensembles[l_idx]))




