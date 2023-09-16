# %% [markdown]
# # Multi Level Statistics

# %% [markdown]
# ### Classes and modules

# %%

#Import packages we need
import numpy as np
import sys, os
import copy

#For plotting
import matplotlib
from matplotlib import pyplot as plt

import pycuda.driver as cuda

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "MLproperties/DoubleJet/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")

import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.DoubleJetPlot import *
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
# ## Ensemble

# %% 
xorwow_seeds = len(ls)*[None]
np_seeds = len(ls)*[None]
enkf_seed = None

# xorwow_seeds = [1, 2]
# np_seeds = [3, 4]
# enkf_seed = 5


# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

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
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

os.makedirs(output_path+"/figs")

def makePlots():
    eta_vlim = 4
    huv_vlim = 750
    ##########################
    # Fields
    cmap="coolwarm"

    fig, axs = plt.subplots(len(sims),3, figsize=(15,10))
    
    for l_idx in range(len(sims)):
        eta, hu, hv = sims[l_idx].download(interior_domain_only=False)

        im = axs[l_idx,0].imshow(eta, vmin=-eta_vlim, vmax=eta_vlim, cmap=cmap, origin="lower")
        im = axs[l_idx,1].imshow(hu, vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap, origin="lower")
        im = axs[l_idx,2].imshow(hv, vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap, origin="lower")

        if l_idx == 0: 
            labels = ["$\eta$", "$hu$", "$hv$"]
            for i in range(3):
                ax_divider = make_axes_locatable(axs[0,i])
                ax_cb = ax_divider.append_axes("top", size="10%", pad="5%")
                cbar = colorbar(im, cax=ax_cb, orientation="horizontal")
                ax_cb.xaxis.set_ticks_position("top")
                ax_cb.set_title(labels[i])


    plt.savefig(output_path+"/figs/state_"+str(int(sims[0].t))+".pdf", bbox_inches="tight")
    plt.close("all")


    ###########################
    # Cross section

    fig, axs = plt.subplots(1, 3, figsize=(15,3))

    axs[0].set_xlim((0, args_list[-1]["ny"]))
    axs[0].set_ylim((-eta_vlim,eta_vlim))

    axs[1].set_xlim((0, args_list[-1]["ny"]))
    axs[1].set_ylim((-huv_vlim,huv_vlim))

    axs[2].set_xlim((0, args_list[-1]["ny"]))
    axs[2].set_ylim((-huv_vlim,huv_vlim))


    for l_idx, sim in enumerate(sims):
        eta, hu, hv = sim.download(interior_domain_only=True)
        x_idx = int(eta.shape[1]/2)
        axs[0].plot(eta[:,x_idx].repeat(2**(len(ls)-l_idx-1)))
        axs[1].plot(hu[:,x_idx].repeat(2**(len(ls)-l_idx-1)))
        axs[2].plot(hv[:,x_idx].repeat(2**(len(ls)-l_idx-1)), label=str(eta.shape[1])+" x "+str(eta.shape[0]))

    plt.legend(labelcolor="black")
    plt.savefig(output_path+"/figs/crossSec"+str(int(sims[0].t))+".pdf", bbox_inches="tight")
    plt.close("all")


# %%
# Pair of sims
mekl_stream = cuda.Stream()
mekls = []
for l_idx in range(len(args_list)): 
    grid_args = {key: args_list[l_idx][key] for key in ('nx', 'ny', 'dx', 'dy', 'gpu_ctx', 'boundary_conditions')}
    mekls.append( ModelErrorKL.ModelErrorKL(gpu_stream=mekl_stream, **grid_args, **sim_model_error_basis_args, 
                                    xorwow_seed=xorwow_seeds[l_idx], np_seed=np_seeds[l_idx]) )

sims = []
for l_idx in range(len(ls)):
    sim = CDKLM16.CDKLM16(**args_list[l_idx], **init_list[l_idx]) 
    sim.model_error = mekls[l_idx]
    sim.model_time_step = 60.0
    sims.append(sim)

# %%


# Forward step
numTsteps = int((T_spinup + T_da + T_forecast)/sim_model_error_timestep) 
for Tstep in range(numTsteps):
    sims[0].step(sim_model_error_timestep, apply_stochastic_term=False)
    mekls[0].perturbSim(sims[0])
    for l_idx in range(1, len(ls)):
        sims[l_idx].step(sim_model_error_timestep, apply_stochastic_term=False)
        mekls[l_idx].perturbSimSimilarAs(sims[l_idx], modelError=mekls[0])

    if sims[0].t % 3600 < 0.1:
        print("plotting at ", sims[0].t)
        makePlots()
    

