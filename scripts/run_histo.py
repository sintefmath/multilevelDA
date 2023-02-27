# %% [markdown]
# # Multi Resolution Simulation

# %% [markdown]
# ### Classes and modules

# %%
import os
import sys

#Import packages we need
import numpy as np
import datetime
import copy
import gc

#For plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common, WindStress
from gpuocean.SWEsimulators import CDKLM16

# %%
gpu_ctx = Common.CUDAContext()

# %%
# %% 
import datetime
print(datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S") + ": Starting ML experiment")

output_path = "OutputHisto/bash_loop"
os.makedirs(output_path, exist_ok=True)

# %% [markdown]
# Rossby utils

# %%
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
sys.path.insert(0, "/home/florianb/havvarsel/multilevelDA/")
from utils.WindPerturb import *
from utils.RossbyInit import *
from utils.RossbyEnsembleInit import *
from utils.RossbyAnalysis import *

from utils.RossbySL import *

# %%
from gpuocean.ensembles import MultiLevelOceanEnsemble
from gpuocean.dataassimilation import MLEnKFOcean

# %%
wind_N = 100
t_splits = 26

# %%
KLSampler = KarhunenLoeve_Sampler(t_splits + 3, wind_N, decay=1.15, scaling=0.9)
wind_weight = wind_bump(KLSampler.N,KLSampler.N)

# %% [markdown]
# ## Data Assimilation
# 

# %%
from skimage.measure import block_reduce

# %%
def imshow3(etahuhv):
    fig, axs = plt.subplots(1,3, figsize=(15,10))
    im = axs[0].imshow(etahuhv[0], vmin=-0.05, vmax=0.05, cmap="coolwarm")
    plt.colorbar(im, ax=axs[0], shrink=0.5)
    axs[0].set_title("$\eta$", fontsize=15)

    im = axs[1].imshow(etahuhv[1], vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, ax=axs[1], shrink=0.5)
    axs[1].set_title("$hu$", fontsize=15)

    im = axs[2].imshow(etahuhv[2], vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, ax=axs[2], shrink=0.5)
    axs[2].set_title("$hv$", fontsize=15)

    return fig, axs


def imshow3var(est_var):
    fig, axs = plt.subplots(1,3, figsize=(15,10))
    im = axs[0].imshow(est_var[0], vmin=0.0, vmax=0.00005, cmap="Reds")
    plt.colorbar(im, ax=axs[0], shrink=0.5)
    axs[0].set_title("$\eta$", fontsize=15)

    im = axs[1].imshow(est_var[1], vmin=0, vmax=0.25, cmap="Reds")
    plt.colorbar(im, ax=axs[1], shrink=0.5)
    axs[1].set_title("$hu$", fontsize=15)

    im = axs[2].imshow(est_var[2], vmin=0, vmax=0.25, cmap="Reds")
    plt.colorbar(im, ax=axs[2], shrink=0.5)
    axs[2].set_title("$hv$", fontsize=15)

    return fig, axs


# %%
ls = [6, 7, 8, 9, 10]
T = 125000
T_forecast = 12500

# %% [markdown]
# ### Variance-level analysis

# %% 
vars_file = "/home/florianb/havvarsel/multilevelDA/scripts/OutputVarianceLevels/2023-02-24T16_29_26_Rossby/vars.npy"
diff_vars_file = "/home/florianb/havvarsel/multilevelDA/scripts/OutputVarianceLevels/2023-02-24T16_29_26_Rossby/diff_vars.npy"

# %%
rossbyAnalysis = RossbyAnalysis(ls, vars_file, diff_vars_file)
ML_Nes = rossbyAnalysis.optimal_Ne(tau=3.0*1e-7)
#print(ML_Nes)
#ML_Nes = [500,10,5,3,2]
# %%
SL_Ne = np.int32(np.ceil(rossbyAnalysis.work(ML_Nes)/rossbyAnalysis._level_work(ls[-1])))

# %% [markdown]
# ### Util functions

# %%
def generate_truth():
    data_args = initLevel(ls[-1])
    true_wind = wind_sample(KLSampler, T + 15000, wind_weight=wind_weight, wind_speed=0.0)
    truth = CDKLM16.CDKLM16(gpu_ctx, **data_args, wind=true_wind)
    truth.step(T)

    return truth

def generate_obs_from_truth(truth, Hy, Hx, R):
    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)

    Hfield = np.zeros((truth.ny,truth.nx))
    
    Hfield[Hy,Hx] = 1.0

    obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

    return obs

Hy, Hx = 800, 600
R = [0.0001, 0.01, 0.01]


# %% [markdown]
# #### Rank utils

# %%
def MLcdf4true(truth, Hx, Hy, ML_ensemble, ML_final_state):

    Nes = np.zeros(len(ML_ensemble))
    Nes[0] = len(ML_ensemble[0])
    for l_idx in range(1,len(ML_ensemble)):
        Nes[l_idx] = len(ML_ensemble[l_idx][0])

    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    true_values = np.array([true_eta[Hy, Hx], true_hu[Hy, Hx], true_hv[Hy, Hx]])

    Xs = np.linspace(0, ML_ensemble[-1][0][0].nx * ML_ensemble[-1][0][0].dx, ML_ensemble[-1][0][0].nx)
    Ys = np.linspace(0, ML_ensemble[-1][0][0].ny * ML_ensemble[-1][0][0].dy, ML_ensemble[-1][0][0].ny)
    X, Y = np.meshgrid(Xs, Ys)

    lvl_Xs = np.linspace(0, ML_ensemble[0][0].nx * ML_ensemble[0][0].dx, ML_ensemble[0][0].nx)
    lvl_Ys = np.linspace(0, ML_ensemble[0][0].ny * ML_ensemble[0][0].dy, ML_ensemble[0][0].ny)
    lvl_X, lvl_Y = np.meshgrid(lvl_Xs, lvl_Ys)

    obs_idxs = np.unravel_index(np.argmin((lvl_X - X[0,Hx])**2 + (lvl_Y - Y[Hy,0])**2), ML_final_state[0][0].shape[:-1])

    ML_Fy = 1/Nes[0] * np.sum(ML_final_state[0][:,obs_idxs[0],obs_idxs[1],:] < true_values[:,np.newaxis], axis=1)

    for l_idx in range(1,len(ls)):
        l = ls[l_idx]

        lvl_Xs0 = np.linspace(0, ML_ensemble[l_idx][0][0].nx * ML_ensemble[l_idx][0][0].dx, ML_ensemble[l_idx][0][0].nx)
        lvl_Ys0 = np.linspace(0, ML_ensemble[l_idx][0][0].ny * ML_ensemble[l_idx][0][0].dy, ML_ensemble[l_idx][0][0].ny)
        lvl_X0, lvl_Y0 = np.meshgrid(lvl_Xs0, lvl_Ys0)
        obs_idxs0 = np.unravel_index(np.argmin((lvl_X0 - X[0,Hx])**2 + (lvl_Y0 - Y[Hy,0])**2), ML_final_state[l_idx][0][0].shape[:-1])

        lvl_Xs1 = np.linspace(0, ML_ensemble[l_idx][1][0].nx * ML_ensemble[l_idx][1][0].dx, ML_ensemble[l_idx][1][0].nx)
        lvl_Ys1 = np.linspace(0, ML_ensemble[l_idx][1][0].ny * ML_ensemble[l_idx][1][0].dy, ML_ensemble[l_idx][1][0].ny)
        lvl_X1, lvl_Y1 = np.meshgrid(lvl_Xs1, lvl_Ys1)
        obs_idxs1 = np.unravel_index(np.argmin((lvl_X1 - X[0,Hx])**2 + (lvl_Y1 - Y[Hy,0])**2), ML_final_state[l_idx][1][0].shape[:-1])

        ML_Fy += 1/Nes[l_idx] * np.sum(1 * (ML_final_state[l_idx][0][:,obs_idxs0[0],obs_idxs0[1],:] < true_values[:,np.newaxis]) - 1 * (ML_final_state[l_idx][1][:,obs_idxs1[0],obs_idxs1[1],:] < true_values[:,np.newaxis]), axis=1)
    
    return ML_Fy

# %%
def SLcdf4true(truth, Hx, Hy, SL_ensemble, SL_final_state):
    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    true_values = np.array([true_eta[Hy, Hx], true_hu[Hy, Hx], true_hv[Hy, Hx]])

    SL_Fy = np.sum(SL_final_state[:,Hy,Hx,:] < true_values[:,np.newaxis], axis=1)/SL_Ne    

    return SL_Fy

# %% [markdown]
# ### Rank computations

# %%
rank_idxs = [[Hx,Hy], [850, 650], [750,650], [750, 550], [850, 550]]

MLrank_files = []
for f in range(len(rank_idxs)):
    MLrank_files.append(output_path+"/MLranks_"+str(rank_idxs[f][0])+"_"+str(rank_idxs[f][1]))

# %%
# -> the actuval loop
truth = generate_truth()
truth.step(T)
obs = generate_obs_from_truth(truth, Hy, Hx, R)
truth.step(T_forecast)



ML_ensemble = initMLensemble(gpu_ctx, ls, ML_Nes, KLSampler, wind_weight, T + T_forecast, 0.0)
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)

MLOceanEnsemble.step(T)

MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)
MLEnKF.assimilate(MLOceanEnsemble, obs, Hx, Hy, R, r = 2.5*1e7, relax_factor = 1.0)

MLOceanEnsemble.step(T_forecast)
ML_final_state = MLOceanEnsemble.download()



for f in range(len(rank_idxs)):
    with open(MLrank_files[f], "a") as file:
        file.write(",".join(map(str, MLcdf4true(truth, rank_idxs[f][0], rank_idxs[f][1], ML_ensemble, ML_final_state))) + "\n")
        file.close()


# %%

 

# SLrank_files = []
# for f in range(len(rank_idxs)):
#     SLrank_files.append(output_path+"/SLranks_"+str(rank_idxs[f][0])+"_"+str(rank_idxs[f][1]))

# for r in range(rank_N):
#     print(datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S") + ': SLrun ' + str(r) +'\n')

#     truth = generate_truth()
#     truth.step(T)
#     obs = generate_obs_from_truth(truth, Hy, Hx, R)
#     truth.step(12500)
    
#     SL_ensemble = initSLensemble(gpu_ctx, ls, SL_Ne, KLSampler, wind_weight, T + 15000, 0.0)
#     SLstep(SL_ensemble, T)
#     SL_posterior = SLEnKF(SL_ensemble, obs, Hx, Hy, R)
#     SLstep(SL_ensemble, 12500)
#     SL_final_state = SLdownload(SL_ensemble)

#     for f in range(len(rank_idxs)):
#         with open(SLrank_files[f], "a") as file:
#             file.write(",".join(map(str, SLcdf4true(truth, rank_idxs[f][0], rank_idxs[f][1], SL_ensemble, SL_final_state)))+"\n")
#             file.close()
