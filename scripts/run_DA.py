# import threading
# import time
# import psutil

# def printmemorytofile():
#     while True:
#         with open("memory.txt", "a") as f:
#             f.write(f"{ psutil.virtual_memory()[2]} {psutil.virtual_memory()[3]/1000000000}\n")
#         time.sleep(1)
# printthread = threading.Thread(target=printmemorytofile)
# printthread.start()
# %% [markdown]
# # Multi Level Simulation

# %% [markdown]
# ### Classes and modules

# %%
import os, sys

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

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16

# %%
gpu_ctx = Common.CUDAContext()

# %% 
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

if not os.path.exists("OutputDA"):
    os.makedirs("OutputDA")

output_path = "OutputDA/"+timestamp
os.makedirs(output_path)    


log_f = open(output_path+"/log.txt", 'w')

def log2file(file, string):
    file.write(string)
    print(string)

log2file(log_f, timestamp + ': Staring ML-DA simulation'+'\n')

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
KLSampler = KarhunenLoeve_Sampler(t_splits, wind_N)
wind_weight = wind_bump(KLSampler.N,KLSampler.N)

# %% [markdown]
# ## Data Assimilation

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

# %% [markdown]
# ### Truth

# %%
data_args = initLevel(ls[-1])
true_wind = wind_sample(KLSampler, T, wind_weight=wind_weight)
truth = CDKLM16.CDKLM16(gpu_ctx, **data_args, wind=true_wind)
truth.step(T)

# %%
true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)

# %%
Hfield = np.zeros((truth.ny,truth.nx))
Hy, Hx = 800, 600
Hfield[Hy,Hx] = 1.0

R = [0.0001, 0.01, 0.01]

obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

# %%
fig, axs = imshow3([true_eta, true_hu, true_hv])
axs[0].scatter(Hx,Hy, marker="x", c="black", label=str(round(obs[0],5)), s=100)
axs[0].legend(labelcolor="black", fontsize=15)
axs[1].scatter(Hx, Hy, marker="x", c="black", label=str(round(obs[1],5)), s=100)
axs[1].legend(labelcolor="black", fontsize=15)
axs[2].scatter(Hx, Hy, marker="x", c="black", label=str(round(obs[2],5)), s=100)
axs[2].legend(labelcolor="black", fontsize=15)
fig.suptitle("Truth", y=0.85)
plt.savefig(output_path+"/Truth.png")


# %% [markdown]
# ### Multi-level Ensemble

# %%
vars_file = "../scripts/OutputVarianceLevels/Rossby-vars-no_wind_dir.npy"
diff_vars_file = "../scripts/OutputVarianceLevels/Rossby-diff_vars-no_wind_dir.npy"


# %%
rossbyAnalysis = RossbyAnalysis(ls, vars_file, diff_vars_file)
Nes = rossbyAnalysis.optimal_Ne(tau=1.5*1e-7)
# Nes = np.array([500,100,50,10,5])

log2file(log_f, "Ensemble size of multilevel"+"\n")
log2file(log_f, str(Nes)+"\n")

# %%
log2file(log_f, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")+ ": Starting ensemble init" + "\n")

ML_ensemble = initMLensemble(gpu_ctx, ls, Nes, KLSampler, wind_weight, T)
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)

# %%
log2file(log_f, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + "Starting ensemble simulation"+"\n")
MLOceanEnsemble.step(T)

# %%
ML_prior_state = copy.deepcopy(MLOceanEnsemble.download())

# %%
fig, axs = imshow3(MLOceanEnsemble.estimate(np.mean))
plt.savefig(output_path+"/MLprior_mean.png")

fig, axs = imshow3var(MLOceanEnsemble.estimate(np.var))
plt.savefig(output_path+"/MLprior_var.png")

# %%
log2file(log_f,  datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + ":Starting MLDA" +"\n")
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)
MLEnKF.assimilate(MLOceanEnsemble, obs, Hx, Hy, R)
log2file(log_f,  datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + ":Finished MLDA" +"\n")

# %%
fig, axs = imshow3(MLOceanEnsemble.estimate(np.mean))
plt.savefig(output_path+"/MLposterior_mean.png")

fig, axs = imshow3var(MLOceanEnsemble.estimate(np.var))
plt.savefig(output_path+"/MLposterior_var.png")


# %% [markdown]
# ### Single-level Ensemble

# %%
SL_Ne = int(round(rossbyAnalysis.work(Nes)/rossbyAnalysis._level_work(ls[-1])))
log2file(log_f, "Ensemble size of singlelevel"+"\n")
log2file(log_f, str(SL_Ne)+"\n")

# %%
log2file(log_f,  datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + ": Starting ensemble init" +"\n")
SL_ensemble = initSLensemble(gpu_ctx, ls, SL_Ne, KLSampler, wind_weight, T)


# %%
log2file(log_f, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + ": Starting ensemble simulation" +"\n")

SLstep(SL_ensemble, T)

# %%
SL_state = SLdownload(SL_ensemble)

SL_prior = copy.deepcopy(SL_state)

# %%
fig, axs = imshow3(np.average(SL_prior, axis=-1))
plt.savefig(output_path+"/SLprior_mean.png")

fig, axs = imshow3var(np.var(SL_prior, axis=-1))
plt.savefig(output_path+"/SLprior_var.png")

# %%
log2file(log_f, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")+ ": Starting SLDA" +"\n")
SL_posterior = SLEnKF(SL_ensemble, obs, Hx, Hy, R)
log2file(log_f, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")+ ": Finished SLDA" +"\n")

# %%
fig, axs = imshow3(np.average(SL_posterior, axis=-1))
plt.savefig(output_path+"/SL_posterior_mean.png")

fig, axs = imshow3var(np.var(SL_posterior, axis=-1))
plt.savefig(output_path+"/SL_posterior_var.png")

# %% 
log_f.close()
