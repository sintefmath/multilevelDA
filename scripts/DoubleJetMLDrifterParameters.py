# %% [markdown]
# # Double Jet
# 
# With varioous drifter parameters

# %%
from gpuocean.utils import Common

import numpy as np
import copy
import pycuda.driver as cuda
from matplotlib import pyplot as plt

plt.rcParams["image.origin"] = "lower"


# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

# %%
import sys, os


# %% [markdown]
# ### Load Ensemble

# %%
source_path = "/home/florianb/havvarsel/multilevelDA/doublejet/scripts/DataAssimilation/DoubleJetMLDA/2023-11-09T18_07_14"

# %%
from gpuocean.utils import DoubleJetCase

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.DoubleJetParametersReplication import *
from utils.DoubleJetEnsembleInit import *

from gpuocean.ensembles import MultiLevelOceanEnsemble

def loadMLensemble(source_path):
    l_max = int((len(os.listdir(source_path+"/MLstates"))-1)/2)+1

    ML_state = []
    ML_state.append(np.load(source_path+"/MLstates/MLensemble_0.npy"))
    for l_idx in range(1, l_max):
        ML_state.append( [np.load(source_path+"/MLstates/MLensemble_"+str(l_idx)+"_0.npy"), np.load(source_path+"/MLstates/MLensemble_"+str(l_idx)+"_1.npy")] )

    args_list = []
    init_list = []

    for l_idx in range(l_max):
        if l_idx == 0: 
            ny, nx = ML_state[l_idx].shape[1:3]
        else:
            ny, nx = ML_state[l_idx][0].shape[1:3]
        doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=ny, nx=nx)
        doubleJetCase_args, doubleJetCase_init, _ = doubleJetCase.getInitConditions()

        args_list.append(doubleJetCase_args)
        init_list.append(doubleJetCase_init)

    ML_Ne = []
    ML_Ne.append(ML_state[0].shape[-1])
    for l_idx in range(1, l_max):
        ML_Ne.append(ML_state[l_idx][0].shape[-1])

    ML_ensemble = initMLensemble(ML_Ne, args_list, init_list,
                            sim_model_error_basis_args=sim_model_error_basis_args, 
                            sim_model_error_time_step=sim_model_error_timestep)
    
    MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)
    MLOceanEnsemble.upload(ML_state)

    for e in range(ML_Ne[0]):
        MLOceanEnsemble.ML_ensemble[0][e].t = np.float32(T_spinup + T_da)

    for l_idx in range(1, l_max):
        for e in range(ML_Ne[l_idx]):
            MLOceanEnsemble.ML_ensemble[l_idx][0][e].t = np.float32(T_spinup + T_da)
            MLOceanEnsemble.ML_ensemble[l_idx][1][e].t = np.float32(T_spinup + T_da)
    MLOceanEnsemble.t = np.float32(T_spinup + T_da)

    return MLOceanEnsemble


# %% [markdown]
# ### Attach drifters

# %%
# Prepare drifters
drifter_ensemble_size = 250
num_drifters = len(init_positions)

drift_dts = [60]#, 300, 900, 1800, 3600, 3*3600, 6*3600, 12*3600]

for drift_dt in drift_dts: 
    # Prepare ensemble
    MLOceanEnsemble = loadMLensemble(source_path)
    MLOceanEnsemble.attachDrifters(drifter_ensemble_size, drifterPositions=np.array(init_positions), drift_dt=drift_dt)
    
    # Forecast period
    while MLOceanEnsemble.t < T_spinup + T_da + T_forecast:
        print(MLOceanEnsemble.t)
        
        MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + np.maximum(da_timestep, drift_dt))
        MLOceanEnsemble.registerDrifterPositions()

    os.makedirs(source_path+"/mldrifters_"+str(drift_dt), exist_ok=True)
    MLOceanEnsemble.saveDriftTrajectoriesToFile(source_path+"/mldrifters_"+str(drift_dt), "mldrifters")
