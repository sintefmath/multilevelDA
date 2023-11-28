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

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../')))
from utils.DoubleJetPlot import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %%
import argparse
parser = argparse.ArgumentParser(description='Ensemble inputs')
parser.add_argument("-ls", "--level", required=True, nargs="*", type=int, default=[7, 8])
parser.add_argument("-Ne", "--ensembleSize", required=True, nargs="*", type=int, default=[50, 25])
parser.add_argument("--timestamp", type=str, default="")
parser.add_argument("-n", "--experimentNumber", type=str, default="")

pargs = parser.parse_args()
ls = pargs.level
ML_Nes = pargs.ensembleSize
timestamp = pargs.timestamp
exp_n = pargs.experimentNumber

assert len(ls) == len(ML_Nes), "Non-matching levels and ensemble sizes"
# ls = [7, 8]
# ML_Nes = [100, 25]


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
output_path = os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), 'RankHistograms'))
if timestamp is not "":
    output_path = os.path.join(output_path, timestamp)

os.makedirs(output_path, exist_ok=True)

# %% 
truth = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init)
truth.updateDt()
truth.setKLModelError(**sim_model_error_basis_args)
truth.model_time_step = sim_model_error_timestep

# %%
# Ensemble
from utils.DoubleJetEnsembleInit import *
ML_ensemble = initMLensemble(ML_Nes, args_list, init_list,
                             sim_model_error_basis_args=sim_model_error_basis_args, 
                             sim_model_error_time_step=sim_model_error_timestep)

from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)

# %%
# Rank locations 
rank_xs = np.array([0.5, 128.5, 256.5, 384.5])*2601.5625
rank_ys = np.array([0.5,  42.5, 84.5, 126.5, 168.5, 210.5])*2601.5625

rank_obs_xs, rank_obs_ys = np.array(np.meshgrid(rank_xs, rank_ys)).T.reshape(-1, 2).T

# Convert to indices
rank_idxs = MLOceanEnsemble.loc2idxs(rank_obs_xs[0], rank_obs_ys[0])

rank_idxs[0] = np.array(rank_idxs[0])
for l_idx in range(1, len(ls)):
    rank_idxs[l_idx][0] = np.array(rank_idxs[l_idx][0])
    rank_idxs[l_idx][1] = np.array(rank_idxs[l_idx][1])

for r_idx in range(1, len(rank_obs_xs)):
    tmp_rank_idxs = MLOceanEnsemble.loc2idxs(rank_obs_xs[r_idx], rank_obs_ys[r_idx])
    
    rank_idxs[0] = np.c_[rank_idxs[0], tmp_rank_idxs[0]]
    for l_idx in range(1, len(ls)):
        rank_idxs[l_idx][0] = np.c_[rank_idxs[l_idx][0], tmp_rank_idxs[l_idx][0]]
        rank_idxs[l_idx][1] = np.c_[rank_idxs[l_idx][1], tmp_rank_idxs[l_idx][1]]


# %% 
def write2file(MLOceanEnsemble, truth):
    """
    File location: RankHistograms/<timestamp>/<exp_n>_***

    Structure:
    - One file per level/level partners
    - Shape 3 x N_e^l x #RanksLocs 
    """

    t = int((MLOceanEnsemble.t - T_spinup)/3600)
    
    t_path = os.path.join(output_path, str(t))
    os.makedirs(t_path, exist_ok=True)
    
    ML_state = MLOceanEnsemble.download()
    np.save(t_path+"/"+exp_n+"_MLvalues_0.npy", np.array(ML_state[0][:,rank_idxs[0][0],rank_idxs[0][1]]))
    for l_idx in range(1,len(ls)):
        np.save(t_path+"/"+exp_n+"_MLvalues_"+str(l_idx)+"_0.npy", np.array(ML_state[l_idx][0][:,rank_idxs[l_idx][0][0],rank_idxs[l_idx][0][1]]))
        np.save(t_path+"/"+exp_n+"_MLvalues_"+str(l_idx)+"_1.npy", np.array(ML_state[l_idx][1][:,rank_idxs[l_idx][1][0],rank_idxs[l_idx][1][1]]))

    true_state = np.array(truth.download(interior_domain_only=True))
    np.save(t_path+"/"+exp_n+"_TRUEvalues.npy", true_state[:,rank_idxs[-1][0][0],rank_idxs[-1][0][1]])


# %%
from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

precomp_GC = []
for obs_x, obs_y in zip(obs_xs, obs_ys):
    precomp_GC.append( MLEnKF.GCweights(obs_x, obs_y, r) )

# %%
##########################
# Spin up period
truth.dataAssimilationStep(T_spinup)
MLOceanEnsemble.stepToObservation(T_spinup)
    
# %% 
# DA period
write2file(MLOceanEnsemble, truth)

Ts = [3600, 6*3600, 12*3600, 24*3600]

for t_idx in range(len(Ts)):
    print("Ts = ", Ts[t_idx])
    while MLOceanEnsemble.t < T_spinup + Ts[t_idx]:
        # Forward step
        MLOceanEnsemble.stepToObservation( np.minimum(MLOceanEnsemble.t + da_timestep, T_spinup + Ts[t_idx]) )
        truth.dataAssimilationStep( MLOceanEnsemble.t )

        assert np.abs( MLOceanEnsemble.t - truth.t ) < 0.01, "Truth and ensemble out of sync"

        # DA step
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
        ML_state = copy.deepcopy(MLOceanEnsemble.download())
        
        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))
            
            ML_state = MLEnKF.assimilate(ML_state, obs, obs_x, obs_y, R, 
                                    r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                    min_localisation_level=min_location_level,
                                    precomp_GC=precomp_GC[h])
        
        MLOceanEnsemble.upload(ML_state)

    write2file(MLOceanEnsemble, truth)


MLOceanEnsemble.stepToObservation( T_spinup + Ts[-1] + 3*3600 )
truth.dataAssimilationStep( MLOceanEnsemble.t )

write2file(MLOceanEnsemble, truth)
