# %% [markdown]
# # Error convergence

# %% [markdown]
# ### Classes and modules

# %%
import os, sys

#Import packages we need
import numpy as np
import datetime

import pycuda.driver as cuda


# %% [markdown]
# GPU Ocean-modules:

# %%
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *


# %%
from gpuocean.utils import Common

gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

# %%
ls = [6, 7, 8, 9, 10]

# %%
args_list = []

for l in ls:
    lvl_grid_args = initGridSpecs(l)
    args_list.append( {
        "nx": lvl_grid_args["nx"],
        "ny": lvl_grid_args["ny"],
        "dx": lvl_grid_args["dx"],
        "dy": lvl_grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        } )

# %%
from utils.BasinParameters import * 


# %% 
import argparse
parser = argparse.ArgumentParser(description='Single run inputs')
parser.add_argument("-m", "--mode", required=True, type=str, choices=["T", "SL", "ML"])
parser.add_argument("-Ne", "--ensembleSize", nargs="*", type=int)

pargs = parser.parse_args()
mode = pargs.mode
Nes = pargs.ensembleSize

# %% 
print("Reducing T_da for debugging purposes!")
T_da = 0

# %%
##############################################
# Truth
if mode == "T": 

    def writeTruth2file(T):
        true_state = truth.download(interior_domain_only=True)
        os.makedirs("tmpTruth", exist_ok=True)
        np.save("tmpTruth/truth_"+str(T)+".npy", np.array(true_state))

    data_args = make_init_steady_state(args_list[-1], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)


    from gpuocean.SWEsimulators import ModelErrorKL
    init_mekl = ModelErrorKL.ModelErrorKL(**args_list[-1], **init_model_error_basis_args)
    sim_mekl = ModelErrorKL.ModelErrorKL(**args_list[-1], **sim_model_error_basis_args)

    truth = make_sim(args_list[-1], sample_args=sample_args, init_fields=data_args)
    init_mekl.perturbSim(truth)
    truth.model_error = sim_mekl
    truth.model_time_step = sim_model_error_timestep

    writeTruth2file(int(truth.t))
    while truth.t < T_da:
        # Forward step
        truth.dataAssimilationStep(truth.t+da_timestep)
        # DA time
        writeTruth2file(int(truth.t))


##############################################
# Single Level 
elif mode == "SL":
    
    # Initialisation
    data_args = make_init_steady_state(args_list[-1], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

    from utils.BasinSL import *
    SL_ensemble = initSLensemble(Nes[0], args_list[-1], data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)

    localisation_weights_list = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        localisation_weights_list.append( GCweights(SL_ensemble, obs_x, obs_y, r) ) 

    # Data assimilation
    while SL_ensemble[0].t < T_da:
        # Forward step
        SLstepToObservation(SL_ensemble, SL_ensemble[0].t + da_timestep)
        
        # Observation
        true_eta, true_hu, true_hv = np.load("tmpTruth/truth_"+str(int(SL_ensemble[0].t))+".npy")
        
        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

            SL_K = SLEnKF(SL_ensemble, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
                relax_factor=relax_factor, localisation_weights=localisation_weights_list[h])

    # Saving results
    true_eta, true_hu, true_hv = np.load("tmpTruth/truth_"+str(int(SL_ensemble[0].t))+".npy")
    mean_eta, mean_hu, mean_hv = SLestimate(SL_ensemble, np.mean)
    
    os.makedirs("tmpSL", exist_ok=True)
    np.save("tmpSL/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), np.array([true_eta-mean_eta, true_hu-mean_hu, true_hv-mean_hv]))

    # Crash to let OS do the gc
    os._exit(0)

##############################################
# Multi Level 
elif mode == "ML":
    
    # Initialisation
    data_args_list = []
    for l_idx in range(len(args_list)):
        data_args_list.append( make_init_steady_state(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) )
    
    from gpuocean.ensembles import MultiLevelOceanEnsembleCase
    MLOceanEnsemble = MultiLevelOceanEnsembleCase.MultiLevelOceanEnsemble(Nes, args_list, data_args_list, sample_args, make_sim,
                                init_model_error_basis_args=init_model_error_basis_args, 
                                sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_timestep=sim_model_error_timestep)

    from gpuocean.dataassimilation import MLEnKFOcean
    MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

    precomp_GC = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        precomp_GC.append( MLEnKF.GCweights(obs_x, obs_y, r) )

    # Data assimilation
    while MLOceanEnsemble.t < T_da:
        # Forward step
        MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)

        # DA step
        true_eta, true_hu, true_hv = np.load("tmpTruth/truth_"+str(int(MLOceanEnsemble.t))+".npy")

        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)
            
            ML_K = MLEnKF.assimilate(MLOceanEnsemble, obs, obs_x, obs_y, R, 
                                    r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                    min_localisation_level=min_location_level,
                                    precomp_GC=precomp_GC[h])
            
    # Saving results
    true_eta, true_hu, true_hv = np.load("tmpTruth/truth_"+str(int(MLOceanEnsemble.t))+".npy")
    mean_eta, mean_hu, mean_hv = MLOceanEnsemble.estimate(np.mean)

    os.makedirs("tmpML", exist_ok=True)
    np.save("tmpML/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), np.array([true_eta-mean_eta, true_hu-mean_hu, true_hv-mean_hv]))

    # Crash to let OS do the gc
    os._exit(0)
    


# %%
