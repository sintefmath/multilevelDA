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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *

# %% 
base_path = "KGainConvergence/Basin/"

# %%
from gpuocean.utils import Common

gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

# %%
# ls = [6, 7, 8, 9, 10]

# %%
# args_list = []

# for l in ls:
#     lvl_grid_args = initGridSpecs(l)
#     args_list.append( {
#         "nx": lvl_grid_args["nx"],
#         "ny": lvl_grid_args["ny"],
#         "dx": lvl_grid_args["dx"],
#         "dy": lvl_grid_args["dy"],
#         "gpu_ctx": gpu_ctx,
#         "gpu_stream": gpu_stream,
#         "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
#         } )

# %%
from utils.BasinParameters import * 


# %% 
import argparse
parser = argparse.ArgumentParser(description='Single run inputs')
parser.add_argument("-m", "--mode", required=True, type=str, choices=["T", "R", "SL", "ML", "MC"])
parser.add_argument("-Ne", "--ensembleSize", nargs="*", type=int)
parser.add_argument("-ls", "--MLlevels", nargs="*", type=int)
parser.add_argument("-L", "--SLlevel", type=int, default=9)

pargs = parser.parse_args()
mode = pargs.mode
Nes = pargs.ensembleSize
SL_level = pargs.SLlevel
ML_levels = pargs.MLlevels

# %% 
print("Reducing T_da for debugging purposes!")
# T_da = 0
T_da = 1*3600

# %%
##############################################
# TRUTH

if mode == "T": 

    grid_args = initGridSpecs(10)
    args =  {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        } 

    def writeTruth2file(T):
        true_state = truth.download(interior_domain_only=True)
        os.makedirs(os.path.join(base_path,"tmpTruth"), exist_ok=True)
        np.save(base_path+"/tmpTruth/truth_"+str(T)+".npy", np.array(true_state))

    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)


    from gpuocean.SWEsimulators import ModelErrorKL
    init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)
    sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)

    truth = make_sim(args, sample_args=sample_args, init_fields=data_args)
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
# REFERENCE 

elif mode == "R":

    grid_args = initGridSpecs(10)
    args =  {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        } 

    #############################
    # Initialisation
    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

    from utils.BasinSL import *
    SL_ensemble = initSLensemble(250, args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)


    ############################
    # First Gain
    if T_da == 0.0:
        # Observation
        true_eta, true_hu, true_hv = np.load(base_path+"/tmpTruth/truth_"+str(int(SL_ensemble[0].t))+".npy")
        
        Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_xs[0], obs_ys[0])
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))

        localisation_weights = GCweights(SL_ensemble, obs_xs[0], obs_ys[0], r) 
        SL_K = SLEnKF(SL_ensemble, obs, obs_xs[0], obs_ys[0], R=R, obs_var=slice(1,3), 
                relax_factor=relax_factor, localisation_weights=localisation_weights)




    localisation_weights_list = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        localisation_weights_list.append( GCweights(SL_ensemble, obs_x, obs_y, r) ) 

    ####################
    # Data assimilation

    while SL_ensemble[0].t < T_da:
        # Forward step
        SLstepToObservation(SL_ensemble, SL_ensemble[0].t + da_timestep)
        
        # Observation
        true_eta, true_hu, true_hv = np.load(base_path+"/tmpTruth/truth_"+str(int(SL_ensemble[0].t))+".npy")
        
        print("DA at t = ", SL_ensemble[0].t)
        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))

            SL_K = SLEnKF(SL_ensemble, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
                relax_factor=relax_factor, localisation_weights=localisation_weights_list[h])

    # Saving results
    os.makedirs(os.path.join(base_path,"tmpRefGain"), exist_ok=True)
    np.save(base_path+"tmpRefGain/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), SL_K)

    os.makedirs(os.path.join(base_path,"RefGain"), exist_ok=True)
    np.save(base_path+"/RefGain/slK_"+str(T_da)+"_"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), SL_K)

    # Crash to let OS do the gc
    os._exit(0)



##############################################
# SINGLE LEVEL

elif mode == "SL":

    grid_args = initGridSpecs(SL_level)
    args =  {
        "nx": grid_args["nx"],
        "ny": grid_args["ny"],
        "dx": grid_args["dx"],
        "dy": grid_args["dy"],
        "gpu_ctx": gpu_ctx,
        "gpu_stream": gpu_stream,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
        } 
    
    #####################
    # Initialisation
    data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

    from utils.BasinSL import *
    SL_ensemble = initSLensemble(Nes[0], args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)

    dummy_truth = make_sim(args, 
                           sample_args=sample_args, 
                           init_fields=make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist))

    #####################
    # First Gain
    if T_da == 0.0:
        # Observation
        true_eta, true_hu, true_hv = np.load(base_path+"/tmpTruth/truth_"+str(int(SL_ensemble[0].t))+".npy")
        
        Hx, Hy = SLobsCoord2obsIdx([dummy_truth], obs_xs[0], obs_ys[0])
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))

        localisation_weights = GCweights(SL_ensemble, obs_xs[0], obs_ys[0], r) 
        SL_K = SLEnKF(SL_ensemble, obs, obs_xs[0], obs_ys[0], R=R, obs_var=slice(1,3), 
                relax_factor=relax_factor, localisation_weights=localisation_weights)




    localisation_weights_list = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        localisation_weights_list.append( GCweights(SL_ensemble, obs_x, obs_y, r) ) 

    ####################
    # Data assimilation

    while SL_ensemble[0].t < T_da:
        # Forward step
        SLstepToObservation(SL_ensemble, SL_ensemble[0].t + da_timestep)
        
        # Observation
        true_eta, true_hu, true_hv = np.load(base_path+"/tmpTruth/truth_"+str(int(SL_ensemble[0].t))+".npy")
        
        print("DA at t = ", SL_ensemble[0].t)
        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = SLobsCoord2obsIdx([dummy_truth], obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))

            SL_K = SLEnKF(SL_ensemble, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
                relax_factor=relax_factor, localisation_weights=localisation_weights_list[h])

    # Saving results
    os.makedirs(os.path.join(base_path,"tmpSLGain"), exist_ok=True)
    np.save(base_path+"/tmpSLGain/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), SL_K)

    os.makedirs(os.path.join(base_path,"tmpSLmean"), exist_ok=True)
    np.save(base_path+"/tmpSLmean/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), SLestimate(SL_ensemble, np.mean))

    # Crash to let OS do the gc
    os._exit(0)



##############################################
# MULTI LEVEL

elif mode == "ML":
    
    ls = ML_levels

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

    # Initialisation
    data_args_list = []
    for l_idx in range(len(ls)): 
        data_args_list.append( make_init_steady_state(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) )

    from gpuocean.ensembles import MultiLevelOceanEnsemble
    MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsembleCase(Nes, args_list, data_args_list, sample_args, make_sim,
                                init_model_error_basis_args=init_model_error_basis_args, 
                                sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_timestep=sim_model_error_timestep)

    from gpuocean.dataassimilation import MLEnKFOcean
    MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

    log = open(os.path.join(base_path, "logMLEnKF.txt"), "a")

    #####################
    # First Gain
    if T_da == 0.0:
        # Observation
        true_eta, true_hu, true_hv = np.load(base_path+"/tmpTruth/truth_"+str(int(MLOceanEnsemble.t))+".npy")

        Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_xs[0], obs_ys[0])
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))
        
        ML_K = MLEnKF.assimilate(MLOceanEnsemble, obs, obs_xs[0], obs_ys[0], R, 
                                        r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                        min_localisation_level=min_location_level,
                                        log=log)



    precomp_GC = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        precomp_GC.append( MLEnKF.GCweights(obs_x, obs_y, r) )
    
    #####################
    # Data assimilation
    while MLOceanEnsemble.t < T_da:
        # Forward step
        MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)

        # DA step
        true_eta, true_hu, true_hv = np.load(base_path+"/tmpTruth/truth_"+str(int(MLOceanEnsemble.t))+".npy")

        print("DA at t = ", MLOceanEnsemble.t)
        
        log.write("MLOceanEnsemble at " + str(MLOceanEnsemble.t) +"\n")

        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))
            
            log.write("MLEnKF for obs " + str(h) + ". ")
            ML_K = MLEnKF.assimilate(MLOceanEnsemble, obs, obs_x, obs_y, R, 
                                    r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                    min_localisation_level=min_location_level,
                                    precomp_GC=precomp_GC[h],
                                    log=log)
    
    log.close()

    # Saving results
    os.makedirs(os.path.join(base_path,"tmpMLGain"), exist_ok=True)
    np.save(base_path+"/tmpMLGain/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), ML_K.reshape(3,MLOceanEnsemble.nys[-1],MLOceanEnsemble.nxs[-1],ML_K.shape[-1]))

    os.makedirs(os.path.join(base_path,"tmpMLmean"), exist_ok=True)
    np.save(base_path+"/tmpMLmean/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), MLOceanEnsemble.estimate(np.mean))

    # Crash to let OS do the gc
    os._exit(0)
    


##############################################
# MONTE CARLO

elif mode == "MC":
    
    ls = [6, 7, 8, 9]

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

    # Initialisation
    data_args_list = []
    for l_idx in range(len(args_list)): #TODO: Check indices 
        data_args_list.append( make_init_steady_state(args_list, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) )

    from gpuocean.ensembles import MultiLevelOceanEnsemble
    MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsembleCase(Nes, args_list, data_args_list, sample_args, make_sim,
                                init_model_error_basis_args=init_model_error_basis_args, 
                                sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_timestep=sim_model_error_timestep)


    #####################
    # Data assimilation
    while MLOceanEnsemble.t < T_da:
        # Forward step
        MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)

        # DA step
        print("no DA at t = ", MLOceanEnsemble.t)

    os.makedirs(os.path.join(base_path,"tmpMLmean"), exist_ok=True)
    np.save(base_path+"/tmpMLmean/"+datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), MLOceanEnsemble.estimate(np.mean))

    # Crash to let OS do the gc
    os._exit(0)
    


# %%
