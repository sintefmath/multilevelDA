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

import pycuda.driver as cuda

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "VarianceLevelsDA/Basin/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

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
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }


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
steady_state_bump_a = 3
steady_state_bump_fractal_dist = 7

# %% 
init_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 6,
    "basis_y_start": 2,
    "basis_y_end": 7,

    "kl_decay": 1.25,
    "kl_scaling": 0.18,
}

# %% 
sim_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 7,
    "basis_y_start": 2,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.004,
}

# %% [markdown]
## Set-up statisitcs

# %% 
Ts = [0, 15*60, 3600, 6*3600, 12*3600]

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Nvar', type=int, default=2)
parser.add_argument('--Ne', type=int, default=100)
parser.add_argument('--Tda', type=float, default=6*3600)
parser.add_argument('--Tforecast', type=float, default=6*3600)
parser.add_argument('--initSteadyState', type=int, default=1, choices=[0,1])
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=60.0) 


pargs = parser.parse_args()

# Option was to test for pure noise
init_steady_state = bool(pargs.initSteadyState)
if init_steady_state:
    make_data_args = make_init_steady_state
else:
    make_data_args = make_init_fields

Ne = pargs.Ne

init_model_error = bool(pargs.init_error)
sim_model_error = bool(pargs.sim_error)
sim_model_error_timestep = pargs.sim_error_timestep

T_da = pargs.Tda
T_forecast = pargs.Tforecast

N_var = pargs.Nvar

# %% 
# Truth observation
Hxs = [ 500,  400,  600,  400,  600]
Hys = [1000,  900,  900, 1100, 1100]
# Hxs = [ 250,  200,  300,  200,  300]
# Hys = [ 500,  450,  450,  550,  550]
# Hxs = [ 125,  100,  150,  100,  150]
# Hys = [ 250,  225,  225,  275,  270]
R = [0.05, 1.0, 1.0]

# %% 
# Assimilation
localisation = False
r = 2.5e4
relax_factor = 0.1
obs_var = slice(1,3)

da_timestep = 900 
# print("SHORT DA STEP FOR TESTING")

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

grid_args = initGridSpecs(ls[-1])
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T = " + ", ".join([str(T) for T in Ts]) +"\n\n")

log.write("Init State\n")
if init_steady_state: 
    log.write("Double Bump\n")
    log.write("Bump size [m]: " + str(steady_state_bump_a) +"\n")
    log.write("Bump dist [fractal]: " + str(steady_state_bump_fractal_dist) + "\n\n")
else:
    log.write("Lake-at-rest\n\n")

log.write("Init Perturbation\n")
if init_model_error:
    log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")
else: 
    log.write("False\n\n")

log.write("Temporal Perturbation\n")
if sim_model_error:
    log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
    log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")
else:
    log.write("False\n\n")

log.write("Truth\n")
log.write("Hx, Hy: " + " / ".join([str(Hx) + ", " + str(Hy)   for Hx, Hy in zip(Hxs,Hys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
log.write("obs_var = slice(1,3)\n")
log.write("DA time steps: " + str(da_timestep) + "\n")
if localisation:
    log.write("r = " +str(r) + "\n")

log.write("\nStatistics\n")
log.write("N = " + str(N_var) + "\n")

log.close()

# %% 
def g_functional(SL_ensemble):
    """
    L_g functional as in notation of Kjetil's PhD thesis.
    This should be the functional that is under investigation for the variance level plot

    Returns a ndarray of same size as SL_ensemble (3, ny, nx, Ne)
    """
    return (SLdownload(SL_ensemble) - SLestimate(SL_ensemble, np.mean)[:,:,:,np.newaxis])**2


# %% 
varsNs = np.zeros((N_var, 3))
varsTs = [copy.deepcopy(varsNs) for T in Ts]
lvlvarsTs = [copy.deepcopy(varsTs) for l in ls]

difflvlvarsTs = copy.deepcopy(lvlvarsTs[:-1])

center_lvlvarsTs = copy.deepcopy(lvlvarsTs)
center_difflvlvarsTs = copy.deepcopy(lvlvarsTs[:-1])


# %% 
data_args_list = [make_data_args(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) for l_idx in range(len(ls))]

# %%
sim_args_list = []
for l_idx in range(len(ls)):
    sim_args = {
        "gpu_ctx" : args_list[l_idx]["gpu_ctx"],
        "nx" : args_list[l_idx]["nx"],
        "ny" : args_list[l_idx]["ny"],
        "dx" : args_list[l_idx]["dx"],
        "dy" : args_list[l_idx]["dy"],
        "f"  : sample_args["f"],
        "g"  : sample_args["g"],
        "r"  : 0,
        "dt" : 0,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
        "eta0" : data_args_list[l_idx]["eta"],
        "hu0"  : data_args_list[l_idx]["hu"],
        "hv0"  : data_args_list[l_idx]["hv"],
        "H"    : data_args_list[l_idx]["Hi"],
    }

    sim_args_list.append( sim_args )

# %%
#######################
# loop over levels
for l_idx in range(len(ls)):  
    print("Level ", l_idx)

    if init_model_error:
        init_mekl = ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args)
        coarse_init_mekl = ModelErrorKL.ModelErrorKL(**args_list[l_idx-1], **init_model_error_basis_args)

    if sim_model_error:
        sim_mekl  = ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args)
        coarse_sim_mekl  = ModelErrorKL.ModelErrorKL(**args_list[l_idx-1], **sim_model_error_basis_args)


    # Dummy Truth
    truth = CDKLM16.CDKLM16(**sim_args_list[l_idx])
    if init_model_error:
        init_mekl.perturbSim(truth)
    if sim_model_error:
        truth.model_error = sim_mekl
        truth.model_time_step = sim_model_error_timestep

    # Dummy Ensemble
    SL_ensemble = []
    coarse_SL_ensemble = []

    for e in range(Ne):
        sim = CDKLM16.CDKLM16(**sim_args_list[l_idx]) 
        if init_model_error:
            init_mekl.perturbSim(sim)
        if sim_model_error:
            sim.model_error = sim_mekl
            sim.model_time_step = sim_model_error_timestep
        SL_ensemble.append( sim )

        if l_idx > 0:
            coarse_sim = CDKLM16.CDKLM16(**sim_args_list[l_idx-1])
            if init_model_error:
                coarse_init_mekl.perturbSimSimilarAs(coarse_sim, modelError=init_mekl)
            if sim_model_error:
                coarse_sim.model_error = coarse_sim_mekl
                coarse_sim.model_time_step = sim_model_error_timestep
            coarse_SL_ensemble.append( coarse_sim )
        else:
            coarse_SL_ensemble.append( None )
    
    # Assimilation locations
    lvlHxs = [int(Hx/2**(len(ls)-1-l_idx)) for Hx in Hxs]
    lvlHys = [int(Hy/2**(len(ls)-1-l_idx)) for Hy in Hys]


    ##########################
    # loop over samples
    for n in range(N_var): 
        print(l_idx, n)

        # New Truth
        truth.upload(data_args_list[l_idx]["eta"], data_args_list[l_idx]["hu"], data_args_list[l_idx]["hv"])
        truth.t = 0.0
        if init_model_error:
            init_mekl.perturbSim(truth)

        # New Ensemble
        for e in range(Ne):
            SL_ensemble[e].upload(data_args_list[l_idx]["eta"], data_args_list[l_idx]["hu"], data_args_list[l_idx]["hv"])
            SL_ensemble[e].t = 0.0
            if init_model_error:
                init_mekl.perturbSim(SL_ensemble[e])

            if l_idx > 0:
                coarse_SL_ensemble[e].upload(data_args_list[l_idx-1]["eta"], data_args_list[l_idx-1]["hu"], data_args_list[l_idx-1]["hv"])
                coarse_SL_ensemble[e].t = 0.0
                if init_model_error:
                    coarse_init_mekl.perturbSimSimilarAs(coarse_SL_ensemble[e], modelError=init_mekl)


        # Weights
        localisation_weights_list = len(Hxs)*[None]
        if localisation:
            for h, [Hx, Hy] in enumerate(zip(Hxs, Hys)):
                localisation_weights_list[h] = GCweights(SL_ensemble, Hx, Hy, r) 

        coarse_localisation_weights_list = len(Hxs)*[None]
        if localisation:
            for h, [Hx, Hy] in enumerate(zip(Hxs, Hys)):
                coarse_localisation_weights_list[h] = GCweights(coarse_SL_ensemble, int(Hx/2), int(Hy/2), r) 


        ##########################
        # loop over time
        t_now = 0.0
        for t_idx, T in enumerate(Ts):

            numDAsteps = int((np.minimum(T, T_da + T_forecast)-t_now)/da_timestep)  

            for step in range(numDAsteps):
                # Forward step
                truth.dataAssimilationStep(t_now+da_timestep)

                for e in range(len(SL_ensemble)):
                    SL_ensemble[e].dataAssimilationStep(t_now+da_timestep, otherSim=coarse_SL_ensemble[e])
                
                t_now += da_timestep

                # Update step
                if step < numDAsteps-1 and truth.t <= T_da:
                    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
                    for h, [Hx, Hy] in enumerate(zip(lvlHxs, lvlHys)):
                        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                        SL_K, SL_perts = SLEnKF(SL_ensemble, obs, Hx, Hy, R=R, obs_var=obs_var, 
                                relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
                                return_perts=True)

                        if l_idx > 0:
                            # Update coarse ensemble
                            # print("Using fine Kalman gain for coarse update")
                            # coarse_SL_K = block_reduce(SL_K, block_size=(1,2,2,1), func=np.mean)

                            # coarse_SL_state = SLdownload(coarse_SL_ensemble)
                            # coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,int(Hy/2),int(Hx/2)] - SL_perts.T))
                            # SLupload(coarse_SL_ensemble, coarse_SL_state)

                            SL_K = SLEnKF(coarse_SL_ensemble, obs, int(Hx/2), int(Hy/2), R=R, obs_var=obs_var, 
                                relax_factor=relax_factor, localisation_weights=coarse_localisation_weights_list[h],
                                perts=SL_perts)


            print("Saving estimator variance at t=" + str(truth.t))
            lvlvarsTs[l_idx][t_idx][n] = np.mean(np.var(g_functional(SL_ensemble), axis=-1), axis=(1,2))

            center_N = int(args_list[l_idx]["nx"]/4)
            center_x = int(args_list[l_idx]["nx"]/2)
            center_y = int(args_list[l_idx]["ny"]/2)
            center_lvlvarsTs[l_idx][t_idx][n] = np.mean(np.var(g_functional(SL_ensemble)[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), axis=(1,2))
            
            if l_idx > 0:
                difflvlvarsTs[l_idx-1][t_idx][n] = np.mean(np.var(g_functional(SL_ensemble) - g_functional(coarse_SL_ensemble).repeat(2,1).repeat(2,2), axis=-1), axis=(1,2))
                center_difflvlvarsTs[l_idx-1][t_idx][n] = np.mean(np.var((g_functional(SL_ensemble) - g_functional(coarse_SL_ensemble).repeat(2,1).repeat(2,2))[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), axis=(1,2))

            if truth.t <= T_da:
                true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
                for h, [Hx, Hy] in enumerate(zip(lvlHxs, lvlHys)):
                    obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                    SL_K, SL_perts = SLEnKF(SL_ensemble, obs, Hx, Hy, R=R, obs_var=obs_var, 
                            relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
                            return_perts=True)

                    if l_idx > 0:
                        # Update coarse ensemble
                        coarse_SL_K = block_reduce(SL_K, block_size=(1,2,2,1), func=np.mean)

                        coarse_SL_state = SLdownload(coarse_SL_ensemble)
                        coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,int(Hy/2),int(Hx/2)] - SL_perts.T))
                        SLupload(coarse_SL_ensemble, coarse_SL_state)


# %% 
for t_idx, T in enumerate(Ts):
    varsT = [np.mean(lvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls))]
    np.save(output_path+"/vars_"+str(T), np.array(varsT))

    diff_varsT = [np.mean(difflvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls)-1)]
    np.save(output_path+"/diff_vars_"+str(T), np.array(diff_varsT))

    center_varsT = [np.mean(center_lvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls))]
    np.save(output_path+"/center_vars_"+str(T), np.array(center_varsT))

    center_diff_varsT = [np.mean(center_difflvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls)-1)]
    np.save(output_path+"/center_diff_vars_"+str(T), np.array(center_diff_varsT))

    