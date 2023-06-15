# %% [markdown]
# # Single Level Statistics

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

# import time
# print("Gonna sleep now!")
# time.sleep(3*3600) # Sleep for 3 seconds

# %% 
import signal

def handler(signum, frame):
    raise Exception("Time Out: Experiment aborted!")

signal.signal(signal.SIGALRM, handler)


# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "RankHistograms/BasinSL/"+timestamp 
os.makedirs(output_path)
os.makedirs(output_path+"/dump")

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
L = 10

# %% 
from utils.BasinParameters import *

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--N', type=int, default=100)

pargs = parser.parse_args()

N_ranks = pargs.N

# %%
# ## Ensemble
Ne = 100

# %% 
# Assimilation
localisation = True

# %% 
# Simulation
Ts = [0, 15*60, 30*60, 3600, 2*3600, 3*3600]

# %%
# Book keeping
log.write("L = " + str(L) + "\n\n")
log.write("Ne = " + str(Ne) + "\n\n")

grid_args = initGridSpecs(L)
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T = " + ", ".join([str(T) for T in Ts]) +"\n\n")

log.write("Init State\n")
log.write("Lake-at-rest\n\n")

log.write("Init Perturbation\n")
log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")

log.write("Temporal Perturbation\n")
log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")

log.write("Truth\n")
log.write("obs_x, obs_y: " + " / ".join([str(obs_x) + ", " + str(obs_y)   for obs_x, obs_y in zip(obs_xs,obs_ys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("DA time steps: " + str(da_timestep) + "\n")
log.write("obs_var = slice(1,3)\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
if localisation:
    log.write("r = " +str(r) + "\n")

log.write("\nStatistics\n")
log.write("N = " + str(N_ranks) + "\n")

log.close()


# %%
args = {
    "nx": grid_args["nx"],
    "ny": grid_args["ny"],
    "dx": grid_args["dx"],
    "dy": grid_args["dy"],
    "gpu_ctx": gpu_ctx,
    "gpu_stream": gpu_stream,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2)
    }

data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)

# %% 
# Truth
truth = make_sim(args, sample_args=sample_args, init_fields=data_args)
truth_init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)
truth_init_mekl.perturbSim(truth)
truth.setKLModelError(**sim_model_error_basis_args)
truth.model_time_step = sim_model_error_timestep

# %% 
# Initialise ensemble
# Since we aim to re-use all objects, we do NOT use the `BasinEnsembleInit.py`

# Model errors
init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args) 
sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)


# Ensemble
SL_ensemble = initSLensemble(Ne, args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)


localisation_weights = None
if localisation:
    localisation_weights_list = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        localisation_weights_list.append( GCweights(SL_ensemble, obs_x, obs_y, r) ) 

# %%
freq = int(np.floor(grid_args["nx"]/20))
rankHxs = np.arange(0, grid_args["nx"], freq)
rankHys = np.arange(0, grid_args["ny"], 2*freq)

SL_ranks = np.zeros((len(rankHxs)*N_ranks,3))

SL_prior_ranksTs = [copy.deepcopy(SL_ranks) for T in Ts]
SL_posterior_ranksTs = [copy.deepcopy(SL_ranks) for T in Ts]

# %% 
n = 0
while n < N_ranks:
    print("\n\nExperiment: ", n)

    try:
        signal.alarm(60*60)

        # New Truth
        print("Make a new truth")
        truth.upload(data_args["eta"], data_args["hu"], data_args["hv"])
        truth.t = 0.0
        truth_init_mekl.perturbSim(truth)

        # New Ensemble
        # 0-level
        for e in range(Ne):
            SL_ensemble[e].upload(data_args["eta"], data_args["hu"], data_args["hv"])
            SL_ensemble[e].t = 0.0
            init_mekl.perturbSim(SL_ensemble[e])

        print("Lets start to move")
        t_now = 0.0
        for t_idx, T in enumerate(Ts):

            numDAsteps = int((T-t_now)/da_timestep)  

            for step in range(numDAsteps):
                # Forward step
                truth.dataAssimilationStep(t_now+da_timestep)
                SLstepToObservation(SL_ensemble, t_now+da_timestep)
                t_now += da_timestep

                # DA step
                if step < numDAsteps-1:
                    print("non-recorded DA at " + str(truth.t))
                    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
                    for o, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
                        Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y)
                        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                        SLEnKF(SL_ensemble, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
                            relax_factor=relax_factor, localisation_weights=localisation_weights_list[o])

            print("recorded DA at " + str(truth.t))
            SL_prior_ranksTs[t_idx][n*len(rankHxs):(n+1)*len(rankHxs)] = SLrank(SL_ensemble, truth, [z for z in zip(rankHxs, rankHys)], R)

            true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
            for o, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
                Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y)
                obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                SLEnKF(SL_ensemble, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
                    relax_factor=relax_factor, localisation_weights=localisation_weights_list[o])

            SL_posterior_ranksTs[t_idx][n*len(rankHxs):(n+1)*len(rankHxs)] = SLrank(SL_ensemble, truth, [z for z in zip(rankHxs, rankHys)], R)

        for t_idx, T in enumerate(Ts):
            np.save(output_path+"/dump/SLpriorRanks_"+str(T)+"_dump_"+str(n)+".npy", SL_prior_ranksTs[t_idx][n*len(rankHxs):(n+1)*len(rankHxs)])
            np.save(output_path+"/dump/SLposteriorRanks_"+str(T)+"_dump_"+str(n)+".npy", SL_posterior_ranksTs[t_idx][n*len(rankHxs):(n+1)*len(rankHxs)])

        n = n+1
        signal.alarm(0)

    except Exception as exc:
        print(exc)
        signal.alarm(0)
        pass

# %% 
for t_idx, T in enumerate(Ts):
    np.save(output_path+"/SLpriorRanks_"+str(T)+".npy", SL_prior_ranksTs[t_idx])
    np.save(output_path+"/SLposteriorRanks_"+str(T)+".npy", SL_posterior_ranksTs[t_idx])
