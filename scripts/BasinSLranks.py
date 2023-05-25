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
L = 9

# %% 
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }


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
    "basis_x_start": 2, 
    "basis_x_end": 7,
    "basis_y_start": 3,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.004,
}

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=60) 


pargs = parser.parse_args()

init_model_error = bool(pargs.init_error)
sim_model_error = bool(pargs.sim_error)
sim_model_error_timestep = pargs.sim_error_timestep

# init_model_error = True
# sim_model_error = True
# sim_model_error_timestep = 60.0

N_ranks = pargs.N

# %% [markdown] 
# ## Ensemble

# %% 
Ne = 100

# %% 
# Truth observation
Hx, Hy = 250, 500
R = [0.1, 1.0, 1.0]

# %% 
# Assimilation
localisation = False
r = 2.5e4
relax_factor = 0.1

da_timestep = 900

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
if init_model_error:
    log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")
else: 
    init_model_error_basis_args = None
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
    sim_model_error_basis_args = None
    log.write("False\n\n")

log.write("Truth\n")
log.write("Hx, Hy: " + str(Hx) + ", " + str(Hy) + "\n")
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

data_args = make_init_steady_state(args)

# %% 
# Truth
truth = make_sim(args, sample_args=sample_args, init_fields=data_args)
if init_model_error:
    init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)
    init_mekl.perturbSim(truth)
if sim_model_error:
    truth.setKLModelError(**sim_model_error_basis_args)
    truth.model_time_step = sim_model_error_timestep

# %% 
# Initialise ensemble
# Since we aim to re-use all objects, we do NOT use the `BasinEnsembleInit.py`

# Model errors
if init_model_error: 
    init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args) 

if sim_model_error: 
    sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)


# Ensemble
SL_ensemble = initSLensemble(Ne, args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)


localisation_weights = None
if localisation:
    localisation_weights = GCweights(SL_ensemble, Hx, Hy, r) 

# %%
freq = 50
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
        signal.alarm(5*60)

        # New Truth
        print("Make a new truth")
        truth.upload(data_args["eta"], data_args["hu"], data_args["hv"])
        truth.t = 0.0
        if init_model_error:
            init_mekl.perturbSim(truth)

        # New Ensemble
        # 0-level
        for e in range(Ne):
            SL_ensemble[e].upload(data_args["eta"], data_args["hu"], data_args["hv"])
            SL_ensemble[e].t = 0.0
            if init_model_error:
                init_mekl.perturbSim(SL_ensemble[e])

        print("Lets start to move")
        t_now = 0.0
        for t_idx, T in enumerate(Ts):

            numDAsteps = int((T-t_now)/da_timestep)  

            for step in range(numDAsteps):
                truth.dataAssimilationStep(t_now+da_timestep)
                SLstepToObservation(SL_ensemble, t_now+da_timestep)
                t_now += da_timestep

                if step < numDAsteps-1:
                    print("non-recorded DA")
                    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
                    obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                    SLEnKF(SL_ensemble, obs, Hx, Hy, R=R, obs_var=slice(1,3), 
                            relax_factor=relax_factor, localisation_weights=localisation_weights)

            print("recorded DA")
            SL_prior_ranksTs[t_idx][n*len(rankHxs):(n+1)*len(rankHxs)] = SLrank(SL_ensemble, truth, [z for z in zip(rankHxs, rankHys)], R)

            true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

            SLEnKF(SL_ensemble, obs, Hx, Hy, R=R, obs_var=slice(1,3), 
                    relax_factor=relax_factor, localisation_weights=localisation_weights)

            SL_posterior_ranksTs[t_idx][n*len(rankHxs):(n+1)*len(rankHxs)] = SLrank(SL_ensemble, truth, [z for z in zip(rankHxs, rankHys)], R)

            print(T)

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
