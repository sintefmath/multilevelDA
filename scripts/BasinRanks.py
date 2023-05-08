# %% [markdown]
# # Multi Level Statistics

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

output_path = "RankHistograms/Basin/"+timestamp 
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
init_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 6,
    "basis_y_start": 2,
    "basis_y_end": 7,

    "kl_decay": 1.25,
    "kl_scaling": 0.05,
}

# %% 
sim_model_error_basis_args = {
    "basis_x_start": 2, 
    "basis_x_end": 7,
    "basis_y_start": 3,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.005,
}

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=5*60) 


args = parser.parse_args()

init_model_error = bool(args.init_error)
sim_model_error = bool(args.sim_error)
sim_model_error_timestep = args.sim_error_timestep

N_ranks = args.N

# %% [markdown] 
# ## Ensemble

# %% 
read_path = "/home/florianb/havvarsel/multilevelDA/scripts/VarianceLevels/Basin/2023-05-05T14_03_12"

vars = np.load(read_path+"/vars_43200.npy")
diff_vars = np.load(read_path+"/diff_vars_43200.npy")

from utils.BasinAnalysis import *
analysis = Analysis(ls, vars, diff_vars, args_list)

ML_Nes = analysis.optimal_Ne(tau=1.25e-4)

# %% 
# Truth observation
Hx, Hy = 500, 1000
R = [5e-5, 5e-3, 5e-3]

# %% 
# Assimilation
r = 5e4
relax_factor = 0.5
min_location_level = 0

da_timestep = 300

# %% 
# Simulation
Ts = [0, 15*60, 30*60, 3600, 2*3600]

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("Nes = " + ", ".join([str(Ne) for Ne in ML_Nes])+"\n\n")

data_args = initGridSpecs(ls[-1])
log.write("nx = " + str(data_args["nx"]) + ", ny = " + str(data_args["ny"])+"\n")
log.write("dx = " + str(data_args["dx"]) + ", dy = " + str(data_args["dy"])+"\n")
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
log.write("r = " +str(r) + "\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
log.write("obs_var = slice(1,3)\n")
log.write("min_location_level = " + str(min_location_level) +"\n\n")
log.write("DA time steps: " + str(da_timestep) + "\n")

log.write("Statistics\n")
log.write("N = " + str(N_ranks) + "\n")

log.close()


# %% 
data_args_list = []
for l_idx in range(len(args_list)):
    data_args_list.append( make_init_steady_state(args_list[l_idx]) )

# %% 
# Truth
truth = make_sim(args_list[-1], sample_args=sample_args, init_fields=data_args_list[-1])
if init_model_error:
    init_mekl = ModelErrorKL.ModelErrorKL(**args_list[-1], **init_model_error_basis_args)
    init_mekl.perturbSim(truth)
if sim_model_error:
    truth.setKLModelError(**sim_model_error_basis_args)
    truth.model_time_step = 60.0

# %% 
# Initialise ensemble
# Since we aim to re-use all objects, we do NOT use the `BasinEnsembleInit.py`

# Model errors
if init_model_error_basis_args: 
    init_mekls = []
    for l_idx in range(len(args_list)): 
        init_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args) )

if sim_model_error_basis_args: 
    sim_mekls = []
    for l_idx in range(len(args_list)): 
        sim_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args) )

sim_model_error_time_step=60.0

 ## MultiLevel ensemble
ML_ensemble = []

# 0-level
lvl_ensemble = []
for i in range(ML_Nes[0]):
    if i % 100 == 0: print(i)
    sim = make_sim(args_list[0], sample_args, init_fields=data_args_list[0])
    if init_model_error_basis_args:
        init_mekls[0].perturbSim(sim)
    if sim_model_error_basis_args:
        # sim.setKLModelError(**sim_model_error_basis_args)
        sim.model_error = sim_mekls[0]
    sim.model_time_step = sim_model_error_time_step
    lvl_ensemble.append( sim )

ML_ensemble.append(lvl_ensemble)

# diff-levels
for l_idx in range(1,len(ML_Nes)):
    print(l_idx)
    lvl_ensemble0 = []
    lvl_ensemble1 = []
    
    for e in range(ML_Nes[l_idx]):
        sim0 = make_sim(args_list[l_idx], sample_args, init_fields=data_args_list[l_idx])
        sim1 = make_sim(args_list[l_idx-1], sample_args, init_fields=data_args_list[l_idx-1])
        
        if init_model_error_basis_args:
            init_mekls[l_idx].perturbSim(sim0)
            init_mekls[l_idx-1].perturbSimSimilarAs(sim1, modelError=init_mekls[l_idx])

        if sim_model_error_basis_args:
            sim0.model_error = sim_mekls[l_idx]
            sim1.model_error = sim_mekls[l_idx-1]

        sim0.model_time_step = sim_model_error_time_step
        sim1.model_time_step = sim_model_error_time_step

        lvl_ensemble0.append(sim0)
        lvl_ensemble1.append(sim1)
    
    ML_ensemble.append([lvl_ensemble0,lvl_ensemble1])

#%% 
from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)

print("Prior saved!")
ML_prior = copy.deepcopy(MLOceanEnsemble.download())

# %%
from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

# %%
freq = 50
Hxs = np.arange( 512, 1024, freq)
Hys = np.arange(1024, 2048, 2*freq)

ML_ranks = np.zeros((len(Hxs)*N_ranks,3))

ML_prior_ranksTs = [copy.deepcopy(ML_ranks) for T in Ts]
ML_posterior_ranksTs = [copy.deepcopy(ML_ranks) for T in Ts]

# %% 
n = 0
while n < N_ranks:
    print("\n\nExperiment: ", n)

    try:
        signal.alarm(10*60)

        # New Truth
        print("Make a new truth")
        truth.upload(data_args_list[-1]["eta"], data_args_list[-1]["hu"], data_args_list[-1]["hv"])
        truth.t = 0.0
        if init_model_error:
            init_mekl.perturbSim(truth)

        # New Ensemble
        # 0-level
        for e in range(ML_Nes[0]):
            MLOceanEnsemble.ML_ensemble[0][e].upload(data_args_list[0]["eta"], data_args_list[0]["hu"], data_args_list[0]["hv"])
            MLOceanEnsemble.ML_ensemble[0][e].t = 0.0
            if init_model_error_basis_args:
                init_mekls[0].perturbSim(MLOceanEnsemble.ML_ensemble[0][e])

        # diff-levels
        for l_idx in range(1,len(ML_Nes)):
            for e in range(ML_Nes[l_idx]):
                MLOceanEnsemble.ML_ensemble[l_idx][0][e].upload(data_args_list[l_idx]["eta"], data_args_list[l_idx]["hu"], data_args_list[l_idx]["hv"])
                MLOceanEnsemble.ML_ensemble[l_idx][0][e].t = 0.0
                
                MLOceanEnsemble.ML_ensemble[l_idx][1][e].upload(data_args_list[l_idx-1]["eta"], data_args_list[l_idx-1]["hu"], data_args_list[l_idx-1]["hv"])
                MLOceanEnsemble.ML_ensemble[l_idx][1][e].t = 0.0

                if init_model_error_basis_args:
                    init_mekls[l_idx].perturbSim(MLOceanEnsemble.ML_ensemble[l_idx][0][e])
                    init_mekls[l_idx-1].perturbSimSimilarAs(MLOceanEnsemble.ML_ensemble[l_idx][1][e], modelError=init_mekls[l_idx])


        print("Lets start to move")
        t_now = 0.0
        for t_idx, T in enumerate(Ts):

            numDAsteps = int((T-t_now)/da_timestep)  

            for step in range(numDAsteps):
                truth.dataAssimilationStep(t_now+300)
                MLOceanEnsemble.stepToObservation(t_now+300)
                t_now += 300

                if step < numDAsteps-1:
                    print("non-recorded DA")
                    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
                    obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                    MLEnKF.assimilate(MLOceanEnsemble, obs, Hx, Hy, R, 
                                        r=r, obs_var=slice(1,3), relax_factor=relax_factor, min_localisation_level=min_location_level)

            print("recorded DA")
            ML_prior_ranksTs[t_idx][n*len(Hxs):(n+1)*len(Hxs)] = MLOceanEnsemble.rank(truth, [z for z in zip(Hxs, Hys)])

            true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

            MLEnKF.assimilate(MLOceanEnsemble, obs, Hx, Hy, R, 
                                        r=r, obs_var=slice(1,3), relax_factor=relax_factor, min_localisation_level=min_location_level)

            ML_posterior_ranksTs[t_idx][n*len(Hxs):(n+1)*len(Hxs)] = MLOceanEnsemble.rank(truth, [z for z in zip(Hxs, Hys)])

            print(T)

        for t_idx, T in enumerate(Ts):
            np.save(output_path+"/MLpriorRanks_"+str(T)+"_dump_"+str(n)+".npy", ML_prior_ranksTs[t_idx][n*len(Hxs):(n+1)*len(Hxs)])
            np.save(output_path+"/MLposteriorRanks_"+str(T)+"_dump_"+str(n)+".npy", ML_posterior_ranksTs[t_idx][n*len(Hxs):(n+1)*len(Hxs)])

        n = n+1
        signal.alarm(0)

    except Exception as exc:
        print(exc)
        signal.alarm(0)
        pass

# %% 
for t_idx, T in enumerate(Ts):
    np.save(output_path+"/MLpriorRanks_"+str(T)+".npy", ML_prior_ranksTs)
    np.save(output_path+"/MLposteriorRanks_"+str(T)+".npy", ML_posterior_ranksTs)
