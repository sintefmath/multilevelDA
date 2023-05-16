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
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "DataAssimilation/Basin/"+timestamp 
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
parser.add_argument('--Tda', type=float, default=6*3600)
parser.add_argument('--Tforecast', type=float, default=6*3600)
parser.add_argument('--init_error', type=int, default=0,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=5*60) 

args = parser.parse_args()

T_da = args.Tda
T_forecast = args.Tforecast
init_model_error = bool(args.init_error)
sim_model_error = bool(args.sim_error)
sim_model_error_timestep = args.sim_error_timestep

# %% [markdown] 
# ## Ensemble

# %% 
read_path = "/home/florianb/havvarsel/multilevelDA/scripts/VarianceLevels/Basin/2023-05-05T14_03_12"

vars = np.load(read_path+"/vars_43200.npy")
diff_vars = np.load(read_path+"/diff_vars_43200.npy")

from utils.BasinAnalysis import *
analysis = Analysis(ls, vars, diff_vars, args_list)

ML_Nes = analysis.optimal_Ne(tau=5e-5)

# %% 
# Truth observation
Hxs = [ 500,  400,  600,  400,  600]
Hys = [1000,  900,  900, 1100, 1100]
R = [5e-4, 5e-2, 5e-2]

# %% 
# Assimilation
r = 5e4
relax_factor = 0.25
min_location_level = 0

da_timestep = 300

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("Nes = " + ", ".join([str(Ne) for Ne in ML_Nes])+"\n\n")

data_args = initGridSpecs(ls[-1])
log.write("nx = " + str(data_args["nx"]) + ", ny = " + str(data_args["ny"])+"\n")
log.write("dx = " + str(data_args["dx"]) + ", dy = " + str(data_args["dy"])+"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

log.write("Init State\n")
log.write("Double Bump\n\n")

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
log.write("Hx, Hy: " + " / ".join([str(Hx) + ", " + str(Hy)   for Hx, Hy in zip(Hxs,Hys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("r = " +str(r) + "\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
log.write("obs_var = slice(1,3)\n")
log.write("min_location_level = " + str(min_location_level) +"\n\n")
log.write("DA time steps: " + str(da_timestep) + "\n")

log.close()

# %% 
def write2file(T):
    true_state = truth.download(interior_domain_only=True)
    np.save(output_path+"/truth_"+str(T)+".npy", np.array(true_state))

    ML_state = MLOceanEnsemble.download()
    np.save(output_path+"/MLensemble_0_"+str(T)+".npy", np.array(ML_state[0]))
    for l_idx in enumerate(1,len(ls)):
        np.save(output_path+"/MLensemble_"+str(l_idx)+"_0_"+str(T)+".npy", np.array(ML_state[l_idx][0]))
        np.save(output_path+"/MLensemble_"+str(l_idx)+"_1_"+str(T)+".npy", np.array(ML_state[l_idx][1]))

# %% 
# initial fields
data_args_list = []
for l_idx in range(len(args_list)):
    data_args_list.append( make_init_steady_state(args_list[l_idx]) )

# %%
# Ensemble
from utils.BasinEnsembleInit import *
ML_ensemble = initMLensemble(ML_Nes, args_list, data_args_list, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=60.0)

from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)


# %%
from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

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
# DA period
while truth.t < T_da:
    # Forward step
    truth.dataAssimilationStep(truth.t+300)
    MLOceanEnsemble.stepToObservation(truth.t)

    # DA step
    print("DA at ", truth.t)
    for Hx, Hy in zip(Hxs, Hys):
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

        MLEnKF.assimilate(MLOceanEnsemble, obs, Hx, Hy, R, 
                            r=r, obs_var=slice(1,3), relax_factor=relax_factor, min_localisation_level=min_location_level)

# %% 
write2file(int(truth.t))

# %%
# Forecast period
truth.dataAssimilationStep(truth.t+T_forecast)
MLOceanEnsemble.stepToObservation(truth.t)


# %% 
write2file(int(truth.t))