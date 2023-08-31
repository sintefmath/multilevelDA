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

output_path = "DataAssimilation/DoubleJetTruth/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")

import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.DoubleJetPlot import *

# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
L = 9

# %% 
from utils.DoubleJetParametersReplication import *

# %%
from gpuocean.utils import DoubleJetCase

doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=2**L, nx=2**(L+1))
doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()

# %% 
args = {key: doubleJetCase_args[key] for key in ('nx', 'ny', 'dx', 'dy', 'gpu_ctx', 'boundary_conditions')}
args["gpu_stream"] = gpu_stream

data_args = {"eta" : doubleJetCase_init["eta0"],
             "hu" : doubleJetCase_init["hu0"],
             "hv" : doubleJetCase_init["hv0"],
             "Hi" : doubleJetCase_args["H"]}

sample_args = {"f": doubleJetCase_args["f"], "g": doubleJetCase_args["g"]}

# %%
# Book keeping
log.write("L = " + str(L) + "\n")


log.write("nx = " + str(args["nx"]) + ", ny = " + str(args["ny"])+"\n")
log.write("dx = " + str(args["dx"]) + ", dy = " + str(args["dy"])+"\n")
log.write("T (spinup) = " + str(T_spinup) +"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

log.write("Temporal Perturbation\n")
log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")

log.close()

# %% 
def write2file(T):
    print("Saving at time ", T)
    true_state = truth.download(interior_domain_only=True)
    np.save(output_path+"/truth_"+str(T)+".npy", np.array(true_state))
    

# %% 
# Truth
sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)

sim_args = {
    "gpu_ctx" : args["gpu_ctx"],
    "nx" : args["nx"],
    "ny" : args["ny"],
    "dx" : args["dx"],
    "dy" : args["dy"],
    "f"  : sample_args["f"],
    "g"  : sample_args["g"],
    "r"  : 0,
    "dt" : 0,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
    "eta0" : data_args["eta"],
    "hu0"  : data_args["hu"],
    "hv0"  : data_args["hv"],
    "H"    : data_args["Hi"],
}

truth = CDKLM16.CDKLM16(**sim_args) 
truth.model_error = sim_mekl
truth.model_time_step = sim_model_error_timestep

# %% 
# Spin-up
truth.dataAssimilationStep(T_spinup)
write2file(int(truth.t))

# %% 
# DA period
while truth.t < T_spinup + T_da:
    # Forward step
    truth.dataAssimilationStep(truth.t+da_timestep)
    write2file(int(truth.t))

    imshowSim(truth)
    plt.savefig(output_path+"/truth_"+str(int(truth.t))+".pdf", bbox_inches="tight")
    plt.close("all")


sys.exit(0)
# %%
# Prepare drifters
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.utils import Observation
from gpuocean.dataassimilation import DataAssimilationUtils as dautils
observation_args = {'observation_type': dautils.ObservationType.UnderlyingFlow,
                'nx': grid_args["nx"], 'ny': grid_args["ny"],
                'domain_size_x': grid_args["nx"]*grid_args["dx"],
                'domain_size_y': grid_args["ny"]*grid_args["dy"],
               }

num_drifters = len(init_positions)


forecast = Observation.Observation(**observation_args)
drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                        boundaryConditions = args["boundary_conditions"],
                                        domain_size_x = forecast.domain_size_x,
                                        domain_size_y = forecast.domain_size_y)
drifters.setDrifterPositions(init_positions)
truth.attachDrifters(drifters)
forecast.add_observation_from_sim(truth)

# %% 
while truth.t < T_da + T_forecast:
    # Forward step
    truth.dataAssimilationStep(truth.t+300)
    forecast.add_observation_from_sim(truth)

    # DA step
    write2file(int(truth.t))

    imshowSim(truth)
    plt.savefig(output_path+"/truth_"+str(int(truth.t))+".pdf", bbox_inches="tight")
    plt.close("all")

# %% 
forecast.to_pickle(output_path+"/truth_trajectories.pickle")