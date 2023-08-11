# %% [markdown]
# # Error convergence

# %% [markdown]
# ### Classes and modules

# %%
import os, sys

#Import packages we need
import numpy as np
import datetime
import copy

#For plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"

import pycuda.driver as cuda

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "ErrorConvergence/Basin/"+timestamp 
os.makedirs(output_path)

# %%
import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for Error Convergence\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")


# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL
from gpuocean.ensembles import MultiLevelOceanEnsemble

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *
from utils.BasinAnalysis import *
from utils.BasinSL import *
from utils.BasinEnsembleInit import *

# %%
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

# %% [markdown]
# ### Multi-Level Ensemble Sizes 

# %%
source_path = "/home/florianb/havvarsel/multilevelDA/scripts/VarianceLevelsDA/Basin/2023-06-28T11_25_15sharedOmegaLocalised"

center_vars = np.load(source_path+"/center_vars_21600_L2norm.npy")
center_diff_vars = np.load(source_path+"/center_diff_vars_21600_L2norm.npy")

# %%
work_path = "/home/florianb/havvarsel/multilevelDA/scripts/PracticalCost/Basin/2023-06-29T14_54_02"

# %%
def raw2costsEnsemble(filename):
    rawCosts = np.load(filename)
    return np.mean(np.sort(rawCosts), axis=1)

# %%
costsPure = raw2costsEnsemble(work_path+"/costsPureEnsemble.npy")
costsPartnered = raw2costsEnsemble(work_path+"/costsPartneredEnsemble.npy")

# %%
analysis = Analysis(args_list, center_vars, center_diff_vars, costsPure, costsPartnered)

# %%
taus = [4.0e-2, 3.0e-2, 2.0e-2]#, 1.5e-2, 1.25e-2]

Nes = []

for tau in taus:
    ML_Nes = analysis.optimal_Ne(tau=tau)
    SL_Ne = np.ceil(analysis.work(ML_Nes)/costsPure[-1]).astype(int)

    Nes.append([ML_Nes, SL_Ne])

# %%
from utils.BasinParameters import * 

# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("ML_Nes = " + ", ".join([str(Ne[0]) for Ne in Nes])+"\n")
log.write("ML_Nes = " + ", ".join([str(Ne[1]) for Ne in Nes])+"\n\n")

log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

log.write("Init State\n")
log.write("Double Bump\n")
log.write("Bump size [m]: " + str(steady_state_bump_a) +"\n")
log.write("Bump dist [fractal]: " + str(steady_state_bump_fractal_dist) + "\n\n")

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

log.write("Statistics\n")
N_exp = 2
log.write("N_exp = " + str(N_exp) + "\n")
log.write("Metric: RMSE\n")


# %% 
# initial fields
data_args_list = []
for l_idx in range(len(args_list)):
    data_args_list.append( make_init_steady_state(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) )


# %% 
def writeTruth2file(T):
    true_state = truth.download(interior_domain_only=True)
    os.makedirs(output_path+"/tmpTruth", exist_ok=True)
    np.save(output_path+"/tmpTruth/truth_"+str(T)+".npy", np.array(true_state))
    

# %% 
# Model errors
if init_model_error_basis_args: 
    init_mekls = []
    for l_idx in range(len(args_list)): 
        init_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args) )

if sim_model_error_basis_args: 
    sim_mekls = []
    for l_idx in range(len(args_list)): 
        sim_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args) )

# %%
###########################################
###########################################
# tau-LOOP 

########################################
# Truth
truth = make_sim(args_list[-1], sample_args=sample_args, init_fields=data_args_list[-1])
init_mekls[-1].perturbSim(truth)
truth.model_error = sim_mekls[-1]
truth.model_time_step = sim_model_error_timestep

ML_Nes = Nes[0][0]
SL_Ne  = Nes[0][1]

########################################
## Single-Level ensemble

sim_args = {
    "gpu_ctx" : args_list[-1]["gpu_ctx"],
    "nx" : args_list[-1]["nx"],
    "ny" : args_list[-1]["ny"],
    "dx" : args_list[-1]["dx"],
    "dy" : args_list[-1]["dy"],
    "f"  : sample_args["f"],
    "g"  : sample_args["g"],
    "r"  : 0,
    "dt" : 0,
    "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
    "eta0" : data_args_list[-1]["eta"],
    "hu0"  : data_args_list[-1]["hu"],
    "hv0"  : data_args_list[-1]["hv"],
    "H"    : data_args_list[-1]["Hi"],
}

SL_ensemble = []

for e in range(SL_Ne):
    sim = CDKLM16.CDKLM16(**sim_args) 
    init_mekls[-1].perturbSim(sim)
    sim.model_error = sim_mekls[-1]
    sim.model_time_step = sim_model_error_timestep
    SL_ensemble.append( sim )

########################################
## Multi-Level ensemble 
ML_ensemble = []

# 0-level
lvl_ensemble = []
for i in range(ML_Nes[0]):
    sim = make_sim(args_list[0], sample_args, init_fields=data_args_list[0])
    init_mekls[0].perturbSim(sim)
    sim.model_error = sim_mekls[0]
    sim.model_time_step = sim_model_error_timestep
    lvl_ensemble.append( sim )

ML_ensemble.append(lvl_ensemble)

# diff-levels
for l_idx in range(1,len(ML_Nes)):
    lvl_ensemble0 = []
    lvl_ensemble1 = []
    
    for e in range(ML_Nes[l_idx]):
        sim0 = make_sim(args_list[l_idx], sample_args, init_fields=data_args_list[l_idx])
        sim1 = make_sim(args_list[l_idx-1], sample_args, init_fields=data_args_list[l_idx-1])
        
        init_mekls[l_idx].perturbSim(sim0)
        init_mekls[l_idx-1].perturbSimSimilarAs(sim1, modelError=init_mekls[l_idx])

        sim0.model_error = sim_mekls[l_idx]
        sim1.model_error = sim_mekls[l_idx-1]

        sim0.model_time_step = sim_model_error_timestep
        sim1.model_time_step = sim_model_error_timestep

        lvl_ensemble0.append(sim0)
        lvl_ensemble1.append(sim1)
    
    ML_ensemble.append([lvl_ensemble0,lvl_ensemble1])

MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)


for tau_idx in range(len(taus)): 
    print("-----------------------------")
    print("tau = ", taus[tau_idx])

    ML_Nes = Nes[tau_idx][0]
    SL_Ne  = Nes[tau_idx][1]

    if tau_idx > 0: 
        
        # Single level ensemble
        for e in range(SL_Ne - Nes[tau_idx-1][1]):
            sim = CDKLM16.CDKLM16(**sim_args) 
            init_mekls[-1].perturbSim(sim)
            sim.model_error = sim_mekls[-1]
            sim.model_time_step = sim_model_error_timestep
            SL_ensemble.append( sim )

        # Multi level ensemble



    


        MLOceanEnsemble.Nes = ML_Nes





    ###########################################
    ###########################################
    # N-LOOP 
    for n in range(N_exp):


        ########################################
        # NEW truth
        truth.upload(data_args_list[-1]["eta"], data_args_list[-1]["hu"], data_args_list[-1]["hv"])
        truth.t = 0.0
        init_mekls[-1].perturbSim(truth)

        # writeTruth2file(int(truth.t))
        # while truth.t < T_da:
        #     # Forward step
        #     truth.dataAssimilationStep(truth.t+300)
        #     # DA step
        #     writeTruth2file(int(truth.t))


        ########################################
        # NEW SL Ensemble
        
        for e in range(SL_Ne):
            SL_ensemble[e].upload(data_args_list[-1]["eta"], data_args_list[-1]["hu"], data_args_list[-1]["hv"])
            SL_ensemble[e].t = 0.0
            init_mekls[-1].perturbSim(SL_ensemble[e])









        ########################################
        # NEW ML Ensemble

        # 0-level
        for e in range(ML_Nes[0]):
            MLOceanEnsemble.ML_ensemble[0][e].upload(data_args_list[0]["eta"], data_args_list[0]["hu"], data_args_list[0]["hv"])
            MLOceanEnsemble.ML_ensemble[0][e].t = 0.0
            init_mekls[0].perturbSim(MLOceanEnsemble.ML_ensemble[0][e])

        # diff-levels
        for l_idx in range(1,len(ML_Nes)):
            for e in range(ML_Nes[l_idx]):
                MLOceanEnsemble.ML_ensemble[l_idx][0][e].upload(data_args_list[l_idx]["eta"], data_args_list[l_idx]["hu"], data_args_list[l_idx]["hv"])
                MLOceanEnsemble.ML_ensemble[l_idx][0][e].t = 0.0
                
                MLOceanEnsemble.ML_ensemble[l_idx][1][e].upload(data_args_list[l_idx-1]["eta"], data_args_list[l_idx-1]["hu"], data_args_list[l_idx-1]["hv"])
                MLOceanEnsemble.ML_ensemble[l_idx][1][e].t = 0.0

                init_mekls[l_idx].perturbSim(MLOceanEnsemble.ML_ensemble[l_idx][0][e])
                init_mekls[l_idx-1].perturbSimSimilarAs(MLOceanEnsemble.ML_ensemble[l_idx][1][e], modelError=init_mekls[l_idx])









        # Cleaning Up 
        if os.path.isdir(output_path+"/tmpTruth"):
            for f in os.listdir(output_path+"/tmpTruth"):
                os.remove(os.path.join(output_path+"/tmpTruth", f))

        # end: n-loop 

    # end: tau-loop

if os.path.isdir(output_path+"/tmpTruth"):
    os.rmdir(output_path+"/tmpTruth")
