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
taus = [4.0e-2, 3.0e-2, 2.0e-2, 1.5e-2]#, 1.25e-2]

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
N_exp = 10
log.write("N_exp = " + str(N_exp) + "\n")
log.write("Metric: RMSE\n")




# %%
###########################################
###########################################
# tau-LOOP

rmse_SL = np.zeros((len(taus),3))
rmse_ML = np.zeros((len(taus),3))

center_N = int(args_list[-1]["nx"]/4)
center_x = int(args_list[-1]["nx"]/2)
center_y = int(args_list[-1]["ny"]/2)

for tau_idx in range(len(taus)): 
    print("-----------------------------")
    print("tau = ", taus[tau_idx])


    ###########################################
    ###########################################
    # N-LOOP 
    for n in range(N_exp):

        ########################################
        # NEW truth
        print("Truth")
        os.system("python BasinErrConvSingle.py -m T")


        ########################################
        # NEW SL Ensemble
        print("SL", Nes[tau_idx][1])
        os.system("python BasinErrConvSingle.py -m SL -Ne "+str(Nes[tau_idx][1]))

        if os.path.isdir("tmpSL"):
            assert len(os.listdir("tmpSL")) == 1, "Please remove old files"
            f = os.listdir("tmpSL")[0]
            err_eta, err_hu, err_hv = np.load(os.path.join("tmpSL", f))
            rmse_SL[tau_idx] = 1/N_exp * np.array([np.sqrt(np.sum(err[center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N]**2)) for err in [err_eta, err_hu, err_hv]])
            os.remove(os.path.join("tmpSL", f))


        ########################################
        # NEW ML Ensemble
        print("ML", Nes[tau_idx][0])
        os.system("python BasinErrConvSingle.py -m ML -Ne "+" ".join(str(Ne) for Ne in Nes[tau_idx][0]))

        if os.path.isdir("tmpML"):
            assert len(os.listdir("tmpML")) == 1, "Please remove old files"
            f = os.listdir("tmpML")[0]
            err_eta, err_hu, err_hv = np.load(os.path.join("tmpML", f))
            rmse_ML[tau_idx] = 1/N_exp * np.array([np.sqrt(np.sum(err[center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N]**2)) for err in [err_eta, err_hu, err_hv]])
            os.remove(os.path.join("tmpML", f))


        # Cleaning Up 
        if os.path.isdir("tmpTruth"):
            for f in os.listdir("tmpTruth"):
                os.remove(os.path.join("tmpTruth", f))

        # end: n-loop 

    # end: tau-loop

if os.path.isdir("tmpTruth"):
    os.rmdir("tmpTruth")


np.savetxt(output_path+"/rmseSL.txt", rmse_SL)
np.savetxt(output_path+"/rmseML.txt", rmse_ML)

if os.path.isdir("tmpSL"):
    os.rmdir("tmpSL")

if os.path.isdir("tmpML"):
    os.rmdir("tmpML")
