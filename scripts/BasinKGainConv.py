# %% [markdown]
# # Kalman Gain convergence

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

output_path = "KGainConvergence/Basin/"+timestamp 
os.makedirs(output_path)

# %%
import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for Kalman Gain Convergence\n\n")

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
# The last level is only used for the reference solution

num_levels = 3

start_l_idx = len(ls)-1-num_levels

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

center_vars = np.load(source_path+"/center_vars_21600_L2norm.npy")[start_l_idx:-1]
center_diff_vars = np.load(source_path+"/center_diff_vars_21600_L2norm.npy")[start_l_idx:-1]

# %%
work_path = "/home/florianb/havvarsel/multilevelDA/scripts/PracticalCost/Basin/2023-06-29T14_54_02"

# %%
def raw2costsEnsemble(filename):
    rawCosts = np.load(filename)
    return np.mean(np.sort(rawCosts), axis=1)

# %%
costsPure = raw2costsEnsemble(work_path+"/costsPureEnsemble.npy")[start_l_idx:-1]
costsPartnered = raw2costsEnsemble(work_path+"/costsPartneredEnsemble.npy")[start_l_idx:-1]

# %%
analysis = Analysis(args_list[start_l_idx:-1], center_vars, center_diff_vars, costsPure, costsPartnered)

# %%
taus = [1.25e-1, 1e-1, 7.5e-2, 5.0e-2, 3.5e-2, 2.5e-2]

Nes = []
for tau in taus:
    ML_Nes = analysis.optimal_Ne(tau=tau)
    SL_Ne = np.ceil(analysis.work(ML_Nes)/costsPure[-1]).astype(int)

    Nes.append([ML_Nes, SL_Ne])


# %%
from utils.BasinParameters import * 

# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls[start_l_idx:-1]])+"\n\n")

log.write("ML_Nes = " + ", ".join([str(Ne[0]) for Ne in Nes])+"\n")
log.write("ML_Nes = " + ", ".join([str(Ne[1]) for Ne in Nes])+"\n\n")

log.write("Reference level (one step finer than finest ensemnble level)\n")
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
log.write("Metric: Variance in the Kalman Gain estimator\n")
log.write("Metric: L2-errror in Kalman Gain estimator\n")

# %% 
MLonly = True

# %%
from utils import VarianceStatistics as VS

if not MLonly:
    var_SL = len(taus)*[VS.WelfordsVariance3((args_list[-1]["ny"], args_list[-1]["nx"]))]
var_ML = len(taus)*[VS.WelfordsVariance3((args_list[-1]["ny"], args_list[-1]["nx"]))]

if not MLonly:
    varNorm_SL = np.zeros((len(taus),3))
varNorm_ML = np.zeros((len(taus),3))

if not MLonly:
    err_SL = np.zeros((len(taus),3))
err_ML = np.zeros((len(taus),3))

os.makedirs(output_path+"/GainFigs", exist_ok=True)

# %% 
########################################
# NEW truth
print("Truth")
if os.path.isdir("tmpTruth"):
    for f in os.listdir("tmpTruth"):
        os.remove(os.path.join("tmpTruth", f))

os.system("python BasinKGainConvSingle.py -m T")


########################################
# NEW reference
print("Reference")

if os.path.isdir("tmpRefGain"):
    for f in os.listdir("tmpRefGain"):
        os.remove(os.path.join("tmpRefGain", f))

os.system("python BasinKGainConvSingle.py -m R")

f = os.listdir("tmpRefGain")[0]
refK_eta, refK_hu, refK_hv = np.load(os.path.join("tmpRefGain", f))[:,:,:,0] #block_reduce(np.load(os.path.join("tmpRefGain", f))[:,:,:,0], block_size=(1,2,2), func=np.mean)

fig, axs = imshow3([refK_eta, refK_hu, refK_hv], eta_vlim=1e-2,huv_vlim=1)
fig.savefig(output_path+"/GainFigs/Ref")
plt.close("all")


# %%
###########################################
# tau-LOOP
for tau_idx in range(len(taus)): 
    print("-----------------------------")
    print("tau = ", taus[tau_idx])

    ###########################################
    # N-LOOP 
    for n in range(N_exp):

        if not MLonly:
            # NEW SL Ensemble
            print("SL experiment number ", n)
            os.system("python BasinKGainConvSingle.py -m SL -Ne "+str(Nes[tau_idx][1]))

            if os.path.isdir("tmpSLGain"):
                assert len(os.listdir("tmpSLGain")) == 1, "Please remove old files"
                f = os.listdir("tmpSLGain")[0]
                K_eta, K_hu, K_hv = np.load(os.path.join("tmpSLGain", f))[:,:,:,0].repeat(2,1).repeat(2,2)
                var_SL[tau_idx].update(K_eta, K_hu, K_hv)

                fig, axs = imshow3([K_eta,K_hu,K_hv], eta_vlim=1e-2,huv_vlim=1)
                fig.savefig(output_path+"/GainFigs/SL"+str(tau_idx)+"_"+str(n))
                plt.close("all")

                err_SL[tau_idx] += 1/N_exp * np.array([np.linalg.norm(err_field) for err_field in [refK_eta-K_eta, refK_hu-K_hu, refK_hv-K_hv] ])

                os.remove(os.path.join("tmpSLGain", f))
            #TODO: Calculate variance in the error?!



        # NEW ML Ensemble
        print("ML experiment number ", n)
        os.system("python BasinKGainConvSingle.py -m ML -Ne "+" ".join(str(Ne) for Ne in Nes[tau_idx][0]))

        if os.path.isdir("tmpMLGain"):
            assert len(os.listdir("tmpMLGain")) == 1, "Please remove old files"
            f = os.listdir("tmpMLGain")[0]
            K_eta, K_hu, K_hv = np.load(os.path.join("tmpMLGain", f))[:,:,:,0].repeat(2,1).repeat(2,2)
            var_ML[tau_idx].update(K_eta, K_hu, K_hv)

            fig, axs = imshow3([K_eta,K_hu,K_hv], eta_vlim=1e-2,huv_vlim=1)
            fig.savefig(output_path+"/GainFigs/ML"+str(tau_idx)+"_"+str(n))
            plt.close("all")

            err_ML[tau_idx] += 1/N_exp * np.array([np.linalg.norm(err_field) for err_field in [refK_eta-K_eta, refK_hu-K_hu, refK_hv-K_hv] ])

            os.remove(os.path.join("tmpMLGain", f))
        #TODO: Calculate variance in the error?!



        # end: n-loop 
    
    if not MLonly:
        eta_varSL, hu_varSL, hv_varSL = var_SL[tau_idx].finalize()
        fig, axs = imshow3var([eta_varSL, hu_varSL, hv_varSL], eta_vlim=1e-5,huv_vlim=1e-1)
        fig.savefig(output_path+"/GainFigs/SLvar"+str(tau_idx))
        plt.close("all")

        varNorm_SL[tau_idx] = [np.linalg.norm(var_field) for var_field in [eta_varSL, hu_varSL, hv_varSL] ]
    
    
    eta_varML, hu_varML, hv_varML = var_ML[tau_idx].finalize()
    fig, axs = imshow3var([eta_varML, hu_varML, hv_varML], eta_vlim=1e-5,huv_vlim=1e-1)
    fig.savefig(output_path+"/GainFigs/MLvar"+str(tau_idx))
    plt.close("all")

    varNorm_ML[tau_idx] = [np.linalg.norm(var_field) for var_field in [eta_varML, hu_varML, hv_varML] ]


    # Saving results (backup after every tau)
    if not MLonly:
        np.savetxt(output_path+"/var_SL.txt", varNorm_SL)
    np.savetxt(output_path+"/var_ML.txt", varNorm_ML)

    if not MLonly:
        np.savetxt(output_path+"/errSLGain.txt", err_SL)
    np.savetxt(output_path+"/errMLGain.txt", err_ML)
    

    # end: tau-loop

# %% 
# Saving results (final)
if not MLonly:
    np.savetxt(output_path+"/var_SL.txt", varNorm_SL)
np.savetxt(output_path+"/var_ML.txt", varNorm_ML)

if not MLonly:
    np.savetxt(output_path+"/errSLGain.txt", err_SL)
np.savetxt(output_path+"/errMLGain.txt", err_ML)

# %% 
# Cleaning Up 
if os.path.isdir("tmpTruth"):
    for f in os.listdir("tmpTruth"):
        os.remove(os.path.join("tmpTruth", f))
if os.path.isdir("tmpTruth"):
    os.rmdir("tmpTruth")

if os.path.isdir("tmpRefGain"):
    for f in os.listdir("tmpRefGain"):
        os.remove(os.path.join("tmpRefGain", f))
if os.path.isdir("tmpRefGain"):
    os.rmdir("tmpRefGain")

if os.path.isdir("tmpSLGain"):
    for f in os.listdir("tmpSLGain"):
        os.remove(os.path.join("tmpSLGain", f))
if os.path.isdir("tmpSLGain"):
    os.rmdir("tmpSLGain")

if os.path.isdir("tmpMLGain"):
    for f in os.listdir("tmpMLGain"):
        os.remove(os.path.join("tmpMLGain", f))
if os.path.isdir("tmpMLGain"):
    os.rmdir("tmpMLGain")

# %%
