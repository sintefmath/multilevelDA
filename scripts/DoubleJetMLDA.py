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

output_path = "DataAssimilation/DoubleJet/"+timestamp 
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
from utils.BasinInit import *
from utils.DoubleJetPlot import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %%
ls = [8, 9]

# %% 
from utils.DoubleJetParametersReplication import * 

# %%
from gpuocean.utils import DoubleJetCase

args_list = []
data_args_list = []

for l in ls:
    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=2**l, nx=2**(l+1))
    doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()

    args = {key: doubleJetCase_args[key] for key in ('nx', 'ny', 'dx', 'dy', 'gpu_ctx', 'boundary_conditions')}
    args["gpu_stream"] = gpu_stream
    args_list.append(args)

    data_args = {"eta" : doubleJetCase_init["eta0"],
                "hu" : doubleJetCase_init["hu0"],
                "hv" : doubleJetCase_init["hv0"],
                "Hi" : doubleJetCase_args["H"]}
    data_args_list.append(data_args)

sample_args = {"f": doubleJetCase_args["f"], "g": doubleJetCase_args["g"]}



# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--truth_path', type=str, default="NEW")

pargs = parser.parse_args()

truth_path = pargs.truth_path


# %% [markdown] 
# ## Ensemble

# %% 
ML_Nes = [150, 50]

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("Nes = " + ", ".join([str(Ne) for Ne in ML_Nes])+"\n\n")

log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
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

log.write("Truth\n")
if truth_path != "NEW":
    log.write("from file: " + truth_path + "\n")

    truth0 = np.load(truth_path+"/truth_0.npy")
    assert truth0.shape[1] == args_list[-1]["ny"], "Truth has wrong dimensions"
    assert truth0.shape[2] == args_list[-1]["nx"], "Truth has wrong dimensions"
else:
    log.write("saved to file\n")

log.write("obs_x, obs_y: " + " / ".join([str(obs_x) + ", " + str(obs_y)   for obs_x, obs_y in zip(obs_xs,obs_ys)]) + "\n")
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
    ML_state = MLOceanEnsemble.download()
    np.save(output_path+"/MLensemble_0_"+str(T)+".npy", np.array(ML_state[0]))
    for l_idx in range(1,len(ls)):
        np.save(output_path+"/MLensemble_"+str(l_idx)+"_0_"+str(T)+".npy", np.array(ML_state[l_idx][0]))
        np.save(output_path+"/MLensemble_"+str(l_idx)+"_1_"+str(T)+".npy", np.array(ML_state[l_idx][1]))


def makeTruePlots(truth):
    fig, axs = imshowSim(truth)
    plt.savefig(output_path+"/truth_"+str(int(truth.t))+".pdf")



def makePlots(MLOceanEnsemble):
    # 1 mean
    MLmean = MLOceanEnsemble.estimate(np.mean)
    fig, axs = imshow3(MLmean)
    plt.savefig(output_path+"/MLmean_"+str(int(MLOceanEnsemble.t))+".pdf")
    plt.close('all')

    # 2 var 
    MLstd  = MLOceanEnsemble.estimate(np.std)
    fig, axs = imshow3var(MLstd)
    plt.savefig(output_path+"/MLvar_"+str(int(MLOceanEnsemble.t))+".pdf")
    plt.close('all')
   


# %% 
if truth_path=="NEW":
    truth = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init)
    truth.updateDt()
    truth.setKLModelError(**sim_model_error_basis_args)
    truth.model_time_step = sim_model_error_timestep

# %%
# Ensemble
from utils.BasinEnsembleInit import *
ML_ensemble = initMLensemble(ML_Nes, args_list, data_args_list, sample_args, 
                             init_model_error_basis_args=None, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)

from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)


# %%
from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

precomp_GC = []
for obs_x, obs_y in zip(obs_xs, obs_ys):
    precomp_GC.append( MLEnKF.GCweights(obs_x, obs_y, r) )

# Spin up period
truth.dataAssimilationStep(T_spinup)
MLOceanEnsemble.stepToObservation(T_spinup)

# %% 
# DA period
makePlots(MLOceanEnsemble)
if truth_path == "NEW":
    makeTruePlots(truth)

while MLOceanEnsemble.t < T_spinup + T_da:
    # Forward step
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)

    # DA step
    print("DA at ", MLOceanEnsemble.t)
    if truth_path == "NEW":
        truth.dataAssimilationStep(truth.t + da_timestep)
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    else:
        true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(MLOceanEnsemble.t))+".npy")

    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_x, obs_y)
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)
        
        ML_K = MLEnKF.assimilate(MLOceanEnsemble, obs, obs_x, obs_y, R, 
                                r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                min_localisation_level=min_location_level,
                                precomp_GC=precomp_GC[h])

    # if (MLOceanEnsemble.t % (6*3600) == 0):
    makePlots(MLOceanEnsemble)
    if truth_path == "NEW":
        makeTruePlots(truth)

sys.exit(0)
# %%
# DA period
while MLOceanEnsemble.t < T_spinup + T_da + T_forecast:
    # Forward step
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + 3600)
    makePlots(MLOceanEnsemble)