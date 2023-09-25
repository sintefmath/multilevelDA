# %% [markdown]
# # Multi Level Statistics

# %% [markdown]
# ### Classes and modules

# %%

#Import packages we need
import numpy as np
import sys, os
import copy

#For plotting
import matplotlib
from matplotlib import pyplot as plt

import pycuda.driver as cuda

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "DataAssimilation/MLDA/"+timestamp 
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
import argparse
parser = argparse.ArgumentParser(description='Ensemble inputs')
parser.add_argument("-ls", "--level", required=True, nargs="*", type=int, default=[7, 8])
parser.add_argument("-Ne", "--ensembleSize", required=True, nargs="*", type=int, default=[50, 25])

pargs = parser.parse_args()
ls = pargs.level
ML_Nes = pargs.ensembleSize

assert len(ls) == len(ML_Nes), "Non-matching levels and ensemble sizes"
# ls = [7, 8]
# ML_Nes = [100, 25]

# %% 
from utils.DoubleJetParametersReplication import * 

# %%
from gpuocean.utils import DoubleJetCase

args_list = []
init_list = []

for l in ls:
    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=2**l, nx=2**(l+1))
    doubleJetCase_args, doubleJetCase_init, _ = doubleJetCase.getInitConditions()

    args_list.append(doubleJetCase_args)
    init_list.append(doubleJetCase_init)


# %% 
truth_path = "/home/florianb/havvarsel/multilevelDA/doublejet/scripts/DataAssimilation/DoubleJetTruth/2023-09-15T15_08_08"


# %% [markdown] 
# ## Ensemble

# %% 
xorwow_seeds = len(ls)*[None]
np_seeds = len(ls)*[None]
enkf_seed = None

# min_location_level = 1
# xorwow_seeds = [1, 2]
# np_seeds = [3, 4]
# enkf_seed = 5


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

    truth0 = np.load(truth_path+"/truth_"+str(T_spinup)+".npy")
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
    os.makedirs(output_path+"/figs", exist_ok=True)
    fig, axs = imshowSim(truth)
    plt.savefig(output_path+"/figs/truth_"+str(int(truth.t))+".pdf", bbox_inches="tight")



def makePlots(MLOceanEnsemble):
    os.makedirs(output_path+"/figs", exist_ok=True)
    # 1 mean
    MLmean = MLOceanEnsemble.estimate(np.mean)
    fig, axs = imshow3(MLmean)
    plt.savefig(output_path+"/figs/MLmean_"+str(int(MLOceanEnsemble.t))+".pdf", bbox_inches="tight")
    plt.close('all')

    # 2 var 
    MLstd  = MLOceanEnsemble.estimate(np.std)
    fig, axs = imshow3var(MLstd)
    plt.savefig(output_path+"/figs/MLstd_"+str(int(MLOceanEnsemble.t))+".pdf", bbox_inches="tight")
    plt.close('all')
   


# %% 
if truth_path=="NEW":
    truth = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init)
    truth.updateDt()
    truth.setKLModelError(**sim_model_error_basis_args)
    truth.model_time_step = sim_model_error_timestep

# %%
# Ensemble
from utils.DoubleJetEnsembleInit import *
ML_ensemble = initMLensemble(ML_Nes, args_list, init_list,
                             sim_model_error_basis_args=sim_model_error_basis_args, 
                             sim_model_error_time_step=sim_model_error_timestep,
                             xorwow_seeds=xorwow_seeds, np_seeds=np_seeds)

from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)


# %%
from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble, seed=enkf_seed)

precomp_GC = []
for obs_x, obs_y in zip(obs_xs, obs_ys):
    precomp_GC.append( MLEnKF.GCweights(obs_x, obs_y, r) )

# Spin up period
if truth_path=="NEW":
    truth.dataAssimilationStep(T_spinup)
MLOceanEnsemble.stepToObservation(T_spinup)

# %% 
# DA period
makePlots(MLOceanEnsemble)
if truth_path == "NEW":
    makeTruePlots(truth)

ml_log = open(output_path+"/MLlog.txt", 'w')
ml_log.write("Eigenvalues of (Sigma_YY+R)\n\n")

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

    ML_state = copy.deepcopy(MLOceanEnsemble.download())
    
    ml_log.write("\n DA at " + str(MLOceanEnsemble.t) + "\n")
    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_x, obs_y)
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))
        
        ml_log.write("Observation " + str(h) + ":\n")
        ML_state = MLEnKF.assimilate(ML_state, obs, obs_x, obs_y, R, 
                                r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                min_localisation_level=min_location_level,
                                precomp_GC=precomp_GC[h],
                                log=ml_log)
    
    MLOceanEnsemble.upload(ML_state)

    # if (MLOceanEnsemble.t % (6*3600) == 0):
    makePlots(MLOceanEnsemble)
    if truth_path == "NEW":
        makeTruePlots(truth)

# %% 
# Save last state 
MLOceanEnsemble.save2file(os.path.join(output_path, "MLstates"))

# %%
# Prepare drifters
drifter_ensemble_size = 50
num_drifters = len(init_positions)

MLOceanEnsemble.attachDrifters(drifter_ensemble_size, drifterPositions=np.array(init_positions))

if truth_path == "NEW":
    from gpuocean.drifters import GPUDrifterCollection
    from gpuocean.utils import Observation
    from gpuocean.dataassimilation import DataAssimilationUtils as dautils
    observation_args = {'observation_type': dautils.ObservationType.UnderlyingFlow,
                    'nx': args_list[-1]["nx"], 'ny': args_list[-1]["ny"],
                    'domain_size_x': args_list[-1]["nx"]*args_list[-1]["dx"],
                    'domain_size_y': args_list[-1]["ny"]*args_list[-1]["dy"],
                }
    true_trajectories = Observation.Observation(**observation_args)

    true_drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                            boundaryConditions = args_list[-1]["boundary_conditions"],
                                            domain_size_x = true_trajectories.domain_size_x,
                                            domain_size_y = true_trajectories.domain_size_y)
    
    true_drifters.setDrifterPositions(init_positions)

    truth.attachDrifters(true_drifters)

    true_trajectories.add_observation_from_sim(truth)

# %%
# Forecast period
while MLOceanEnsemble.t < T_spinup + T_da + T_forecast:
    
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)
    MLOceanEnsemble.registerDrifterPositions()

    if truth_path == "NEW":
        truth.dataAssimilationStep(MLOceanEnsemble.t)
        true_trajectories.add_observation_from_sim(truth)

    if MLOceanEnsemble.t%3600 < 0.1:
        makePlots(MLOceanEnsemble)

# Save results
drifter_folder = os.path.join(output_path, 'mldrifters')
os.makedirs(drifter_folder)
MLOceanEnsemble.saveDriftTrajectoriesToFile(drifter_folder, "mldrifters")

if truth_path == "NEW":
    true_trajectories.to_pickle(os.path.join(drifter_folder,"true_drifters"))
