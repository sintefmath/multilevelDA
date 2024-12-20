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

output_path = os.path.join(os.path.realpath(os.path.dirname(__file__)),"DataAssimilation/DoubleJetSLDA/"+timestamp)
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(os.path.realpath(os.path.dirname(__file__)), search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")

import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../')))
from utils.DoubleJetPlot import *
from utils.DoubleJetSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
import argparse
parser = argparse.ArgumentParser(description='Ensemble inputs')
parser.add_argument("-L", "--level", required=True, type=int, default=9)
parser.add_argument("-Ne", "--ensembleSize", required=True, type=int, default=50)

pargs = parser.parse_args()
L = pargs.level
Ne = pargs.ensembleSize

# %% 
from utils.DoubleJetParametersReplication import *

# %%
from gpuocean.utils import DoubleJetCase

doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=2**L, nx=2**(L+1))
doubleJetCase_args, doubleJetCase_init, _ = doubleJetCase.getInitConditions()

# %%
truth_path = "/cluster/home/floribei/havvarsel/multilevelDA/scripts/DataAssimilation/DoubleJetTruth/2023-11-02T09_33_29"

# %% 
# Flags for model error
# import argparse
# parser = argparse.ArgumentParser(description='Generate an ensemble.')
# parser.add_argument('--Ne', type=int, default=50)
# parser.add_argument('--truth_path', type=str, default="/home/florianb/havvarsel/multilevelDA/doublejet/scripts/DataAssimilation/DoubleJetTruth/2023-09-15T15_08_08")

# pargs = parser.parse_args()

# Ne = pargs.Ne
# truth_path = pargs.truth_path

# %% 
# Assimilation
localisation = True

# %%
# Book keeping
log.write("L = " + str(L) + "\n")
log.write("Ne = " + str(Ne) + "\n\n")

log.write("nx = " + str(doubleJetCase_args["nx"]) + ", ny = " + str(doubleJetCase_args["ny"])+"\n")
log.write("dx = " + str(doubleJetCase_args["dx"]) + ", dy = " + str(doubleJetCase_args["dy"])+"\n")
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

log.write("Truth\n")
if truth_path != "NEW":
    log.write("from file: " + truth_path + "\n")

    # Load to get some parameters
    truth0 = np.load(truth_path+"/truth_"+str(T_spinup)+".npy")
    true_ny = truth0.shape[1]
    true_nx = truth0.shape[2]

    true_doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=true_ny, nx=true_nx)
    true_doubleJetCase_args, true_doubleJetCase_init, _ = true_doubleJetCase.getInitConditions()
    truth = CDKLM16.CDKLM16(**true_doubleJetCase_args, **true_doubleJetCase_init)

else:
    log.write("Generated on-the-fly and saved to file\n")

    truth = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init)
    truth.updateDt()
    truth.setKLModelError(**sim_model_error_basis_args)
    truth.model_time_step = sim_model_error_timestep

scale2truth_factor = int(np.log2(truth.nx/doubleJetCase_args["nx"])) 

log.write("obs_x, obs_y: " + " / ".join([str(obs_x) + ", " + str(obs_y)   for obs_x, obs_y in zip(obs_xs,obs_ys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("DA time steps: " + str(da_timestep) + "\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
log.write("obs_var = slice(1,3)\n")
if localisation:
    log.write("r = " +str(r) + "\n\n")

log.close()

# %% 

def write2file(SL_ensemble):
    SL_state = SLdownload(SL_ensemble)
    write_path = os.path.join(output_path, "SLstates")
    os.makedirs(write_path, exist_ok=True)
    np.save(write_path+"/SLensemble_"+str(int(SL_ensemble[0].t))+".npy", SL_state)

# %% 
def makeTruePlots(truth):
    os.makedirs(output_path+"/figs", exist_ok=True)
    fig, axs = imshowSim(truth)
    plt.savefig(output_path+"/figs/truth_"+str(int(truth.t))+".pdf", bbox_inches="tight")

def makePlots():
    os.makedirs(output_path+"/figs", exist_ok=True)
    # mean
    SL_mean = SLestimate(SL_ensemble, np.mean)
    fig, axs = imshow3(SL_mean)
    plt.savefig(output_path+"/figs/SLmean_"+str(int(SL_ensemble[0].t))+".pdf", bbox_inches="tight")

    # var
    SL_std = SLestimate(SL_ensemble, np.std)
    fig, axs = imshow3var(SL_std)
    plt.savefig(output_path+"/figs/SLstd_"+str(int(SL_ensemble[0].t))+".pdf", bbox_inches="tight")

    plt.close('all')


# %%
# Ensemble
from utils.DoubleJetSL import * 
SL_ensemble = initSLensemble(Ne, doubleJetCase_args, doubleJetCase_init, sim_model_error_basis_args, 
                             sim_model_error_time_step=sim_model_error_timestep)


# %%
if localisation:
    localisation_weights_list = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        localisation_weights_list.append( GCweights(SL_ensemble, obs_x, obs_y, r) ) 

# %% 
# DA recording 
def L2norm(fields):
    return np.sqrt(np.sum((fields)**2 * truth.dx*truth.dy, axis=(1,2)))

stddevs = []
rmses = []

# %%
##########################
# Spin up period
while SL_ensemble[0].t < T_spinup:
    SLstepToObservation(SL_ensemble, np.minimum(SL_ensemble[0].t + da_timestep, T_spinup))
    if truth_path == "NEW":
        truth.dataAssimilationStep(truth.t + 3600)
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    else:
        true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(SL_ensemble[0].t))+".npy")
    
    stddevs.append(L2norm(SLestimate(SL_ensemble, np.std, ddof=1).repeat(2**scale2truth_factor,1).repeat(2**scale2truth_factor,2)))
    rmses.append(L2norm(SLestimate(SL_ensemble, np.mean).repeat(2**scale2truth_factor,1).repeat(2**scale2truth_factor,2) - [true_eta, true_hu, true_hv]))

makePlots()
if truth_path == "NEW":
    makeTruePlots(truth)

# %% 
#########################
# DA period

while SL_ensemble[0].t < T_spinup + T_da:
    
    # Forward step
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + da_timestep)

    if truth_path == "NEW":
        truth.dataAssimilationStep(truth.t + da_timestep)
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    else:
        true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(SL_ensemble[0].t))+".npy")
    
    # DA step
    print("DA at ", SL_ensemble[0].t)

    SL_state = copy.deepcopy(SLdownload(SL_ensemble))

    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        Hx, Hy = SLobsCoord2obsIdx([truth], obs_x, obs_y)
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))

        SL_state = SLEnKF(SL_state, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
               relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
               dx=SL_ensemble[0].dx, dy=SL_ensemble[0].dy)
        
    SLupload(SL_ensemble, SL_state)

    stddevs.append(L2norm(SLestimate(SL_ensemble, np.std, ddof=1).repeat(2**scale2truth_factor,1).repeat(2**scale2truth_factor,2)))
    rmses.append(L2norm(SLestimate(SL_ensemble, np.mean).repeat(2**scale2truth_factor,1).repeat(2**scale2truth_factor,2) - [true_eta, true_hu, true_hv]))

    makePlots()
    if truth_path == "NEW":
        makeTruePlots(truth)

    
# %% 
# Save last state
write2file(SL_ensemble) 


# %% 
# Prepare drifters
from gpuocean.drifters import GPUDrifterCollection
from gpuocean.drifters import MLDrifterCollection
from gpuocean.utils import Observation
from gpuocean.dataassimilation import DataAssimilationUtils as dautils
from gpuocean.utils import OceanographicUtilities
observation_args = {'observation_type': dautils.ObservationType.UnderlyingFlow,
                'nx': doubleJetCase_args["nx"], 'ny': doubleJetCase_args["ny"],
                'domain_size_x': doubleJetCase_args["nx"]*doubleJetCase_args["dx"],
                'domain_size_y': doubleJetCase_args["ny"]*doubleJetCase_args["dy"],
               }

num_drifters = len(init_positions)

forecasts = []
for e in range(len(SL_ensemble)):
    forecast = Observation.Observation(**observation_args)
    drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                            boundaryConditions = doubleJetCase_args["boundary_conditions"],
                                            domain_size_x = forecast.domain_size_x,
                                            domain_size_y = forecast.domain_size_y)
    drifters.setDrifterPositions(init_positions)
    SL_ensemble[e].attachDrifters(drifters)
    forecast.add_observation_from_sim(SL_ensemble[0])
    forecasts.append(forecast)


# %% 
# Prepare ML drifters
drifterEnsembleSize = 250
drifter_dt = 60

MLdrifters = MLDrifterCollection.MLDrifterCollection(len(init_positions), drifterEnsembleSize, 
                                                        boundaryConditions=SL_ensemble[0].boundary_conditions,
                                                        domain_size_x=SL_ensemble[0].nx*SL_ensemble[0].dx,
                                                        domain_size_y=SL_ensemble[0].ny*SL_ensemble[0].dy)
MLdrifters.setDrifterPositions(init_positions)

MLdriftTrajectory = [None]*drifterEnsembleSize
for e in range(drifterEnsembleSize):
    MLdriftTrajectory[e] = Observation.Observation()


def registerDrifterPositions(MLdrifters, MLdriftTrajectory, t):
    
    for e in range(MLdrifters.ensemble_size):
        MLdriftTrajectory[e].add_observation_from_mldrifters(t, MLdrifters, e)


def estimateVelocity(func, desingularise=0.00001, **kwargs):
    """
    General monte-carlo estimator for some statistic given as func, performed on the ocean currects [u, v]
    func - function that calculates a statistics, e.g. np.mean or np.var
    returns [func(u), func(v)] with shape (2, ny, nx)
    """
    ensemble_state = []
    _, H_m = SL_ensemble[0].downloadBathymetry(interior_domain_only=True)
    for e in range(Ne):
        eta, hu, hv = SL_ensemble[e].download(interior_domain_only=True)
        u = OceanographicUtilities.desingularise(eta + H_m, hu, desingularise)
        v = OceanographicUtilities.desingularise(eta + H_m, hv, desingularise)
        ensemble_state.append(np.array([u, v])) 
    ensemble_state = np.moveaxis(ensemble_state, 0, -1)
    ensemble_estimate = func(ensemble_state, axis=-1, **kwargs)
    return ensemble_estimate


def MLdrift(MLdrifters, dt):
    mean_velocity = estimateVelocity(np.mean)
    var_velocity  = estimateVelocity(np.var, ddof=1)

    MLdrifters.drift(mean_velocity[0], mean_velocity[1], 
                        SL_ensemble[0].dx, SL_ensemble[0].dy, 
                        dt=dt, u_var=var_velocity[0], v_var=var_velocity[1])



if truth_path == "NEW":
    true_trajectories = Observation.Observation(**observation_args)

    true_drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                            boundaryConditions = doubleJetCase_args["boundary_conditions"],
                                            domain_size_x = true_trajectories.domain_size_x,
                                            domain_size_y = true_trajectories.domain_size_y)
    
    true_drifters.setDrifterPositions(init_positions)

    truth.attachDrifters(true_drifters)

    true_trajectories.add_observation_from_sim(truth)

# %%
# Forecast period
while SL_ensemble[0].t < T_spinup + T_da + T_forecast:
    t_start = SL_ensemble[0].t
    
    while SL_ensemble[0].t < t_start + da_timestep:
        SLstepToObservation(SL_ensemble, np.minimum(t_start + da_timestep, SL_ensemble[0].t + drifter_dt) )
        MLdrift(MLdrifters, drifter_dt)
    
    registerDrifterPositions(MLdrifters, MLdriftTrajectory, SL_ensemble[0].t)
    for e in range(len(SL_ensemble)):
        forecasts[e].add_observation_from_sim(SL_ensemble[e])
        
    if truth_path == "NEW":
        truth.dataAssimilationStep(SL_ensemble[0].t)
        true_trajectories.add_observation_from_sim(truth)


    makePlots()
    if truth_path == "NEW":
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    else:
        true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(SL_ensemble[0].t))+".npy")

    stddevs.append(L2norm(SLestimate(SL_ensemble, np.std, ddof=1).repeat(2**scale2truth_factor,1).repeat(2**scale2truth_factor,2)))
    rmses.append(L2norm(SLestimate(SL_ensemble, np.mean).repeat(2**scale2truth_factor,1).repeat(2**scale2truth_factor,2) - [true_eta, true_hu, true_hv]))

# Saving results
drifter_folder = os.path.join(output_path, 'sldrifters')
os.makedirs(drifter_folder)

if truth_path == "NEW":
    true_trajectories.to_pickle(os.path.join(drifter_folder,"true_drifters.pickle"))

for e in range(len(SL_ensemble)):
    forecasts[e].to_pickle(os.path.join(drifter_folder,"sldrifters_"+str(e).zfill(4)+".pickle"))

for e in range(drifterEnsembleSize):
    filename = os.path.join(drifter_folder, "mldrifters_" + str(e).zfill(4) + ".bz2")
    MLdriftTrajectory[e].to_pickle(filename)


np.save(output_path+"/stddev.npy", np.array(stddevs))
np.save(output_path+"/rmse.npy", np.array(rmses))