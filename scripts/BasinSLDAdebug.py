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

output_path = "DataAssimilation/BasinSL/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

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

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *
from utils.BasinSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
L = 9

# %% 
from utils.BasinParameters import *

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Ne', type=int, default=100)
parser.add_argument('--truth_path', type=str, default="/home/florianb/havvarsel/multilevelDA/scripts/DataAssimilation/Truth/2023-06-22T13_23_51")

pargs = parser.parse_args()

Ne = pargs.Ne
truth_path = pargs.truth_path


# %% 
# Assimilation
localisation = True


# %%
# Book keeping
log.write("L = " + str(L) + "\n")
log.write("Ne = " + str(Ne) + "\n\n")

grid_args = initGridSpecs(L)
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
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

log.write("Truth\n")
if truth_path != "NEW":
    log.write("from file: " + truth_path + "\n")

    truth0 = np.load(truth_path+"/truth_0.npy")
    assert truth0.shape[1] == grid_args["ny"], "Truth has wrong dimensions"
    assert truth0.shape[2] == grid_args["nx"], "Truth has wrong dimensions"
else:
    log.write("saved to file\n")


log.write("obs_x, obs_y: " + " / ".join([str(obs_x) + ", " + str(obs_y)   for obs_x, obs_y in zip(obs_xs,obs_ys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("r = " +str(r) + "\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
log.write("obs_var = slice(1,3)\n")
log.write("min_location_level = " + str(localisation) +"\n\n")
log.write("DA time steps: " + str(da_timestep) + "\n")

log.close()

# %% 
def write2file(T, mode=""):
    print("Saving ", mode, " at time ", T)

    SL_state = SLdownload(SL_ensemble)
    np.save(output_path+"/SLensemble_"+str(T)+"_"+mode+".npy", np.array(SL_state))
    

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

data_args = make_init_steady_state(args, a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist)


# %% 
def makeTruePlots(truth):
    fig, axs = imshowSim(truth, eta_vlim=steady_state_bump_a, huv_vlim=100)
    plt.savefig(output_path+"/truth_"+str(int(truth.t))+".pdf")

def makePlots(SL_K):
    # mean
    SL_mean = SLestimate(SL_ensemble, np.mean)
    fig, axs = imshow3(SL_mean, eta_vlim=steady_state_bump_a, huv_vlim=100)
    plt.savefig(output_path+"/SLmean_"+str(int(SL_ensemble[0].t))+".pdf")

    # var
    SL_var = SLestimate(SL_ensemble, np.var)
    fig, axs = imshow3var(SL_var, eta_vlim=0.025, huv_vlim=100)
    plt.savefig(output_path+"/SLvar_"+str(int(SL_ensemble[0].t))+".pdf")

    # Kalman gain
    if SL_K is not None:
        fig, axs = plt.subplots(2,3, figsize=(15,10))

        eta_vlim=5e-3
        huv_vlim=0.5
        cmap="coolwarm"

        for i in range(2):
            etahuhv = SL_K[:,:,:,i]

            im = axs[i,0].imshow(etahuhv[0], vmin=-eta_vlim, vmax=eta_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,0], shrink=0.5)
            axs[i,0].set_title("$\eta$", fontsize=15)

            im = axs[i,1].imshow(etahuhv[1], vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,1], shrink=0.5)
            axs[i,1].set_title("$hu$", fontsize=15)

            im = axs[i,2].imshow(etahuhv[2], vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,2], shrink=0.5)
            axs[i,2].set_title("$hv$", fontsize=15)

        plt.savefig(output_path+"/SLK_"+str(int(SL_ensemble[0].t))+".pdf")

    plt.close('all')


# %% 
if truth_path=="NEW":
    truth = make_sim(args, sample_args=sample_args, init_fields=data_args)
    init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)
    init_mekl.perturbSim(truth)
    truth.setKLModelError(**sim_model_error_basis_args)
    truth.model_time_step = sim_model_error_timestep

# %%
# Ensemble
SL_ensemble = initSLensemble(Ne, args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)


# %%
if localisation:
    localisation_weights_list = []
    for obs_x, obs_y in zip(obs_xs, obs_ys):
        localisation_weights_list.append( GCweights(SL_ensemble, obs_x, obs_y, r) ) 

# %% 
# DA period
# write2file(int(truth.t), "")
makePlots(None)

while SL_ensemble[0].t < T_da:
    # Forward step
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + da_timestep)

    # DA step
    print("DA at ", SL_ensemble[0].t)
    # write2file(int(truth.t), "prior")
    if truth_path == "NEW":
        truth.dataAssimilationStep(truth.t + da_timestep)
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    else:
        true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(SL_ensemble[0].t))+".npy")
    
    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y)
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

        SL_K = SLEnKF(SL_ensemble, obs, obs_x, obs_y, R=R, obs_var=slice(1,3), 
               relax_factor=relax_factor, localisation_weights=localisation_weights_list[h])
    # write2file(int(truth.t), "posterior")

    makePlots(SL_K)
    if truth_path == "NEW":
        makeTruePlots(truth)


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

forecasts = []
for e in range(len(SL_ensemble)):
    forecast = Observation.Observation(**observation_args)
    drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                            boundaryConditions = args["boundary_conditions"],
                                            domain_size_x = forecast.domain_size_x,
                                            domain_size_y = forecast.domain_size_y)
    drifters.setDrifterPositions(init_positions)
    SL_ensemble[e].attachDrifters(drifters)
    forecast.add_observation_from_sim(SL_ensemble[0])
    forecasts.append(forecast)

# %%
# Forecast period
while SL_ensemble[0].t < T_da + T_forecast:
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + 300.0)
    for e in range(len(SL_ensemble)):
        forecasts[e].add_observation_from_sim(SL_ensemble[e])
    # write2file(int(SL_ensemble[0].t), "")
    if SL_ensemble[0].t % 3600 < 0.1:
        makePlots(None)

# %%
for drifter_id in range(len(init_positions)): 
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    domain_extent = [0, SL_ensemble[0].nx*SL_ensemble[0].dx/1000, 0, SL_ensemble[0].ny*SL_ensemble[0].dy/1000]

    ax.imshow(np.zeros((grid_args["ny"], grid_args["nx"])), interpolation="none", origin='lower', 
                cmap=plt.cm.Oranges, extent=domain_extent, zorder=-10)

    for forecast in forecasts:
        path = forecast.get_drifter_path(drifter_id, 0,  SL_ensemble[0].t, in_km = True)[0]
        ax.plot(path[:,0], path[:,1], color="C"+str(drifter_id), ls="-", zorder=-3)

    plt.savefig(output_path+"/drift_"+str(drifter_id)+".pdf", bbox_inches="tight")
