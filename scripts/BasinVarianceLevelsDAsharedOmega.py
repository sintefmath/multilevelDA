# %% [markdown]
# # Multi Level Analysis

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
# import time
# print("Gonna sleep now!")
# time.sleep(3*3600)

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

output_path = "VarianceLevelsDA/Basin/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the experimental set-up\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

repo = git.Repo(search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")

log.write("Script " + str(os.path.basename(__file__))+ "\n\n")

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
from utils.BasinSL import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

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

# %% 
from utils.BasinParameters import *

# %% [markdown]
## Set-up statisitcs

# %% 
Ts = [0, 15*60, 3600, 6*3600, 12*3600]

#debug:
# Ts = [0, 15*60, 3600, 2*3600]
# T_da = 3600
# T_forecast = 3600

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Ne', type=int, default=100)

pargs = parser.parse_args()

Ne = pargs.Ne

# %%
localisation = False#True

# %% 
def g_functional(SL_ensemble):
    """
    L_g functional as in notation of Kjetil's PhD thesis.
    This should be the functional that is under investigation for the variance level plot

    Returns a ndarray of same size as SL_ensemble (3, ny, nx, Ne)
    """
    return (SLdownload(SL_ensemble) - SLestimate(SL_ensemble, np.mean)[:,:,:,np.newaxis])**2


def L2norm(field, lvl_grid_args):
    """
    integral_D(f dx)
    where D are uniform finite volumes

    Input:
    field           - ndarray of shape (3,ny,nx,..)
    lvl_grid_args   - dict with nx, ny and dx, dy information

    Output:
    L2norm          - ndarray of shape (3,...)
    """
    # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
    return np.sqrt(np.sum((field)**2 * lvl_grid_args["dx"]*lvl_grid_args["dy"], axis=(1,2)))


def L1norm(field, lvl_grid_args):
    """
    integral_D(f dx)
    where D are uniform finite volumes

    Input:
    field           - ndarray of shape (3,ny,nx,..)
    lvl_grid_args   - dict with nx, ny and dx, dy information

    Output:
    L1norm          - ndarray of shape (3,...)
    """
    # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
    return np.sum(np.abs(field) * lvl_grid_args["dx"]*lvl_grid_args["dy"], axis=(1,2))


def Enorm(field, lvl_grid_args):
    """
    integral_D(f dx)
    where D are uniform finite volumes

    Input:
    field           - ndarray of shape (3,ny,nx,..)
    lvl_grid_args   - dict with nx, ny and dx, dy information

    Output:
    L1norm          - ndarray of shape (3,...)
    """
    # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
    return np.mean(field, axis=(1,2))


norm = Enorm

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

grid_args = initGridSpecs(ls[-1])
log.write("nx = " + str(grid_args["nx"]) + ", ny = " + str(grid_args["ny"])+"\n")
log.write("dx = " + str(grid_args["dx"]) + ", dy = " + str(grid_args["dy"])+"\n")
log.write("T = " + ", ".join([str(T) for T in Ts]) +"\n\n")

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
log.write("obs_x, obs_y: " + " / ".join([str(obs_x) + ", " + str(obs_y)   for obs_x, obs_y in zip(obs_xs,obs_ys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("relax_factor = " + str(relax_factor) +"\n")
log.write("obs_var = slice(1,3)\n")
log.write("DA time steps: " + str(da_timestep) + "\n")
if localisation:
    log.write("r = " +str(r) + "\n")


log.write("\nStatistics\n")
log.write("Ne = " +str(Ne)+"\n")
log.write("g(u) = (u-E[u])^2\n")
log.write("norm = "+ str(norm.__name__) + "\n")
log.close()


# %% 
lvlvarsTs = np.zeros((len(Ts), len(ls), 3))
difflvlvarsTs = np.zeros((len(Ts), len(ls)-1, 3))

center_lvlvarsTs = np.zeros((len(Ts), len(ls), 3))
center_difflvlvarsTs = np.zeros((len(Ts), len(ls)-1, 3))


# %% 
data_args_list = [make_init_steady_state(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) for l_idx in range(len(ls))]

# %%
sim_args_list = []
for l_idx in range(len(ls)):
    sim_args = {
        "gpu_ctx" : args_list[l_idx]["gpu_ctx"],
        "nx" : args_list[l_idx]["nx"],
        "ny" : args_list[l_idx]["ny"],
        "dx" : args_list[l_idx]["dx"],
        "dy" : args_list[l_idx]["dy"],
        "f"  : sample_args["f"],
        "g"  : sample_args["g"],
        "r"  : 0,
        "dt" : 0,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
        "eta0" : data_args_list[l_idx]["eta"],
        "hu0"  : data_args_list[l_idx]["hu"],
        "hv0"  : data_args_list[l_idx]["hv"],
        "H"    : data_args_list[l_idx]["Hi"],
    }

    sim_args_list.append( sim_args )


# %%
init_mekls = [ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args) for l_idx in range(len(ls))]
sim_mekls = [ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args) for l_idx in range(len(ls))]

truth_init_mekl = ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args)
truth_sim_mekl = ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args)


# %%
# Dummy Truth
truth = CDKLM16.CDKLM16(**sim_args_list[l_idx])
truth_init_mekl.perturbSim(truth)
truth.model_error = truth_sim_mekl
truth.model_time_step = sim_model_error_timestep

# Dummy Ensemble
SL_ensembles = []
SL_ensembles.append([])
for l_idx in range(1,len(ls)):
    SL_ensembles.append([])

coarse_SL_ensembles = []
coarse_SL_ensembles.append([])
for l_idx in range(1,len(ls)):
    coarse_SL_ensembles.append([])

for e in range(Ne):
    sim = CDKLM16.CDKLM16(**sim_args_list[0]) 
    init_mekls[0].perturbSim(sim)
    SL_ensembles[0].append( sim )

    for l_idx in range(1, len(ls)):    
        sim = CDKLM16.CDKLM16(**sim_args_list[l_idx]) 
        init_mekls[l_idx].perturbSimSimilarAs(sim, modelError=init_mekls[0])
        SL_ensembles[l_idx].append( sim )

    coarse_SL_ensembles[0].append( None )
    for l_idx in range(1, len(ls)):    
        coarse_sim = CDKLM16.CDKLM16(**sim_args_list[l_idx-1]) 
        init_mekls[l_idx-1].perturbSimSimilarAs(coarse_sim, modelError=init_mekls[0])
        coarse_SL_ensembles[l_idx].append( coarse_sim )

# %%
for l_idx in range(len(ls)):
        np.save(output_path+"/SLensemble_"+str(l_idx)+"_init.npy", SLdownload(SL_ensembles[l_idx]))

# %%
# Weights
localisation_weights_list = len(obs_xs)*[None]
if localisation:
    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        localisation_weights_list[h] = GCweights(SL_ensembles[-1], obs_x, obs_y, r) 

# %%
# loop over time
t_now = 0.0
for t_idx, T in enumerate(Ts):

    numDAsteps = int((np.minimum(T, T_da + T_forecast)-t_now)/da_timestep)
    for DAstep in range(numDAsteps):
        
        # Forward step
        truth.dataAssimilationStep(t_now + da_timestep)

        numTsteps = int(da_timestep/sim_model_error_timestep) 
        for Tstep in range(numTsteps):
            for e in range(Ne):
                SL_ensembles[0][e].step(sim_model_error_timestep, apply_stochastic_term=False)
                sim_mekls[0].perturbSim(SL_ensembles[0][e])
                for l_idx in range(1, len(ls)):
                    SL_ensembles[l_idx][e].step(sim_model_error_timestep, apply_stochastic_term=False)
                    sim_mekls[l_idx].perturbSimSimilarAs(SL_ensembles[l_idx][e], modelError=sim_mekls[0])

                    coarse_SL_ensembles[l_idx][e].step(sim_model_error_timestep, apply_stochastic_term=False)
                    sim_mekls[l_idx-1].perturbSimSimilarAs(coarse_SL_ensembles[l_idx][e], modelError=sim_mekls[0])

            t_now = t_now + sim_model_error_timestep
            print(datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S"), ": ", t_now)

        # Update step
        # if DAstep < numDAsteps ?!?! DEBUG
        if DAstep < numDAsteps-1 and truth.t <= T_da:
            print("DA at " + str(truth.t))
            true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
            for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
                Hx, Hy = SLobsCoord2obsIdx(truth, obs_x, obs_y)
                obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)
    
                for l_idx in range(len(ls)):
                    SL_K, SL_perts = SLEnKF(SL_ensembles[l_idx], obs, obs_x, obs_y, R=R, obs_var=obs_var, 
                                            relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
                                            return_perts=True)
                    
                    if l_idx > 0:
                        coarselvlHx, coarselvlHy = SLobsCoord2obsIdx(coarse_SL_ensembles[l_idx], obs_x, obs_y)
                        coarse_SL_K = block_reduce(SL_K, block_size=(1,2,2,1), func=np.mean)

                        coarse_SL_state = SLdownload(coarse_SL_ensembles[l_idx])
                        coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,coarselvlHy,coarselvlHx] - SL_perts.T))
                        SLupload(coarse_SL_ensembles[l_idx], coarse_SL_state)

    print("Saving estimator variance at t=" + str(truth.t))
    for l_idx in range(len(ls)):
        lvlvarsTs[t_idx,l_idx] = norm(np.var(g_functional(SL_ensembles[l_idx]), axis=-1), args_list[l_idx])

        center_N = int(args_list[l_idx]["nx"]/4)
        center_x = int(args_list[l_idx]["nx"]/2)
        center_y = int(args_list[l_idx]["ny"]/2)
        center_lvlvarsTs[t_idx,l_idx] = norm(np.var(g_functional(SL_ensembles[l_idx])[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), args_list[l_idx])
            
        if l_idx > 0:
            difflvlvarsTs[t_idx,l_idx-1] = norm(np.var(g_functional(SL_ensembles[l_idx]) - g_functional(coarse_SL_ensembles[l_idx]).repeat(2,1).repeat(2,2), axis=-1), args_list[l_idx])
            center_difflvlvarsTs[t_idx,l_idx-1] = norm(np.var((g_functional(SL_ensembles[l_idx]) - g_functional(coarse_SL_ensembles[l_idx]).repeat(2,1).repeat(2,2))[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), args_list[l_idx])

    # Remaining Update step
    if T>0 and truth.t <= T_da:
        print("DA at " + str(truth.t))
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
        for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
            Hx, Hy = SLobsCoord2obsIdx(truth, obs_x, obs_y)
            obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)
  
            for l_idx in range(len(ls)):
                SL_K, SL_perts = SLEnKF(SL_ensembles[l_idx], obs, obs_x, obs_y, R=R, obs_var=obs_var, 
                                        relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
                                        return_perts=True)
                
                if l_idx > 0:
                    coarselvlHx, coarselvlHy = SLobsCoord2obsIdx(coarse_SL_ensembles[l_idx], obs_x, obs_y)
                    coarse_SL_K = block_reduce(SL_K, block_size=(1,2,2,1), func=np.mean)

                    coarse_SL_state = SLdownload(coarse_SL_ensembles[l_idx])
                    coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,coarselvlHy,coarselvlHx] - SL_perts.T))
                    SLupload(coarse_SL_ensembles[l_idx], coarse_SL_state)


    for l_idx in range(len(ls)):
        np.save(output_path+"/SLensemble_"+str(l_idx)+"_"+str(T)+".npy", SLdownload(SL_ensembles[l_idx]))
        if l_idx > 0:
            np.save(output_path+"/SLensemble_"+str(l_idx)+"_"+str(T)+"_coarse.npy", SLdownload(coarse_SL_ensembles[l_idx]))

# %% 
for t_idx, T in enumerate(Ts):
    varsT = [lvlvarsTs[t_idx,l_idx] for l_idx in range(len(ls))]
    np.save(output_path+"/vars_"+str(T), np.array(varsT))

    diff_varsT = [difflvlvarsTs[t_idx,l_idx] for l_idx in range(len(ls)-1)]
    np.save(output_path+"/diff_vars_"+str(T), np.array(diff_varsT))

    center_varsT = [center_lvlvarsTs[t_idx,l_idx] for l_idx in range(len(ls))]
    np.save(output_path+"/center_vars_"+str(T), np.array(center_varsT))

    center_diff_varsT = [center_difflvlvarsTs[t_idx,l_idx] for l_idx in range(len(ls)-1)]
    np.save(output_path+"/center_diff_vars_"+str(T), np.array(center_diff_varsT))

