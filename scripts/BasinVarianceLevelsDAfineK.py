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

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Nvar', type=int, default=5)
parser.add_argument('--Ne', type=int, default=150)

pargs = parser.parse_args()

Ne = pargs.Ne
N_var = pargs.Nvar

# %%
localisation = True


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
log.write("N = " + str(N_var) + "\n")

log.close()

# %% 
def g_functional(SL_ensemble):
    """
    L_g functional as in notation of Kjetil's PhD thesis.
    This should be the functional that is under investigation for the variance level plot

    Returns a ndarray of same size as SL_ensemble (3, ny, nx, Ne)
    """
    return (SLdownload(SL_ensemble) - SLestimate(SL_ensemble, np.mean)[:,:,:,np.newaxis])**2


# %% 
varsNs = np.zeros((N_var, 3))
varsTs = [copy.deepcopy(varsNs) for T in Ts]
lvlvarsTs = [copy.deepcopy(varsTs) for l in ls]

difflvlvarsTs = copy.deepcopy(lvlvarsTs[:-1])

center_lvlvarsTs = copy.deepcopy(lvlvarsTs)
center_difflvlvarsTs = copy.deepcopy(lvlvarsTs[:-1])


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
    SL_ensembles.append([[],[]])


for e in range(Ne):
    sim = CDKLM16.CDKLM16(**sim_args_list[0]) 
    init_mekls[0].perturbSim(sim)
    sim.model_error = sim_mekls[0]
    sim.model_time_step = sim_model_error_timestep
    SL_ensembles[0].append( sim )

    for l_idx in range(1, len(ls)):    
        sim = CDKLM16.CDKLM16(**sim_args_list[l_idx]) 
        init_mekls[l_idx].perturbSimSimilarAs(sim, modelError=init_mekls[0])
        sim.model_error = sim_mekls[l_idx]
        sim.model_time_step = sim_model_error_timestep
        SL_ensembles[l_idx][0].append( sim )

        coarse_sim = CDKLM16.CDKLM16(**sim_args_list[l_idx-1])
        init_mekls[l_idx-1].perturbSimSimilarAs(coarse_sim, modelError=init_mekls[0])
        coarse_sim.model_error = sim_mekls[l_idx-1]
        coarse_sim.model_time_step = sim_model_error_timestep
        SL_ensembles[l_idx][1].append( coarse_sim )

# %%
# Weights
localisation_weights_list = len(obs_xs)*[None]
if localisation:
    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        localisation_weights_list[h] = GCweights(SL_ensembles[-1][0], obs_x, obs_y, r) 

# %%
##########################
# loop over samples
for n in range(N_var): 

    # New Truth
    truth.upload(data_args_list[l_idx]["eta"], data_args_list[l_idx]["hu"], data_args_list[l_idx]["hv"])
    truth.t = 0.0
    truth_init_mekl.perturbSim(truth)

    # New Ensemble
    for e in range(Ne):
        SL_ensembles[0][e].upload(data_args_list[0]["eta"], data_args_list[0]["hu"], data_args_list[0]["hv"])
        SL_ensembles[0][e].t = 0.0
        init_mekls[0].perturbSim(SL_ensembles[0][e])

        for l_idx in range(1, len(ls)):    
            SL_ensembles[l_idx][0][e].upload(data_args_list[l_idx]["eta"], data_args_list[l_idx]["hu"], data_args_list[l_idx]["hv"])
            SL_ensembles[l_idx][0][e].t = 0.0
            init_mekls[l_idx].perturbSimSimilarAs(SL_ensembles[l_idx][0][e], modelError=init_mekls[0])
            
            SL_ensembles[l_idx][1][e].upload(data_args_list[l_idx-1]["eta"], data_args_list[l_idx-1]["hu"], data_args_list[l_idx-1]["hv"])
            SL_ensembles[l_idx][1][e].t = 0.0
            init_mekls[l_idx-1].perturbSimSimilarAs(SL_ensembles[l_idx][1][e], modelError=init_mekls[0])
            

    ##########################
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
                    SL_ensembles[0][e].step(sim_model_error_timestep)
                    sim_mekls[0].perturbSim(SL_ensembles[0][e])
                    for l_idx in range(1, len(ls)):
                        SL_ensembles[l_idx][0][e].step(sim_model_error_timestep)
                        sim_mekls[l_idx].perturbSimSimilarAs(SL_ensembles[l_idx][0][e], modelError=sim_mekls[0])

                        SL_ensembles[l_idx][1][e].step(sim_model_error_timestep)
                        sim_mekls[l_idx-1].perturbSimSimilarAs(SL_ensembles[l_idx][1][e], modelError=sim_mekls[0])
                
                t_now = t_now + sim_model_error_timestep
            
            print(t_now, SL_ensembles[0][0].t)

            # Update step
            if DAstep < numDAsteps-1 and truth.t <= T_da:
                true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
                for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
                    Hx, Hy = SLobsCoord2obsIdx(truth, obs_x, obs_y)
                    obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                    SL_K, SL_perts = SLEnKF(SL_ensembles[-1][0], obs, obs_x, obs_y, R=R, obs_var=obs_var, 
                            relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
                            return_perts=True)

                    # Update coarse ensemble
                    lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[-1][1], obs_x, obs_y)
                    coarse_SL_K = block_reduce(SL_K, block_size=(1,2,2,1), func=np.mean)

                    coarse_SL_state = SLdownload(SL_ensembles[-1][1])
                    coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                    SLupload(SL_ensembles[-1][1], coarse_SL_state)


                    # Update 0-level ensemble
                    lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[0], obs_x, obs_y)
                    coarse_SL_K = block_reduce(SL_K, block_size=(1,2**(len(ls)-1),2**(len(ls)-1),1), func=np.mean)

                    coarse_SL_state = SLdownload(SL_ensembles[0])
                    coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                    SLupload(SL_ensembles[0], coarse_SL_state)

                    for l_idx in range(1, len(ls)-1):
                        # Update l+ ensemble
                        lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[l_idx][0], obs_x, obs_y)
                        coarse_SL_K = block_reduce(SL_K, block_size=(1,2**(len(ls)-1-l_idx),2**(len(ls)-1-l_idx),1), func=np.mean)

                        coarse_SL_state = SLdownload(SL_ensembles[l_idx][0])
                        coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                        SLupload(SL_ensembles[l_idx][0], coarse_SL_state)

                        # Update l- ensemble
                        lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[l_idx][1], obs_x, obs_y)
                        coarse_SL_K = block_reduce(SL_K, block_size=(1,2**(len(ls)-l_idx),2**(len(ls)-l_idx),1), func=np.mean)

                        coarse_SL_state = SLdownload(SL_ensembles[l_idx][1])
                        coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                        SLupload(SL_ensembles[l_idx][1], coarse_SL_state)


        print("Saving estimator variance at t=" + str(truth.t))
        lvlvarsTs[0][t_idx][n] = np.mean(np.var(g_functional(SL_ensembles[0]), axis=-1), axis=(1,2))

        center_N = int(args_list[0]["nx"]/4)
        center_x = int(args_list[0]["nx"]/2)
        center_y = int(args_list[0]["ny"]/2)
        center_lvlvarsTs[l_idx][t_idx][n] = np.mean(np.var(g_functional(SL_ensembles[0])[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), axis=(1,2))
            
        for l_idx in range(1,len(ls)):
            lvlvarsTs[l_idx][t_idx][n] = np.mean(np.var(g_functional(SL_ensembles[l_idx][0]), axis=-1), axis=(1,2))

            center_N = int(args_list[l_idx]["nx"]/4)
            center_x = int(args_list[l_idx]["nx"]/2)
            center_y = int(args_list[l_idx]["ny"]/2)
            center_lvlvarsTs[l_idx][t_idx][n] = np.mean(np.var(g_functional(SL_ensembles[l_idx][0])[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), axis=(1,2))
            
            difflvlvarsTs[l_idx-1][t_idx][n] = np.mean(np.var(g_functional(SL_ensembles[l_idx][0]) - g_functional(SL_ensembles[l_idx][1]).repeat(2,1).repeat(2,2), axis=-1), axis=(1,2))
            center_difflvlvarsTs[l_idx-1][t_idx][n] = np.mean(np.var((g_functional(SL_ensembles[l_idx][0]) - g_functional(SL_ensembles[l_idx][1]).repeat(2,1).repeat(2,2))[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), axis=(1,2))


        # Remaining Update step
        if truth.t <= T_da:
            true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
            for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
                Hx, Hy = SLobsCoord2obsIdx(truth, obs_x, obs_y)
                obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

                SL_K, SL_perts = SLEnKF(SL_ensembles[-1][0], obs, obs_x, obs_y, R=R, obs_var=obs_var, 
                        relax_factor=relax_factor, localisation_weights=localisation_weights_list[h],
                        return_perts=True)

                # Update coarse ensemble
                lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[-1][1], obs_x, obs_y)
                coarse_SL_K = block_reduce(SL_K, block_size=(1,2,2,1), func=np.mean)

                coarse_SL_state = SLdownload(SL_ensembles[-1][1])
                coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                SLupload(SL_ensembles[-1][1], coarse_SL_state)


                # Update 0-level ensemble
                lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[0], obs_x, obs_y)
                coarse_SL_K = block_reduce(SL_K, block_size=(1,2**(len(ls)-1),2**(len(ls)-1),1), func=np.mean)

                coarse_SL_state = SLdownload(SL_ensembles[0])
                coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                SLupload(SL_ensembles[0], coarse_SL_state)

                for l_idx in range(1, len(ls)-1):
                    # Update l+ ensemble
                    lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[l_idx][0], obs_x, obs_y)
                    coarse_SL_K = block_reduce(SL_K, block_size=(1,2**(len(ls)-1-l_idx),2**(len(ls)-1-l_idx),1), func=np.mean)

                    coarse_SL_state = SLdownload(SL_ensembles[l_idx][0])
                    coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                    SLupload(SL_ensembles[l_idx][0], coarse_SL_state)

                    # Update l- ensemble
                    lvlHx, lvlHy = SLobsCoord2obsIdx(SL_ensembles[l_idx][1], obs_x, obs_y)
                    coarse_SL_K = block_reduce(SL_K, block_size=(1,2**(len(ls)-l_idx),2**(len(ls)-l_idx),1), func=np.mean)

                    coarse_SL_state = SLdownload(SL_ensembles[l_idx][1])
                    coarse_SL_state = coarse_SL_state + (coarse_SL_K @ (obs[obs_var,np.newaxis] - coarse_SL_state[obs_var,lvlHy,lvlHx] - SL_perts.T))
                    SLupload(SL_ensembles[l_idx][1], coarse_SL_state)


# %% 
for t_idx, T in enumerate(Ts):
    varsT = [np.mean(lvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls))]
    np.save(output_path+"/vars_"+str(T), np.array(varsT))

    diff_varsT = [np.mean(difflvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls)-1)]
    np.save(output_path+"/diff_vars_"+str(T), np.array(diff_varsT))

    center_varsT = [np.mean(center_lvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls))]
    np.save(output_path+"/center_vars_"+str(T), np.array(center_varsT))

    center_diff_varsT = [np.mean(center_difflvlvarsTs[l_idx][t_idx], axis=0) for l_idx in range(len(ls)-1)]
    np.save(output_path+"/center_diff_vars_"+str(T), np.array(center_diff_varsT))

    