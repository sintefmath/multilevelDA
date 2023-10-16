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
output_path = "RankHistograms/ML/tmp" 
os.makedirs(output_path, exist_ok=True)

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *

# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %% 
from utils.BasinParameters import * 

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('-ls', "--MLlevels", type=int, nargs="*", required=True)
parser.add_argument('-Ne', "--ensembleSize", type=int, nargs="*", required=True)
parser.add_argument("-Tda", "--timeDA", type=float)
parser.add_argument("-Tf", "--timeForecast", type=float)

pargs = parser.parse_args()

ls = pargs.MLlevels
Ne = pargs.ensembleSize
T_da = pargs.timeDA
T_forecast = pargs.timeForecast


# %% [markdown] 
# ## Ensemble

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
# Ensemble
data_args_list = []
for l_idx in range(len(args_list)):
    data_args_list.append( make_init_steady_state(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) )


from utils.BasinEnsembleInit import *
ML_ensemble = initMLensemble(Ne, args_list, data_args_list, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)

from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)


from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

precomp_GC = []
for obs_x, obs_y in zip(obs_xs, obs_ys):
    precomp_GC.append( MLEnKF.GCweights(obs_x, obs_y, r) )


# %% 
# Truth
truth = make_sim(args_list[-1], sample_args=sample_args, init_fields=data_args_list[-1])
init_mekl = ModelErrorKL.ModelErrorKL(**args_list[-1], **init_model_error_basis_args)
init_mekl.perturbSim(truth)
truth.setKLModelError(**sim_model_error_basis_args)
truth.model_time_step = sim_model_error_timestep

# %% 
# DA period
while MLOceanEnsemble.t < T_da:
    # Forward step
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)
    truth.dataAssimilationStep(truth.t + da_timestep)

    # DA step
    print("DA at ", MLOceanEnsemble.t)
    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    
    ML_state = copy.deepcopy(MLOceanEnsemble.download())

    for h, [obs_x, obs_y] in enumerate(zip(obs_xs, obs_ys)):
        Hx, Hy = MLOceanEnsemble.obsLoc2obsIdx(obs_x, obs_y)
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.multivariate_normal(np.zeros(3),np.diag(R))

        ML_state = MLEnKF.assimilate(ML_state, obs, obs_x, obs_y, R, 
                                r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                                min_localisation_level=min_location_level,
                                precomp_GC=precomp_GC[h])
        
    MLOceanEnsemble.upload(ML_state)

# %%
# Forecast period
while MLOceanEnsemble.t < T_da + T_forecast:
    truth.dataAssimilationStep(truth.t + da_timestep)
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)

# %%
# Save results along the middle line of the y axis
y_idx = int(truth.ny/2)
y_idxs = [int(y_idx/(2**(len(ls)-1-l_idx))) for l_idx in range(len(ls))]

true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)

np.save(output_path+"/truth.npy", np.array([true_eta[y_idx], true_hu[y_idx], true_hv[y_idx]]))


ML_state = MLOceanEnsemble.download()

np.save(output_path+"/MLstate_0", ML_state[0][:,y_idxs[0]])
for l_idx in range(1,len(ls)):
    np.save(output_path+"/MLstate_"+str(l_idx)+"_0", ML_state[l_idx][0][:,y_idxs[l_idx]])
    np.save(output_path+"/MLstate_"+str(l_idx)+"_1", ML_state[l_idx][1][:,y_idxs[l_idx-1]])
    
# %%
# Crash to let OS do the gc
os._exit(0)