# %% [markdown]
# # Double Jet
# 
# With varioous drifter parameters

# %%
from gpuocean.utils import Common

import numpy as np
import copy
import pycuda.driver as cuda
from matplotlib import pyplot as plt

plt.rcParams["image.origin"] = "lower"


# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

# %%
import sys, os


# %% [markdown]
# ### Load Ensemble

# %%
source_path = "/home/florianb/havvarsel/multilevelDA/doublejet/scripts/DataAssimilation/DoubleJetSLDA/2023-11-10T11_44_24"

# %%
from gpuocean.utils import DoubleJetCase

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.DoubleJetParametersReplication import *
from utils.DoubleJetSL import *

SL_state = np.load(source_path+"/SLstates/SLensemble_864000.npy")

ny, nx = SL_state.shape[1:3]
doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=ny, nx=nx)
doubleJetCase_args, doubleJetCase_init, _ = doubleJetCase.getInitConditions()

SL_Ne = SL_state.shape[-1]

SL_ensemble = initSLensemble(SL_Ne, doubleJetCase_args, doubleJetCase_init,
                        sim_model_error_basis_args, sim_model_error_timestep)

SLupload(SL_ensemble, SL_state)
for e in range(SL_Ne):
    SL_ensemble[e].t = np.float32(T_spinup + T_da)


# %% [markdown]
# ### Attach drifters

# %%
# Prepare drifters

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
    for e in range(SL_Ne):
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


# %% 
# Forecasts
while SL_ensemble[0].t < T_spinup + T_da + T_forecast:
    t_start = SL_ensemble[0].t
    
    while SL_ensemble[0].t < t_start + da_timestep:
        SLstepToObservation(SL_ensemble, np.minimum(t_start + da_timestep, SL_ensemble[0].t + drifter_dt) )
        MLdrift(MLdrifters, drifter_dt)
    
    print(SL_ensemble[0].t)
    registerDrifterPositions(MLdrifters, MLdriftTrajectory, SL_ensemble[0].t)
    for e in range(len(SL_ensemble)):
        forecasts[e].add_observation_from_sim(SL_ensemble[e])


# %% 
# Save results
drifter_folder = os.path.join(source_path, 'sldrifters_'+str(drifter_dt))
os.makedirs(drifter_folder)

for e in range(len(SL_ensemble)):
    forecasts[e].to_pickle(os.path.join(drifter_folder,"sldrifters_"+str(e).zfill(4)+".pickle"))

for e in range(drifterEnsembleSize):
    filename = os.path.join(drifter_folder, "mldrifters_" + str(e).zfill(4) + ".bz2")
    MLdriftTrajectory[e].to_pickle(filename)

