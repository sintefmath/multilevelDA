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
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }


# %% 
init_model_error_basis_args = {
    "basis_x_start": 1, 
    "basis_x_end": 6,
    "basis_y_start": 2,
    "basis_y_end": 7,

    "kl_decay": 1.25,
    "kl_scaling": 0.18,
}

# %% 
sim_model_error_basis_args = {
    "basis_x_start": 2, 
    "basis_x_end": 7,
    "basis_y_start": 3,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.004,
}

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Ne', type=int, default=100)
parser.add_argument('--Tda', type=float, default=6*3600)
parser.add_argument('--Tforecast', type=float, default=6*3600)
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=60) 

pargs = parser.parse_args()

Ne = pargs.Ne
T_da = pargs.Tda
T_forecast = pargs.Tforecast
init_model_error = bool(pargs.init_error)
sim_model_error = bool(pargs.sim_error)
sim_model_error_timestep = pargs.sim_error_timestep

# %% [markdown] 
# ## Ensemble

# %% 
# Truth observation
truth_path = "/home/florianb/havvarsel/multilevelDA/scripts/DataAssimilation/Truth/2023-05-16T13_18_49"
Hxs = [ 250]
Hys = [500]
R = [0.1, 1.0, 1.0]

# %% 
# Assimilation
r = 2.5e4
relax_factor = 0.1
localisation = True

da_timestep = 900

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
log.write("Double Bump\n\n")

log.write("Init Perturbation\n")
if init_model_error:
    log.write("KL bases x start: " + str(init_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(init_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(init_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(init_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(init_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(init_model_error_basis_args["kl_scaling"]) + "\n\n")
else: 
    init_model_error_basis_args = None
    log.write("False\n\n")

log.write("Temporal Perturbation\n")
if sim_model_error:
    log.write("Model error timestep: " + str(sim_model_error_timestep) +"\n")
    log.write("KL bases x start: " + str(sim_model_error_basis_args["basis_x_start"]) + "\n")
    log.write("KL bases x end: " + str(sim_model_error_basis_args["basis_x_end"]) + "\n")
    log.write("KL bases y start: " + str(sim_model_error_basis_args["basis_y_start"]) + "\n")
    log.write("KL bases y end: " + str(sim_model_error_basis_args["basis_y_end"]) + "\n")
    log.write("KL decay: " + str(sim_model_error_basis_args["kl_decay"]) +"\n")
    log.write("KL scaling: " + str(sim_model_error_basis_args["kl_scaling"]) + "\n\n")
else:
    sim_model_error_basis_args = None
    log.write("False\n\n")

log.write("Truth\n")
log.write("from file: " + truth_path + "\n")

truth0 = np.load(truth_path+"/truth_0.npy")
assert truth0.shape[1] == grid_args["ny"], "Truth has wrong dimensions"
assert truth0.shape[2] == grid_args["nx"], "Truth has wrong dimensions"

log.write("Hx, Hy: " + " / ".join([str(Hx) + ", " + str(Hy)   for Hx, Hy in zip(Hxs,Hys)]) + "\n")
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

data_args = make_init_steady_state(args)


# %% 
def makePlots(SL_K):
    # mean
    SL_mean = SLestimate(SL_ensemble, np.mean)
    fig, axs = imshow3(SL_mean)
    plt.savefig(output_path+"/SLmean_"+str(int(SL_ensemble[0].t))+".pdf")

    # var
    SL_var = SLestimate(SL_ensemble, np.var)
    fig, axs = imshow3var(SL_var, eta_vlim=0.01, huv_vlim=50)
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
# Ensemble
SL_ensemble = initSLensemble(Ne, args, data_args, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)


# %%
if localisation:
    localisation_weights_list = []
    for Hx, Hy in zip(Hxs, Hys):
        localisation_weights_list.append( GCweights(SL_ensemble, Hx, Hy, r) ) 

# %% 
# DA period
# write2file(int(truth.t), "")

while SL_ensemble[0].t < T_da:
    # Forward step
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + da_timestep)

    # DA step
    # write2file(int(truth.t), "prior")
    true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(SL_ensemble[0].t))+".npy")
    for h, [Hx, Hy] in enumerate(zip(Hxs, Hys)):
        obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

        SL_K = SLEnKF(SL_ensemble, obs, Hx, Hy, R=R, obs_var=slice(1,3), 
               relax_factor=relax_factor, localisation_weights=localisation_weights_list[h])
    # write2file(int(truth.t), "posterior")

    makePlots(SL_K)



# %%
# Forecast period
while SL_ensemble[0].t < T_da + T_forecast:
    SLstepToObservation(SL_ensemble, SL_ensemble[0].t + 3600)
    # write2file(int(SL_ensemble[0].t), "")
    makePlots(None)