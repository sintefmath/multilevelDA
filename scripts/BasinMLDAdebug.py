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

output_path = "DataAssimilation/Basin/"+timestamp 
os.makedirs(output_path)

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for the SHERLOCKING\n\n")

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common
from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.BasinInit import *
from utils.BasinPlot import *
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()

    
# %% [markdown]
# ## Setting-up case with different resolutions

# %%
ls = [8, 9]

# %% 
sample_args = {
    "g": 9.81,
    "f": 0.0012,
    }


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
steady_state_bump_a = 3
steady_state_bump_fractal_dist = 7

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
    "basis_x_start": 1, 
    "basis_x_end": 7,
    "basis_y_start": 2,
    "basis_y_end": 8,

    "kl_decay": 1.25,
    "kl_scaling": 0.004,
}

# %% 
# Flags for model error
import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--Tda', type=float, default=6*3600)
parser.add_argument('--Tforecast', type=float, default=6*3600)
parser.add_argument('--init_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error', type=int, default=1,choices=[0,1])
parser.add_argument('--sim_error_timestep', type=float, default=60) 
parser.add_argument('--truth_path', type=str, default="/home/florianb/havvarsel/multilevelDA/scripts/DataAssimilation/Truth/2023-05-16T13_18_49")

pargs = parser.parse_args()

T_da = pargs.Tda
T_forecast = pargs.Tforecast
init_model_error = bool(pargs.init_error)
sim_model_error = bool(pargs.sim_error)
sim_model_error_timestep = pargs.sim_error_timestep
truth_path = pargs.truth_path


# %% [markdown] 
# ## Ensemble

# %% 
ML_Nes = [150, 50]

# %% 
# Truth observation
Hx = 250
Hy = 500
R = [0.05, 1.0, 1.0]

# %% 
# Assimilation
r = 2.5e4
relax_factor = 0.1
min_location_level = 0

da_timestep = 900

# %%
# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")

log.write("Nes = " + ", ".join([str(Ne) for Ne in ML_Nes])+"\n\n")

log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
log.write("T (DA) = " + str(T_da) +"\n")
log.write("T (forecast) = " + str(T_forecast) +"\n\n")

log.write("Init State\n")
log.write("Double Bump\n")
log.write("Bump size [m]: " + str(steady_state_bump_a) +"\n")
log.write("Bump dist [fractal]: " + str(steady_state_bump_fractal_dist) + "\n\n")

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
if truth_path != "NEW":
    log.write("from file: " + truth_path + "\n")

    truth0 = np.load(truth_path+"/truth_0.npy")
    assert truth0.shape[1] == args_list[-1]["ny"], "Truth has wrong dimensions"
    assert truth0.shape[2] == args_list[-1]["nx"], "Truth has wrong dimensions"
else:
    log.write("saved to file\n")

log.write("Hx, Hy: " + str(Hx) + ", " + str(Hy) + "\n")
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
    fig, axs = imshowSim(truth, eta_vlim=steady_state_bump_a, huv_vlim=25*steady_state_bump_a)
    plt.savefig(output_path+"/truth_"+str(int(truth.t))+".pdf")



def makePlots(MLOceanEnsemble, ML_K, ML_K0, innov_prior, innov_posterior):
    # 1 mean
    MLmean = MLOceanEnsemble.estimate(np.mean)
    fig, axs = imshow3(MLmean, eta_vlim=steady_state_bump_a, huv_vlim=25*steady_state_bump_a)
    plt.savefig(output_path+"/MLmean_"+str(int(MLOceanEnsemble.t))+".pdf")
    plt.close('all')

    # 1.1 level mean
    # ML_state = MLOceanEnsemble.download()
    # imshow3(np.mean(ML_state[0], axis=-1))
    # plt.savefig(output_path+"/MLmean_"+str(int(T))+"_0.pdf")

    # imshow3(np.mean(ML_state[1][0], axis=-1))
    # plt.savefig(output_path+"/MLmean_"+str(int(T))+"_1_0.pdf")
    
    # imshow3(np.mean(ML_state[1][1], axis=-1))
    # plt.savefig(output_path+"/MLmean_"+str(int(T))+"_1_1.pdf")
    
    # imshow3(np.mean(ML_state[1][0] - ML_state[1][1].repeat(2,1).repeat(2,2), axis=-1), eta_vlim=1e-3, huv_vlim=1e-1)
    # plt.savefig(output_path+"/MLmean_"+str(int(T))+"_1.pdf")
    
    # 2 var 
    MLvar  = MLOceanEnsemble.estimate(np.var)
    fig, axs = imshow3var(MLvar, eta_vlim=0.015, huv_vlim=50)
    plt.savefig(output_path+"/MLvar_"+str(int(MLOceanEnsemble.t))+".pdf")
    plt.close('all')
   
    # 3 ML_K
    eta_vlim=5e-3
    huv_vlim=0.5
    cmap="coolwarm"

    if ML_K is not None:
        fig, axs = plt.subplots(2,3, figsize=(15,10))

        for i in range(2):
            etahuhv = ML_K[:,i].reshape(3, args_list[-1]["ny"], args_list[-1]["nx"])

            im = axs[i,0].imshow(etahuhv[0], vmin=-eta_vlim, vmax=eta_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,0], shrink=0.5)
            axs[i,0].set_title("$\eta$", fontsize=15)

            im = axs[i,1].imshow(etahuhv[1], vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,1], shrink=0.5)
            axs[i,1].set_title("$hu$", fontsize=15)

            im = axs[i,2].imshow(etahuhv[2], vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,2], shrink=0.5)
            axs[i,2].set_title("$hv$", fontsize=15)

        plt.savefig(output_path+"/MLK_"+str(int(MLOceanEnsemble.t))+".pdf")
        plt.close('all')

    # 3.2 
    if ML_K0 is not None:
        fig, axs = plt.subplots(2,3, figsize=(15,10))

        for i in range(2):
            etahuhv = ML_K0[:,:,:,i]

            im = axs[i,0].imshow(etahuhv[0], vmin=-eta_vlim, vmax=eta_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,0], shrink=0.5)
            axs[i,0].set_title("$\eta$", fontsize=15)

            im = axs[i,1].imshow(etahuhv[1], vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,1], shrink=0.5)
            axs[i,1].set_title("$hu$", fontsize=15)

            im = axs[i,2].imshow(etahuhv[2], vmin=-huv_vlim, vmax=huv_vlim, cmap=cmap)
            plt.colorbar(im, ax=axs[i,2], shrink=0.5)
            axs[i,2].set_title("$hv$", fontsize=15)

        plt.savefig(output_path+"/MLK_"+str(int(MLOceanEnsemble.t))+"_0.pdf")
        plt.close('all')

    # 4 Innovation
    with open(output_path+"/innovation.txt", "a") as f:
        if (innov_prior is not None) and (innov_posterior is not None):
            f.write(str(int(MLOceanEnsemble.t)) + ":  ")
            f.write(", ".join(["{:1.6f}".format(iuv) for iuv in innov_prior]))
            f.write(" >> ")
            f.write(", ".join(["{:1.6f}".format(iuv) for iuv in innov_posterior]) + "\n")


# %% 
# initial fields
data_args_list = []
for l_idx in range(len(args_list)):
    data_args_list.append( make_init_steady_state(args_list[l_idx], a=steady_state_bump_a, bump_fractal_dist=steady_state_bump_fractal_dist) )

# %% 
if truth_path=="NEW":
    truth = make_sim(args_list[-1], sample_args=sample_args, init_fields=data_args_list[-1])
    if init_model_error:
        init_mekl = ModelErrorKL.ModelErrorKL(**args_list[-1], **init_model_error_basis_args)
        init_mekl.perturbSim(truth)
    if sim_model_error:
        truth.setKLModelError(**sim_model_error_basis_args)
        truth.model_time_step = sim_model_error_timestep

# %%
# Ensemble
from utils.BasinEnsembleInit import *
ML_ensemble = initMLensemble(ML_Nes, args_list, data_args_list, sample_args, 
                             init_model_error_basis_args=init_model_error_basis_args, 
                             sim_model_error_basis_args=sim_model_error_basis_args, sim_model_error_time_step=sim_model_error_timestep)

from gpuocean.ensembles import MultiLevelOceanEnsemble
MLOceanEnsemble = MultiLevelOceanEnsemble.MultiLevelOceanEnsemble(ML_ensemble)


# %%
from gpuocean.dataassimilation import MLEnKFOcean
MLEnKF = MLEnKFOcean.MLEnKFOcean(MLOceanEnsemble)

obs_x = MLEnKF.X[0,Hx]
obs_y = MLEnKF.Y[Hy,0]
precomp_GC = MLEnKF.GCweights(obs_x, obs_y, r) 


# %% 
def KalmanGain0():
    print("SOMETHING IS WRONG HERE!")
    ML_state = MLOceanEnsemble.download()

    X0 = ML_state[0]
    X0mean = np.average(X0, axis=-1)

    Y0 = ML_state[0][1:3,int(Hy/2),int(Hx/2)] + np.random.multivariate_normal(np.zeros(3)[1:3], np.diag(R[1:3]), size=ML_Nes[0]).T
    Y0mean = np.average(Y0, axis=-1)

    lvl_weight = relax_factor * np.tile(block_reduce(precomp_GC, block_size=(2**(2-1),2**(2-1)), func=np.mean).flatten(),3)

    ML_XY0 = (lvl_weight[:,np.newaxis] 
            * 1/ML_Nes[0] 
            *( (X0-X0mean[:,:,:,np.newaxis]).reshape(-1,X0.shape[-1]) 
            @ (Y0 - Y0mean[:,np.newaxis]).T)
        ).reshape(X0mean.shape + (2,))

    ML_HXY0 = ML_XY0.reshape(ML_state[0].shape[:-1] + (2,))[2,int(Hy/2),int(Hx/2),:]
    ML_YY0  = ML_HXY0 + np.diag(R[1:3])
    ML_K0 = ML_XY0 @ np.linalg.inv(ML_YY0)

    return ML_K0


# %% 
# DA period
makePlots(MLOceanEnsemble, None, None, None, None)
if truth_path == "NEW":
    makeTruePlots(truth)

while MLOceanEnsemble.t < T_da:
    # Forward step
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + da_timestep)

    # DA step
    print("DA at ", MLOceanEnsemble.t)
    if truth_path == "NEW":
        truth.dataAssimilationStep(truth.t + da_timestep)
        true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)
    else:
        true_eta, true_hu, true_hv = np.load(truth_path+"/truth_"+str(int(MLOceanEnsemble.t))+".npy")
    obs = [true_eta[Hy,Hx], true_hu[Hy,Hx], true_hv[Hy,Hx]] + np.random.normal(0,R)

    ML_state = MLOceanEnsemble.download()
    prior_var0 = np.var(ML_state[0][:, int(Hy/2), int(Hx/2),:], axis=-1)
    prior_var1_0 = np.var(ML_state[1][0][:, Hy, Hx,:], axis=-1)
    prior_var1_1 = np.var(ML_state[1][1][:, int(Hy/2), int(Hx/2),:], axis=-1)
    prior_var1 = np.var(ML_state[1][0][:, Hy, Hx,:] - ML_state[1][1][:, int(Hy/2), int(Hx/2),:], axis=-1)

    innovation_prior = MLOceanEnsemble.estimate(np.mean)[:, Hy, Hx][slice(1,3)] - obs[slice(1,3)]
    ML_K = MLEnKF.assimilate(MLOceanEnsemble, obs, Hx, Hy, R, 
                            r=r, obs_var=slice(1,3), relax_factor=relax_factor, 
                            min_localisation_level=min_location_level,
                            precomp_GC=precomp_GC)
    innovation_posterior = MLOceanEnsemble.estimate(np.mean)[:, Hy, Hx][slice(1,3)] - obs[slice(1,3)]

    ML_state = MLOceanEnsemble.download()
    posterior_var0 = np.var(ML_state[0][:, int(Hy/2), int(Hx/2),:], axis=-1)
    posterior_var1_0 = np.var(ML_state[1][0][:, Hy, Hx,:], axis=-1)
    posterior_var1_1 = np.var(ML_state[1][1][:, int(Hy/2), int(Hx/2),:], axis=-1)
    posterior_var1 = np.var(ML_state[1][0][:, Hy, Hx,:] - ML_state[1][1][:, int(Hy/2), int(Hx/2),:], axis=-1)

    with open(output_path+"/var0.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in prior_var0]))
        f.write(("  > >  "))
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in posterior_var0]) + "\n")

    with open(output_path+"/var1_0.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in prior_var1_0]))
        f.write(("  > >  "))
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in posterior_var1_0]) + "\n")

    with open(output_path+"/var1_1.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in prior_var1_1]))
        f.write(("  > >  "))
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in posterior_var1_1]) + "\n")

    with open(output_path+"/var1.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.9f}".format(var_i) for var_i in prior_var1]))
        f.write(("  > >  "))
        f.write(", ".join(["{:1.9f}".format(var_i) for var_i in posterior_var1]) + "\n")

    makePlots(MLOceanEnsemble, ML_K, None, innovation_prior, innovation_posterior)

    if truth_path == "NEW":
        makeTruePlots(truth)


write2file(int(T_da))

# %%
# DA period
while MLOceanEnsemble.t < T_da + T_forecast:
    # Forward step
    MLOceanEnsemble.stepToObservation(MLOceanEnsemble.t + 3600)

    ML_state = MLOceanEnsemble.download()
    prior_var0 = np.var(ML_state[0][:, int(Hy/2), int(Hx/2),:], axis=-1)
    prior_var1_0 = np.var(ML_state[1][0][:, Hy, Hx,:], axis=-1)
    prior_var1_1 = np.var(ML_state[1][1][:, int(Hy/2), int(Hx/2),:], axis=-1)
    prior_var1 = np.var(ML_state[1][0][:, Hy, Hx,:] - ML_state[1][1][:, int(Hy/2), int(Hx/2),:], axis=-1)

    innovation_prior = MLOceanEnsemble.estimate(np.mean)[:, Hy, Hx][slice(1,3)] - obs[slice(1,3)]
 
    with open(output_path+"/var0.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in prior_var0]) + "\n")

    with open(output_path+"/var1_0.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in prior_var1_0]) + "\n")

    with open(output_path+"/var1_1.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.6f}".format(var_i) for var_i in prior_var1_1]) + "\n")

    with open(output_path+"/var1.txt", "a") as f:
        f.write(str(int(MLOceanEnsemble.t)) + ": ")
        f.write(", ".join(["{:1.9f}".format(var_i) for var_i in prior_var1]) + "\n")

    makePlots(MLOceanEnsemble, None, None, innovation_prior, innovation_posterior)
