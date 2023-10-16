# %% [markdown]
# # Kalman Gain convergence

# %% [markdown]
# ### Classes and modules

# %%
import os, sys

#Import packages we need
import numpy as np
import datetime
import copy

#For plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"

import pycuda.driver as cuda

# %% 
import signal

def handler(signum, frame):
    raise Exception("Time Out: Experiment aborted!")

signal.signal(signal.SIGALRM, handler)

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

base_path = "RankHistograms/ML/"
output_path = base_path+timestamp 
os.makedirs(output_path)

# %%
import shutil
shutil.copy(__file__, output_path + os.sep + "script_copy.py")

log = open(output_path+"/log.txt", 'w')
log.write("Parameters for Kalman Gain Convergence\n\n")

gpuocean_path = [p[:-4] for p in sys.path if (p.endswith("gpuocean/src") or p.endswith("gpuocean\\src"))][0]
import git
gpuocean_repo = git.Repo(gpuocean_path)
log.write("GPUOcean code from: " + str(gpuocean_repo.head.object.hexsha) + " on branch " + str(gpuocean_repo.active_branch.name) + "\n")

script_path = os.path.realpath(os.path.dirname(__file__))
repo = git.Repo(script_path, search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")


# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common

sys.path.insert(0, os.path.abspath(os.path.join(script_path, '../')))
from utils.BasinInit import *
from utils.BasinPlot import *
from utils.BasinAnalysis import *
from utils.BasinSL import *
from utils.BasinEnsembleInit import *

# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()


# %% 
if os.path.isfile(os.path.join(base_path, "logMLEnKF.txt")):
    assert False, "Remove old logs!"

# %%
from utils.BasinParameters import * 

# %%
ls = [6, 7, 8, 9]
Ne = [313,  84,  26,   6]
T_da = 6*3600
T_forecast = 3600


# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")
log.write("ML_Nes = " + ",".join([str(ne) for ne in Ne])+"\n")
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

log.write("obs_x, obs_y: " + " / ".join([str(obs_x) + ", " + str(obs_y)   for obs_x, obs_y in zip(obs_xs,obs_ys)]) + "\n")
log.write("R = " + ", ".join([str(Rii) for Rii in R])+"\n\n")

log.write("Assimilation\n")
log.write("DA time steps: " + str(da_timestep) + "\n")
log.write("obs_var = slice(1,3)\n")
log.write("r = " +str(r) + "\n")
log.write("relax_factor = " + str(relax_factor) +"\n")




# %% 
########################################
n = 0

while n < 1000:
    print("Experiment ", n)

    try:
        signal.alarm(15*60)

        if os.path.isdir(os.path.join(base_path,"tmp")):
            for f in os.listdir(os.path.join(base_path,"tmp")):
                os.remove(os.path.join(os.path.join(base_path,"tmp"), f))

        os.system("python "+script_path+"/BasinMLDA4ranks.py -ls "+ " ".join(str(l) for l in ls) + " -Ne "+ " ".join(str(ne) for ne in Ne) + " -Tda " +str(T_da)+ " -Tf " + str(T_forecast) )

        for f in os.listdir(os.path.join(base_path,"tmp")):
            os.rename(os.path.join(os.path.join(base_path,"tmp"), f), os.path.join(output_path, str(n)+"_"+f))

        signal.alarm(0)
        n=n+1
        
    except Exception as exc:
        print(exc)
        signal.alarm(0)
        pass
