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

script_path = os.path.realpath(os.path.dirname(__file__))
output_path = script_path+"/RankHistograms/"+timestamp 
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

repo = git.Repo(script_path, search_parent_directories=True)
log.write("Current repo >>"+str(repo.working_tree_dir.split("/")[-1])+"<< with " +str(repo.head.object.hexsha)+ "on branch " + str(repo.active_branch.name) + "\n\n")


# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common

sys.path.insert(0, os.path.abspath(os.path.join(script_path, '../')))
# %%
gpu_ctx = Common.CUDAContext()
gpu_stream = cuda.Stream()


# %%
from utils.DoubleJetParametersReplication import * 

# %%
ls = [7, 8, 9]
Ne = [88, 37, 12]


# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls])+"\n\n")
log.write("ML_Nes = " + ",".join([str(ne) for ne in Ne])+"\n")
log.write("T (spinup) = " + str(T_spinup) +"\n")
log.close()

# %% 
########################################
n = 0

while n < 1000:
    print("Experiment ", n)

    try:
        signal.alarm(3*60*60)

        os.system("python "+script_path+"/DoubleJetMLDA4Ranks.py -ls "+ " ".join(str(l) for l in ls) + " -Ne "+ " ".join(str(ne) for ne in Ne) + " --timestamp " + timestamp + " -n " + str(n) )

        signal.alarm(0)
        n=n+1
        
    except Exception as exc:
        print(exc)
        signal.alarm(0)
        pass
