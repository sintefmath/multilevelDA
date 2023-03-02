# %% [markdown]
# # Multi Resolution Simulation

# %% [markdown]
# ### Classes and modules

# %%

#Import packages we need
import numpy as np
import sys, os

#For plotting
import matplotlib
from matplotlib import pyplot as plt

# %%
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")

if not os.path.exists("OutputVarianceLevels"):
    os.makedirs("OutputVarianceLevels")

output_path = "OutputVarianceLevels/"+timestamp+"_Rossby"
os.makedirs(output_path)

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common, WindStress
from gpuocean.SWEsimulators import CDKLM16

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.RossbyInit import *
from utils.WindPerturb import *

# %%
gpu_ctx = Common.CUDAContext()

# %% [markdown]
# ## Setting-up case with different resolutions
# 
# IC are the bump from the Rossby adjustment case

# %%
ls = [6, 7, 8, 9, 10]

wind_N = 100
t_splits = 26

# KLSampler = KarhunenLoeve_Sampler(t_splits, wind_N, decay=1.15, scaling=0.9)
KLSampler = KarhunenLoeve_Sampler(t_splits, wind_N, decay=1.15, scaling=3)
wind_weight = wind_bump(KLSampler.N,KLSampler.N)

# %% [markdown]
# ## Variance Level Plot

class WelfordsVariance():
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    def __init__(self, shape):
        self.existingAggregate = (0, np.zeros(shape), np.zeros(shape))


    def update(self, newValue):
        # For a new value newValue, compute the new count, new mean, the new M2.
        # mean accumulates the mean of the entire dataset
        # M2 aggregates the squared distance from the mean
        # count aggregates the number of samples seen so far
        (count, mean, M2) = self.existingAggregate
        
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        
        self.existingAggregate = (count, mean, M2)


    def finalize(self):
        # Retrieve the mean, variance and sample variance from an aggregate
        (count, mean, M2) = self.existingAggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance)
    

# %% 
class WelfordsVariance3():
    def __init__(self, shape):
        self.wv_eta = WelfordsVariance(shape)
        self.wv_hu = WelfordsVariance(shape)
        self.wv_hv = WelfordsVariance(shape)

    def update(self, eta, hu, hv):
        self.wv_eta.update(eta)
        self.wv_hu.update(hu)
        self.wv_hv.update(hv)

    def finalize(self, m=1):
        # m - values in [0, 1, 2]
        # m = 0 : mean
        # m = 1 : variance
        # m = 2 : sample variance
        return (self.wv_eta.finalize()[m], self.wv_hu.finalize()[m], self.wv_hv.finalize()[m])
        

# %%
T = 125000

# %%
N_var = 25

vars = np.zeros((len(ls), 3))
diff_vars = np.zeros((len(ls), 3))

for l_idx, l in enumerate(ls):
    print("Level ", l_idx)
    data_args0 = initLevel(l, ls[-1])
    data_args1 = initLevel(l-1, ls[-1])

    welford_var = WelfordsVariance3((data_args0["ny"],data_args0["nx"]))
    welford_diffvar = WelfordsVariance3((data_args0["ny"],data_args0["nx"]))

    for i in range(N_var):
        print("Level ", l_idx, ", Sample ", i)

        # Perturbation sampling
        wind = wind_sample(KLSampler, T=T, wind_weight=wind_weight, wind_speed=0.0)

        ## Fine sim
        gpu_ctx = Common.CUDAContext()
        sim0 = CDKLM16.CDKLM16(gpu_ctx, **data_args0, wind=wind)
        sim0.step(T)

        eta0, hu0, hv0 = sim0.download(interior_domain_only=True)
        welford_var.update(eta0, hu0, hv0)

        sim0.cleanUp()
        del gpu_ctx

        ## Coarse partner sim
        gpu_ctx = Common.CUDAContext()
        sim1 = CDKLM16.CDKLM16(gpu_ctx, **data_args1, wind=wind)
        sim1.step(T)

        eta1, hu1, hv1 = sim1.download(interior_domain_only=True)
        welford_diffvar.update(eta0 - eta1.repeat(2,0).repeat(2,1), hu0 - hu1.repeat(2,0).repeat(2,1), hv0 - hv1.repeat(2,0).repeat(2,1))

        sim1.cleanUp()
        del gpu_ctx

    vars[l_idx,:] = np.sqrt(np.average(np.array(welford_var.finalize())**2, axis=(1,2)))

    diff_vars[l_idx,:] = np.sqrt(np.average(np.array(welford_diffvar.finalize())**2, axis=(1,2)))

# %%

np.save(output_path+"/vars", vars)
np.save(output_path+"/diff_vars", diff_vars)


# %%
fig, axs = plt.subplots(1,3, figsize=(15,5))

Nxs = (2**np.array(ls))**2
for i in range(3):
    axs[i].loglog(Nxs, vars[:,i], label="$|| Var[u^l] ||_{L^2}$")
    axs[i].loglog(Nxs[1:], diff_vars[1:,i], label="$|| Var[u^l-u^{l-1}] ||_{L^2}$")
    axs[i].set_xlabel("# grid cells")
    axs[i].set_ylabel("variance")
    axs[i].legend(labelcolor="black")

    axs[i].set_xticks(Nxs)
    axs[i].xaxis.grid(True)

    for l_idx, l in enumerate(ls):
        axs[i].annotate(str(l), (Nxs[l_idx], 1e-5), color="black")

axs[0].set_title("eta")
axs[1].set_title("hu")
axs[2].set_title("hv")

plt.savefig(output_path+"/RossbyVarianceLevels.png")
