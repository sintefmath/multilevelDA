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
timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common, WindStress
from gpuocean.SWEsimulators import CDKLM16

# %% 
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.RossbyInit import *

# %%
gpu_ctx = Common.CUDAContext()

# %% [markdown]
# ## Setting-up case with different resolutions
# 
# IC are the bump from the Rossby adjustment case

# %%
ls = [6, 7, 8, 9, 10]

# %% [markdown]
# ### Perturbation from wind direction
# 
# Wind field is faded out towards the walls. 
# *So far only one random parameter but Matern field of wind intended.*

# %%
def wind_bump(ny, nx, sig = None):
    dataShape = (ny, nx )
    w = np.zeros(dataShape, dtype=np.float32, order='C')

    x_center = 0.5*nx
    y_center = 0.5*ny

    if sig is None:
        sig = nx**2/15

    for j in range(ny):
        for i in range(nx):
            x = i - x_center
            y = j - y_center

            d = x**2 + y**2
            
            w[j, i] = np.exp(-1/2*d/sig)    
    
    return w

wind_N = 100
wind_weight = wind_bump(wind_N,wind_N)


# %% 
def KL_perturbations(t_splits, KL_N):
    # Sampling random field based on Karhunen-Loeve expansions
    
    # t_splits (int) - number of how many KL-fields are generated

    # Output: size=(t_splits, N, N) with t_splits-times a KL-field

    KL_DECAY=1.05
    KL_SCALING=0.15

    KL_fields = np.zeros((t_splits,KL_N,KL_N))

    rns = np.random.normal(size=(10,10,t_splits))

    for n in range(1, rns.shape[1]+1):
        for m in range(1, rns.shape[0]+1):
            KL_fields += np.tile(KL_SCALING * m**(-KL_DECAY) * n**(-KL_DECAY) * np.outer(np.sin(m*np.pi*np.linspace(0,1,KL_N)), np.sin(n*np.pi*np.linspace(0,1,KL_N))), (t_splits,1,1)) * rns[m-1, n-1][:,np.newaxis,np.newaxis]

    return KL_fields

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
N_var = 1000

vars = np.zeros((len(ls), 3))
diff_vars = np.zeros((len(ls), 3))

for l_idx, l in enumerate(ls):
    print("Level ", l_idx)
    data_args0 = initLevel(l, ls[-1])
    data_args1 = initLevel(l-1, ls[-1])

    welford_var = WelfordsVariance3((data_args0["ny"],data_args0["nx"]))
    welford_diffvar = WelfordsVariance3((data_args0["ny"],data_args0["nx"]))

    for i in range(N_var):
        print("Sample ", i)

        # Perturbation sampling
        t_splits = 251

        wind_degree = np.deg2rad(np.random.uniform(0,360))
        wind_speed  = 5.0

        init_wind_u = wind_speed * np.sin(wind_degree) * np.ones((wind_N,wind_N))
        init_wind_v = wind_speed * np.cos(wind_degree) * np.ones((wind_N,wind_N))

        KL_fields_u = KL_perturbations(t_splits, wind_N)
        KL_fields_v = KL_perturbations(t_splits, wind_N)

        wind_u = np.repeat(init_wind_u[np.newaxis,:,:], t_splits, axis=0) + np.cumsum(KL_fields_u, axis=0)
        wind_v = np.repeat(init_wind_v[np.newaxis,:,:], t_splits, axis=0) + np.cumsum(KL_fields_v, axis=0)

        wind_u = wind_u *np.repeat(wind_weight[np.newaxis,:,:], t_splits, axis=0)
        wind_v = wind_v *np.repeat(wind_weight[np.newaxis,:,:], t_splits, axis=0)

        ts = np.linspace(0,250000,t_splits)

        wind = WindStress.WindStress(t=ts, wind_u=wind_u.astype(np.float32), wind_v=wind_v.astype(np.float32))

        ## Fine sim
        gpu_ctx = Common.CUDAContext()
        sim0 = CDKLM16.CDKLM16(gpu_ctx, **data_args0, wind=wind)
        sim0.step(250000)

        eta0, hu0, hv0 = sim0.download(interior_domain_only=True)
        welford_var.update(eta0, hu0, hv0)

        sim0.cleanUp()
        del gpu_ctx

        ## Coarse partner sim
        gpu_ctx = Common.CUDAContext()
        sim1 = CDKLM16.CDKLM16(gpu_ctx, **data_args1, wind=wind)
        sim1.step(250000)

        eta1, hu1, hv1 = sim1.download(interior_domain_only=True)
        welford_diffvar.update(eta0 - eta1.repeat(2,0).repeat(2,1), hu0 - hu1.repeat(2,0).repeat(2,1), hv0 - hv1.repeat(2,0).repeat(2,1))

        sim1.cleanUp()
        del gpu_ctx

    vars[l_idx,:] = np.sqrt(np.average(np.array(welford_var.finalize())**2, axis=(1,2)))

    diff_vars[l_idx,:] = np.sqrt(np.average(np.array(welford_diffvar.finalize())**2, axis=(1,2)))

# %%

np.save("Rossby-vars-"+timestamp, vars)
np.save("Rossby-diff_vars-"+timestamp, diff_vars)


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

plt.savefig("RossbyVarianceLevels-"+timestamp+".png")
