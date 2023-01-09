# %% [markdown]
# # Multi Resolution Simulation

# %% [markdown]
# ### Classes and modules

# %%

#Import packages we need
import numpy as np

#For plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"

# %% [markdown]
# GPU Ocean-modules:

# %%
from gpuocean.utils import Common, WindStress
from gpuocean.SWEsimulators import CDKLM16

# %%
gpu_ctx = Common.CUDAContext()

# %% [markdown]
# ## Setting-up case with different resolutions
# 
# IC are the bump from the Rossby adjustment case

# %%
def initBump(data_args, dataShape, d_shift=1e6,D=0.5*1e6):

    eta0  = np.zeros(dataShape, dtype=np.float32, order='C')
    hu0   = np.zeros(dataShape, dtype=np.float32, order='C')
    hv0   = np.zeros(dataShape, dtype=np.float32, order='C')

    x_center = data_args["dx"]*(data_args["nx"]+4)*0.5
    y_center = data_args["dy"]*(data_args["ny"]+4)*0.5

    scale = 1e9
    for j in range(data_args["ny"] + 4):
        for i in range(data_args["nx"] + 4):
            x = data_args["dx"]*i - x_center
            y = data_args["dy"]*j - y_center

            d = np.sqrt(x**2 + y**2)
            
            eta0[j, i] += 0.1*(1.0+np.tanh(-(d-d_shift)/D))

    return eta0, hu0, hv0

# %%
def initLevel(l):
    data_args = {}
    data_args["nx"] = 2**l
    data_args["ny"] = 2**l 
    dataShape = (data_args["ny"] + 4, data_args["nx"] + 4)

    data_args["dx"] = 2**(19-l)*100
    data_args["dy"] = 2**(19-l)*100

    data_args["dt"] = 0.0
    data_args["g"] = 9.81
    data_args["f"] = 1.2e-4
    data_args["r"] = 0.0
    # data_args["boundary_conditions"] = Common.BoundaryConditions(1,1,1,1)

    data_args["eta0"], data_args["hu0"], data_args["hv0"] = initBump(data_args, dataShape)
    
    H0 = 1000.0
    data_args["H"] = np.ma.array(np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')*H0, mask=False)

    return data_args

# %%
ls = [6, 7, 8, 9, 10]

# %% [markdown]
# ### Perturbation from wind direction
# 
# Wind field is faded out towards the walls. 
# *So far only one random parameter but Matern field of wind intended.*

# %%
def wind_bump(data_args, sig = 1e+14):
    dataShape = (data_args["ny"] + 4, data_args["nx"] + 4)
    w = np.zeros(dataShape, dtype=np.float32, order='C')

    x_center = data_args["dx"]*(data_args["nx"]+4)*0.5
    y_center = data_args["dy"]*(data_args["ny"]+4)*0.5

    for j in range(data_args["ny"] + 4):
        for i in range(data_args["nx"] + 4):
            x = data_args["dx"]*i - x_center
            y = data_args["dy"]*j - y_center

            d = x**2 + y**2
            
            w[j, i] = np.exp(-1/2*d/sig)    
    
    return w


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
N_var = 1000

vars = np.zeros((len(ls), 3))
diff_vars = np.zeros((len(ls), 3))

for l_idx, l in enumerate(ls):
    print("Level ", l_idx)
    data_args0 = initLevel(l)
    wind_weight = wind_bump(data_args0)

    data_args1 = initLevel(l-1)

    # Storage allocation
    est_mean = np.zeros((3,data_args0["ny"],data_args0["nx"]))

    samples0 = np.zeros((3,data_args0["ny"],data_args0["nx"],N_var)) 
    samples1 = np.zeros((3,data_args1["ny"],data_args1["nx"],N_var)) 

    for i in range(N_var):
        print("Sample ", i)

        # Perturbation sampling
        wind_degree = np.deg2rad(np.random.uniform(0,360))
        wind_speed  = 10
        data_args0["wind"] = WindStress.WindStress(t=[0], wind_u=[np.array(wind_weight*[[wind_speed*np.sin(wind_degree)]], dtype=np.float32)], wind_v=[np.array(wind_weight*[[wind_speed*np.cos(wind_degree)]], dtype=np.float32)])

        ## Fine sim
        gpu_ctx = Common.CUDAContext()
        sim0 = CDKLM16.CDKLM16(gpu_ctx, **data_args0)
        sim0.step(250000)

        eta, hu, hv = sim0.download(interior_domain_only=True)
        samples0[0,:,:,i] = eta
        samples0[1,:,:,i] = hu
        samples0[2,:,:,i] = hv

        sim0.cleanUp()
        del gpu_ctx

        ## Coarse partner sim
        data_args1["wind"] = WindStress.WindStress(t=[0], wind_u=[np.array(wind_weight*[[wind_speed*np.sin(wind_degree)]], dtype=np.float32)], wind_v=[np.array(wind_weight*[[wind_speed*np.cos(wind_degree)]], dtype=np.float32)])

        gpu_ctx = Common.CUDAContext()
        sim1 = CDKLM16.CDKLM16(gpu_ctx, **data_args1)
        sim1.step(250000)

        eta, hu, hv = sim1.download(interior_domain_only=True)
        samples1[0,:,:,i] = eta
        samples1[1,:,:,i] = hu
        samples1[2,:,:,i] = hv

        sim1.cleanUp()
        del gpu_ctx

    for e in range(3):
        welford_var = WelfordsVariance(samples0[e,:,:,0].shape)
        for s in range(samples0.shape[-1]):
            welford_var.update(samples0[e,:,:,s])
        vars[l_idx,e] = np.sqrt(np.average(welford_var.finalize()[1]**2))

    for e in range(3):
        welford_var = WelfordsVariance(samples0[e,:,:,0].shape)
        for s in range(samples0.shape[-1]):
            welford_var.update(samples0[e,:,:,s]-samples1[e,:,:,s].repeat(2,0).repeat(2,1))
        diff_vars[l_idx,e] = np.sqrt(np.average(welford_var.finalize()[1]**2))

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

plt.savefig("RossbyVarianceLevels.png")
