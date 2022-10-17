# Variance vs level plot!

## Import packages we need
import numpy as np
from IPython.display import display
import copy

#For plotting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["image.origin"] = "lower"

from gpuocean.utils import Common
from gpuocean.utils import NetCDFInitialization
from gpuocean.SWEsimulators import CDKLM16, CDKLM16pair

nk800_url  = ["/sintef/data/NorKyst800/ocean_his.an.20190716.nc"]

data_args = NetCDFInitialization.removeMetadata(NetCDFInitialization.getInitialConditionsNorKystCases(nk800_url, "lofoten", download_data=False, norkyst_data=False))
data_args.keys()

sim_args = {
    "dt": 0.0,
    "write_netcdf":False,
    }

loc = [[50,50],[250,450]]
scale = 1.5
# NOTE: We use constant 
# Could also be lists of length L

# Sample size
N = 5

Lmax = 4

Vars = np.zeros((Lmax,3))
VarDiffs = np.zeros((Lmax,3))

for L in range(1,Lmax+1):
    gpu_ctxs = []
    for i in range(L+1):
        gpu_ctxs.append(Common.CUDAContext())

    slave_gpu_ctxs = []
    for i in range(L+1):
        slave_gpu_ctxs.append(Common.CUDAContext())

    ##################
    ## Mean Estimators!!
    for n in range(N):
        # Set up new simulations
        sim = CDKLM16.CDKLM16(gpu_ctxs[0], **sim_args, **data_args)
        slave = CDKLM16.CDKLM16(slave_gpu_ctxs[0], **sim_args, **data_args)   

        # Initialise L levels
        child = sim
        slave_child = slave
        for l in range(1,L+1):
            child.give_birth(gpu_ctxs[l], loc, scale)
            child = child.children[0]
            if l < L:
                slave_child.give_birth(slave_gpu_ctxs[l], loc, scale)
                slave_child = slave_child.children[0]

        # pair slave and step ahead in time
        sim_pair = CDKLM16pair.CDKLM16pair(sim, slave, small_scale_model_error=True, interpolation_factor=21)
        for five in range(5): 
            sim_pair.step(300, apply_stochastic_term=True) # model error after every step()-call
        
        # Finest levels
        level_sim = sim_pair.sim
        slave_level_sim = sim_pair.slave_sim
        for l in range(1,L+1):
            level_sim = level_sim.children[0]
            if l < L:
                slave_level_sim = slave_level_sim.children[0]
        
        eta, hu, hv = level_sim.download()
        slave_level_sim.give_birth(slave_gpu_ctxs[-1], loc, scale)
        slave_eta, slave_hu, slave_hv = slave_level_sim.children[0].download()
        slave_level_sim.kill_child()

        if n==0:
            Ediff = np.ma.array([(eta - slave_eta)/N, (hu - slave_hu)/N, (hv - slave_hv)/N])
        else:
            Ediff += np.ma.array([(eta - slave_eta)/N, (hu - slave_hu)/N, (hv - slave_hv)/N])

        if n==0:
            E = np.ma.array([eta/N, hu/N, hv/N])
        else:
            E += np.ma.array([eta/N, hu/N, hv/N])

    ################
    ## Variance Estimators!!
    for n in range(N):
        # Set up new simulations
        sim = CDKLM16.CDKLM16(gpu_ctxs[0], **sim_args, **data_args)
        slave = CDKLM16.CDKLM16(slave_gpu_ctxs[0], **sim_args, **data_args)   

        # Initialise L levels
        child = sim
        slave_child = slave
        for l in range(1,L+1):
            child.give_birth(gpu_ctxs[l], loc, scale)
            child = child.children[0]
            if l < L:
                slave_child.give_birth(slave_gpu_ctxs[l], loc, scale)
                slave_child = slave_child.children[0]

        # pair slave and step ahead in time
        sim_pair = CDKLM16pair.CDKLM16pair(sim, slave, small_scale_model_error=True, interpolation_factor=21)
        for five in range(5): 
            sim_pair.step(300, apply_stochastic_term=True) # model error after every step()-call
        
        # Finest levels
        level_sim = sim_pair.sim
        slave_level_sim = sim_pair.slave_sim
        for l in range(1,L+1):
            level_sim = level_sim.children[0]
            if l < L:
                slave_level_sim = slave_level_sim.children[0]
        
        eta, hu, hv = level_sim.download()
        slave_level_sim.give_birth(slave_gpu_ctxs[-1], loc, scale)
        slave_eta, slave_hu, slave_hv = slave_level_sim.children[0].download()
        slave_level_sim.kill_child()

        if n == 0:
            VarDiff = np.ma.array([((eta - slave_eta) - Ediff[0])**2/(N-1), ((hu - slave_hu) - Ediff[1])**2/(N-1), ((hv - slave_hv) - Ediff[2])**2/(N-1)])
        else:
            VarDiff += np.ma.array([((eta - slave_eta) - Ediff[0])**2/(N-1), ((hu - slave_hu) - Ediff[1])**2/(N-1), ((hv - slave_hv) - Ediff[2])**2/(N-1)])

        if n==0:
            Var = np.ma.array([(eta - E[0])**2/(N-1), (hu - E[1])**2/(N-1), (hv - E[2])**2/(N-1)])
        else:
            Var += np.ma.array([(eta - E[0])**2/(N-1), (hu - E[1])**2/(N-1), (hv - E[2])**2/(N-1)])

    # Evaluating level
    Vars[L-1] = [np.linalg.norm(field)/np.prod(field.shape) for field in Var]
    VarDiffs[L-1] = [np.linalg.norm(field)/np.prod(field.shape) for field in VarDiff]

    # CleanUp
    for ctx in gpu_ctxs:
        del ctx
    for ctx in slave_gpu_ctxs:
        del ctx

np.savetxt("LevelVariances", Vars)
np.savetxt("LevelVarianceDiffs", VarDiffs)

# label = ["\eta", "hu", "hv"]
# for q in range(3):
#     plt.plot(np.arange(1,Lmax+1),Vars[:,q])
#     plt.plot(np.arange(1,Lmax+1),VarDiffs[:,q])
#     plt.savefig("VarianceLevelPlot_"+label[q]+".png")