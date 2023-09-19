# %% [markdown]
# # Variance-level plots

# %% [markdown]
# ### Classes and modules

# %%
#Import packages we need
import numpy as np
import copy
import sys, os


# %% [markdown]
# Utils

# %%
ls = [6, 7, 8]

# %%
from gpuocean.utils import Common
gpu_ctx = Common.CUDAContext()

from gpuocean.utils import DoubleJetCase

args_list = []
init_list = []

for l in ls:
    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx, DoubleJetCase.DoubleJetPerturbationType.SteadyState, ny=2**l, nx=2**(l+1))
    doubleJetCase_args, doubleJetCase_init, _ = doubleJetCase.getInitConditions()

    args_list.append(doubleJetCase_args)
    init_list.append(doubleJetCase_init)

# %% [markdown]
# Test

# %%
source_path = "/cluster/home/floribei/havvarsel/multilevelDA/scripts/VarianceLevelsDA/2023-09-18T16_55_34"

# %%
Ts = [3*24*3600, 4*24*3600, 5*24*3600, 6*24*3600, 7*24*3600, 8*24*3600, 9*24*3600, 10*24*3600]

# %%
states = [[np.load(source_path+"/SLensemble_"+str(t)+"_"+str(l_idx)+".npy") for l_idx in range(len(ls))] for t in Ts ]

# %%
def g_functional(SL_ensemble):
    """
    L_g functional as in notation of Kjetil's PhD thesis.
    This should be the functional that is under investigation for the variance level plot

    Returns a ndarray of same size as SL_ensemble (3, ny, nx, Ne)
    """
    return (SL_ensemble - np.mean(SL_ensemble, axis=-1)[:,:,:,np.newaxis])**2
    # return SL_ensemble


def L2norm(field, lvl_grid_args):
    """
    integral_D(f dx)
    where D are uniform finite volumes

    Input:
    field           - ndarray of shape (3,ny,nx,..)
    lvl_grid_args   - dict with nx, ny and dx, dy information

    Output:
    L2norm          - ndarray of shape (3,...)
    """
    # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
    return np.sqrt(np.sum((field)**2 * lvl_grid_args["dx"]*lvl_grid_args["dy"], axis=(1,2)))


def L1norm(field, lvl_grid_args):
    """
    integral_D(f dx)
    where D are uniform finite volumes

    Input:
    field           - ndarray of shape (3,ny,nx,..)
    lvl_grid_args   - dict with nx, ny and dx, dy information

    Output:
    L1norm          - ndarray of shape (3,...)
    """
    # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
    return np.sum(np.abs(field) * lvl_grid_args["dx"]*lvl_grid_args["dy"], axis=(1,2))


def Enorm(field, lvl_grid_args):
    """
    integral_D(f dx)
    where D are uniform finite volumes

    Input:
    field           - ndarray of shape (3,ny,nx,..)
    lvl_grid_args   - dict with nx, ny and dx, dy information

    Output:
    L1norm          - ndarray of shape (3,...)
    """
    # assert field.shape[1:3] == (lvl_grid_args["ny"], lvl_grid_args["nx"]), "field has wrong resolution"
    return np.mean(field, axis=(1,2))


norm = L2norm

# %%
sys.path.insert(0, os.path.abspath(os.path.join(os.path.realpath(os.path.dirname(__file__)), '../')))
from utils.DoubleJetPlot import *
for t_idx, T in enumerate(Ts):
    for l_idx in range(len(ls)):
        fig, axs = imshow3(np.mean(states[t_idx][l_idx], axis=-1))
        plt.savefig(source_path+"/mean_t"+str(T)+"_l"+str(l_idx)+".pdf", bbox_inches="tight")
    plt.close("all")


# %%
vars_listTs = []
diff_vars_listTs = []

for t_idx in range(len(Ts)):
    vars_list = []
    diff_vars_list = []

    center_vars_list = []
    center_diff_vars_list = []
    for l_idx in range(len(ls)):
        vars_list.append(norm(np.var(g_functional(states[t_idx][l_idx]), axis=-1), args_list[l_idx]))

        if l_idx > 0:
            diff_vars_list.append(norm(np.var(g_functional(states[t_idx][l_idx]) - g_functional(states[t_idx][l_idx-1]).repeat(2,1).repeat(2,2), axis=-1), args_list[l_idx]))
            
    vars_listTs.append(vars_list)
    diff_vars_listTs.append(diff_vars_list)

# %%
for t_idx, T in enumerate(Ts):
    varsT = [vars_listTs[t_idx][l_idx] for l_idx in range(len(ls))]
    np.save(source_path+"/vars_"+str(T), np.array(varsT))

    diff_varsT = [diff_vars_listTs[t_idx][l_idx] for l_idx in range(len(ls)-1)]
    np.save(source_path+"/diff_vars_"+str(T), np.array(diff_varsT))

# %% [markdown]
# Better format for tikz 

# %%
Nxs = np.array([nx*ny for nx, ny in zip([da["nx"] for da in args_list], [da["ny"] for da in args_list])])

# %%
vars_file = np.array(vars_listTs)[0]
for l_idx in range(1,len(ls)):
    vars_file = np.c_[vars_file, np.array(vars_listTs)[l_idx]]

# %%
diff_vars_file = np.array(diff_vars_listTs)[0]
for l_idx in range(1,len(ls)):
    diff_vars_file = np.c_[diff_vars_file, np.array(diff_vars_listTs)[l_idx]]

# %%
np.savetxt(source_path+"/vars.txt", np.c_[Nxs, vars_file],
           delimiter=",", fmt="%9.9f")

np.savetxt(source_path+"/diff_vars.txt", np.c_[Nxs[1:], diff_vars_file],
           delimiter=",", fmt="%9.9f")