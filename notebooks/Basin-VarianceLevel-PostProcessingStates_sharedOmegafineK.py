# %% [markdown]
# # Variance-level plots

# %% [markdown]
# ### Classes and modules

# %%
#Import packages we need
import numpy as np
import copy


# %% [markdown]
# Utils
# %%
def initGridSpecs(l):
    data_args = {}
    data_args["nx"] = 2**l
    data_args["ny"] = 2**(l+1) 

    data_args["dx"] = 2**(9-l)*160
    data_args["dy"] = 2**(9-l)*160

    return data_args

# %%
ls = [6, 7, 8, 9, 10]

# %%
grid_args_list = []

for l in ls:
    lvl_grid_args = initGridSpecs(l)
    grid_args_list.append( {
        "nx": lvl_grid_args["nx"],
        "ny": lvl_grid_args["ny"],
        "dx": lvl_grid_args["dx"],
        "dy": lvl_grid_args["dy"],
        } )

# %% [markdown]
# Test

# %%
source_path = "/cluster/home/floribei/havvarsel/multilevelDA/scripts/VarianceLevelsDA/Basin/2023-09-14T18_10_00"

# %%
states = [[np.load(source_path+"/SLensemble_"+str(l_idx)+"_"+str(t)+".npy") for l_idx in range(len(ls))] for t in [0, 15*60, 3600, 6*3600, 12*3600] ]

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
Ts = [0, 15*60, 3600, 6*3600, 12*3600]

vars_listTs = []
diff_vars_listTs = []

center_vars_listTs = []
center_diff_vars_listTs = []
        

for t_idx in range(len(Ts)):
    vars_list = []
    diff_vars_list = []

    center_vars_list = []
    center_diff_vars_list = []
    for l_idx in range(len(ls)):
        vars_list.append(norm(np.var(g_functional(states[t_idx][l_idx]), axis=-1), grid_args_list[l_idx]))

        center_N = int(grid_args_list[l_idx]["nx"]/4)
        center_x = int(grid_args_list[l_idx]["nx"]/2)
        center_y = int(grid_args_list[l_idx]["ny"]/2)
            
        center_vars_list.append( norm(np.var(g_functional(states[t_idx][l_idx])[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), grid_args_list[l_idx]) )
        
        if l_idx > 0:
            diff_vars_list.append(norm(np.var(g_functional(states[t_idx][l_idx]) - g_functional(states[t_idx][l_idx-1]).repeat(2,1).repeat(2,2), axis=-1), grid_args_list[l_idx]))
            center_diff_vars_list.append(norm(np.var((g_functional(states[t_idx][l_idx]) - g_functional(states[t_idx][l_idx-1]).repeat(2,1).repeat(2,2))[:, center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N,:], axis=-1), grid_args_list[l_idx]))
    vars_listTs.append(vars_list)
    diff_vars_listTs.append(diff_vars_list)

    center_vars_listTs.append(center_vars_list)
    center_diff_vars_listTs.append(center_diff_vars_list)

# %%
for t_idx, T in enumerate(Ts):
    varsT = [vars_listTs[t_idx][l_idx] for l_idx in range(len(ls))]
    np.save(source_path+"/vars_"+str(T), np.array(varsT))

    diff_varsT = [diff_vars_listTs[t_idx][l_idx] for l_idx in range(len(ls)-1)]
    np.save(source_path+"/diff_vars_"+str(T), np.array(diff_varsT))

    center_varsT = [center_vars_listTs[t_idx][l_idx] for l_idx in range(len(ls))]
    np.save(source_path+"/center_vars_"+str(T), np.array(center_varsT))

    center_diff_varsT = [center_diff_vars_listTs[t_idx][l_idx] for l_idx in range(len(ls)-1)]
    np.save(source_path+"/center_diff_vars_"+str(T), np.array(center_diff_varsT))

# %%
Nxs = np.array([nx*ny for nx, ny in zip([da["nx"] for da in grid_args_list], [da["ny"] for da in grid_args_list])])

# %%
center_vars_file = np.array(center_vars_listTs)[0]
for l_idx in range(1,len(ls)):
    center_vars_file = np.c_[center_vars_file, np.array(center_vars_listTs)[l_idx]]

# %%
center_diff_vars_file = np.array(center_diff_vars_listTs)[0]
for l_idx in range(1,len(ls)):
    center_diff_vars_file = np.c_[center_diff_vars_file, np.array(center_diff_vars_listTs)[l_idx]]

# %%
np.savetxt(source_path+"/center_vars.txt", np.c_[Nxs, center_vars_file],
           delimiter=",", fmt="%9.9f")

np.savetxt(source_path+"/center_diff_vars.txt", np.c_[Nxs[1:], center_diff_vars_file],
           delimiter=",", fmt="%9.9f")


