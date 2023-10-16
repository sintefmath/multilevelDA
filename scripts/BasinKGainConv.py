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

base_path = "KGainConvergence/Basin/"
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
ls = [6, 7, 8, 9, 10] # The last level is only used for the reference solution

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

center_N = int(args_list[-1]["nx"]/4)
center_x = int(args_list[-1]["nx"]/2)
center_y = int(args_list[-1]["ny"]/2)

# %% [markdown]
# ### Multi-Level Ensemble Sizes 

# %%
source_path = script_path+"/VarianceLevelsDA/Basin/2023-09-14T18_10_00IDUN"

center_vars = np.load(source_path+"/center_vars_21600.npy")[:-1]
center_diff_vars = np.load(source_path+"/center_diff_vars_21600.npy")[:-1]

# %%
work_path = script_path+"/PracticalCost/Basin/2023-06-29T14_54_02"

# %%
def raw2costsEnsemble(filename):
    rawCosts = np.load(filename)
    return np.mean(np.sort(rawCosts), axis=1)

# %%
costsPure = raw2costsEnsemble(work_path+"/costsPureEnsemble.npy")[:-1]
costsPartnered = raw2costsEnsemble(work_path+"/costsPartneredEnsemble.npy")[:-1]


# %% [markdown]
# What we mean with height and depth?
# 
# ML ensembles are defined on an (arbirary) series of levels, 
# where the series has to be dense in the sense that we cannot jump over neighbors
# (Thats a restriction in our code and not the method)
# 
# We usually have the levels [6, 7, 8, 9] in the experiment in this script
# 
# HEIGHT means building the levels as [6, ...] and stack them as high as we want,
# while DEPTH means that we use the levels as [..., 9] and dig as deep as we want
# 
# The full list of levels aka [6, ..., 9] is included in the depth-collection 
# %% 
taus_height = [
    [8e-2, 5.5e-2, 4e-2, 3e-2], # 6 7
    [9e-2, 6e-2, 4e-2, 3e-2]  # 6 7 8 
]

ML_Nes_height= []
ML_works_height = []
for n, num_levels in enumerate(range(2,3+1)):

    analysis = Analysis(args_list[:num_levels], center_vars[:num_levels], center_diff_vars[:num_levels], costsPure[:num_levels], costsPartnered[:num_levels])

    ML_Nes = []
    MLworks = []

    for tau in taus_height[n]:
        ML_Ne = analysis.optimal_Ne(tau=tau)  
        ML_Nes.append( ML_Ne )
        MLworks.append( analysis.work(ML_Ne) )

    ML_works_height.append(MLworks)
    ML_Nes_height.append(ML_Nes)


# %%
taus_depth = [
    [15e-2, 10e-2, 7.5e-2, 5e-2],
    [12.5e-2, 9e-2, 7e-2, 4.5e-2],
    [15e-2, 10e-2, 7.5e-2, 5e-2]
]


ML_Nes_depth = []
ML_works_depth = []
for n, num_levels in enumerate(range(2,4+1)):
    start_l_idx = len(ls)-1-num_levels

    analysis = Analysis(args_list[start_l_idx:-1], center_vars[start_l_idx:], center_diff_vars[start_l_idx:], costsPure[start_l_idx:], costsPartnered[start_l_idx:])

    ML_Nes = []
    MLworks = []

    for tau in taus_depth[n]:
        ML_Ne = analysis.optimal_Ne(tau=tau)  
        ML_Nes.append( ML_Ne )
        MLworks.append( analysis.work(ML_Ne) )

    ML_works_depth.append(MLworks)
    ML_Nes_depth.append(ML_Nes)


# %%
num_SL_shifts = 3 # meaning: SL on coarest level + SL on num_SL_shifts coarser levels

SL_Nes_shifts = []
SL_works_shifts = []
for s in range(num_SL_shifts+1):
    SL_Nes = []
    SL_works = []
    for t in range(len(ML_Nes_depth[-1])):
        SL_Nes.append(int(analysis.work(ML_Nes_depth[-1][t])/analysis.works[-1-s]))
        SL_works.append(analysis.works[-1-s]*SL_Nes[-1])
    SL_Nes_shifts.append(SL_Nes)
    SL_works_shifts.append(SL_works)


# %%
from utils.BasinParameters import * 

# Book keeping
log.write("levels = " + ", ".join([str(l) for l in ls[start_l_idx:-1]])+"\n\n")

for d in range(len(ML_works_depth)):
    log.write("ML_Nes = " + ",".join([str(Ne) for Ne in ML_Nes_depth[d]])+"\n")
log.write("SL_Nes = " + str(SL_Nes) + "\n\n")

log.write("Reference level (one step finer than finest ensemnble level)\n")
log.write("nx = " + str(args_list[-1]["nx"]) + ", ny = " + str(args_list[-1]["ny"])+"\n")
log.write("dx = " + str(args_list[-1]["dx"]) + ", dy = " + str(args_list[-1]["dy"])+"\n")
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


log.write("Statistics\n")
N_exp = 35
log.write("N_exp = " + str(N_exp) + "\n")
log.write("Metric: Variance in the Kalman Gain estimator\n")
log.write("Metric: L2-errror in Kalman Gain estimator\n")


# %%
from utils import VarianceStatistics as VS

os.makedirs(output_path+"/GainFigs", exist_ok=True)
eta_vlim = 0.01
huv_lim = 1

var_eta_vlim = 5e-6
var_huv_vlim = 5e-2













# %% 
# ########################################
# # NEW truth
print("Truth")
# if os.path.isdir(os.path.join(base_path,"tmpTruth")):
#     for f in os.listdir(os.path.join(base_path,"tmpTruth")):
#         os.remove(os.path.join(os.path.join(base_path,"tmpTruth"), f))

# os.system("python "+script_path+"/BasinKGainConvSingle.py -m T")

true_file_names = os.listdir(os.path.join(base_path,"tmpTruth"))
def get_time_from_filename(file_name):
    return int(file_name.split("_")[1].split(".")[0])
sorted_file_names = sorted(true_file_names, key=get_time_from_filename)

true_eta, true_hu, true_hv = np.load(os.path.join(os.path.join(base_path,"tmpTruth"), sorted_file_names[-1])) #block_reduce(np.load(os.path.join("tmpRefGain", f))[:,:,:,0], block_size=(1,2,2), func=np.mean)













# %% 
########################################
# NEW reference
print("Reference")

# if os.path.isdir(os.path.join(base_path,"tmpRefGain")):
#     for f in os.listdir(os.path.join(base_path,"tmpRefGain")):
#         os.remove(os.path.join(os.path.join(base_path,"tmpRefGain"), f))

# os.system("python "+script_path+"/BasinKGainConvSingle.py -m R")

f = os.listdir(os.path.join(base_path,"tmpRefGain"))[0]
refK_eta, refK_hu, refK_hv = np.load(os.path.join(os.path.join(base_path,"tmpRefGain"), f))[:,:,:,0] #block_reduce(np.load(os.path.join("tmpRefGain", f))[:,:,:,0], block_size=(1,2,2), func=np.mean)

fig, axs = imshow3([refK_eta, refK_hu, refK_hv], eta_vlim=eta_vlim, huv_vlim=huv_lim)
fig.savefig(output_path+"/GainFigs/Ref")
plt.close("all")






















# %%
###########################################
print("-----------------------------")
print("Sinlge level")

for s in range(num_SL_shifts+1):


    KvarNorm_SL = np.zeros((len(SL_Nes_shifts[s]),3))
    Kerr_SL = np.zeros((len(SL_Nes_shifts[s]),3))
    Terr_SL = np.zeros((len(SL_Nes_shifts[s]),3))

    for tau_idx, SL_Ne in enumerate(SL_Nes_shifts[s]): 
        print("Ne = ", SL_Ne)

        Kvar_SL = VS.WelfordsVariance3((args_list[-1]["ny"], args_list[-1]["nx"]))

        ###########################################
        # N-LOOP 
        for n in range(N_exp):

            # new SL Ensemble
            print("SL experiment number ", n)

            if os.path.isdir(os.path.join(base_path,"tmpSLGain")):
                for f in os.listdir(os.path.join(base_path,"tmpSLGain")):
                    os.remove(os.path.join(os.path.join(base_path,"tmpSLGain"), f))
            if os.path.isdir(os.path.join(base_path,"tmpSLmean")):
                for f in os.listdir(os.path.join(base_path,"tmpSLmean")):
                    os.remove(os.path.join(os.path.join(base_path,"tmpSLmean"), f))

            
            # RUN EXPERIMENT
            os.system("python BasinKGainConvSingle.py -m SL -L " + str(9-s) + " -Ne " + str(SL_Ne))


            # KGain eval
            if os.path.isdir(os.path.join(base_path,"tmpSLGain")):
                assert len(os.listdir(os.path.join(base_path,"tmpSLGain"))) == 1, "Please remove old files"
                f = os.listdir(os.path.join(base_path,"tmpSLGain"))[0]
                K_eta, K_hu, K_hv = np.load(os.path.join(os.path.join(base_path,"tmpSLGain"), f))[:,:,:,0].repeat(2**(s+1),1).repeat(2**(s+1),2)
                Kvar_SL.update(K_eta, K_hu, K_hv)

                fig, axs = imshow3([K_eta,K_hu,K_hv], eta_vlim=eta_vlim,huv_vlim=huv_lim)
                fig.savefig(output_path+"/GainFigs/SL_shift"+str(s)+"_tau"+str(tau_idx)+"_"+str(n))
                plt.close("all")

                Kerr_SL[tau_idx] += 1/N_exp * np.array([np.linalg.norm(err_field) for err_field in [refK_eta-K_eta, refK_hu-K_hu, refK_hv-K_hv] ])

                os.remove(os.path.join(os.path.join(base_path,"tmpSLGain"), f))

            # State eval
            if os.path.isdir(os.path.join(base_path,"tmpSLmean")):
                assert len(os.listdir(os.path.join(base_path,"tmpSLmean"))) == 1, "Please remove old files"
                f = os.listdir(os.path.join(base_path,"tmpSLmean"))[0]
                eta, hu, hv = np.load(os.path.join(os.path.join(base_path,"tmpSLmean"), f)).repeat(2**(s+1),1).repeat(2**(s+1),2)

                Terr_SL[tau_idx] += 1/N_exp * np.array([np.linalg.norm(err_field[center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N]) for err_field in [true_eta-eta, true_hu-hu, true_hv-hv] ])
                
                os.remove(os.path.join(os.path.join(base_path,"tmpSLmean"), f))


        # Saving results
        eta_varSL, hu_varSL, hv_varSL = Kvar_SL.finalize()
        fig, axs = imshow3var([eta_varSL, hu_varSL, hv_varSL], eta_vlim=var_eta_vlim, huv_vlim=var_huv_vlim)
        fig.savefig(output_path+"/GainFigs/SLvar_shift"+str(s)+"_tau"+str(tau_idx))
        plt.close("all")

        KvarNorm_SL[tau_idx] = [np.linalg.norm(var_field) for var_field in [eta_varSL, hu_varSL, hv_varSL] ]
        
        np.savetxt(output_path+"/Kvar_SL"+str(s)+".txt", np.c_[SL_works_shifts[s],KvarNorm_SL])
        np.savetxt(output_path+"/KerrSLGain"+str(s)+".txt", np.c_[SL_works_shifts[s],Kerr_SL])
        np.savetxt(output_path+"/TrmseSL"+str(s)+".txt", np.c_[SL_works_shifts[s],Terr_SL])











# %%
print("-----------------------------")
print("Multi level high")

os.makedirs(output_path+"/runningResults", exist_ok=True)

for d in range(len(taus_height)):
    print("ML with " + str(len(ML_Nes_height[d][0])) + "levels height")
    num_levels = len(ML_Nes_height[d][0])

    KvarNorm_ML = np.zeros((len(ML_Nes_height[d]),3))
    Kerr_ML = np.zeros((len(ML_Nes_height[d]),3))
    Terr_ML = np.zeros((len(ML_Nes_height[d]),3))

    ###########################################
    # tau-LOOP
    for tau_idx, ML_Nes in enumerate(ML_Nes_height[d]): 
        print("-----------------------------")
        print("Ne = ", ML_Nes)

        Kvar_ML = VS.WelfordsVariance3((args_list[-1]["ny"], args_list[-1]["nx"]))

        runningKgains = np.zeros((N_exp,3,args_list[-1]["ny"], args_list[-1]["nx"]))
        runningKerrs = np.zeros((N_exp,3))
        runningTerrs = np.zeros((N_exp,3))

        ###########################################
        # N-LOOP 
        n = 0
        while (n < N_exp):
            
            try:
                signal.alarm(20*60)

                # NEW ML Ensemble
                print("ML experiment number ", n)

                if os.path.isdir(os.path.join(base_path,"tmpMLGain")):
                    for f in os.listdir(os.path.join(base_path,"tmpMLGain")):
                        os.remove(os.path.join(os.path.join(base_path,"tmpMLGain"), f))
                if os.path.isdir(os.path.join(base_path,"tmpMLmean")):
                    for f in os.listdir(os.path.join(base_path,"tmpMLmean")):
                        os.remove(os.path.join(os.path.join(base_path,"tmpMLmean"), f))

                MLlog = open(os.path.join(base_path, "logMLEnKF.txt"), "a")
                MLlog.write("ML height " +str(d)+ ", work " +str(tau_idx)+ ", Experiment "+str(n)+"\n")
                MLlog.close()


                # RUN EXPERIMENT
                os.system("python BasinKGainConvSingle.py -m ML -ls " + " ".join(str(l) for l in ls[:num_levels]) + " -Ne " + " ".join(str(Ne) for Ne in ML_Nes))


                # KGain eval
                if os.path.isdir(os.path.join(base_path,"tmpMLGain")):
                    assert len(os.listdir(os.path.join(base_path,"tmpMLGain"))) == 1, "Please remove old files"
                    f = os.listdir(os.path.join(base_path,"tmpMLGain"))[0]
                    K_eta, K_hu, K_hv = np.load(os.path.join(os.path.join(base_path,"tmpMLGain"), f))[:,:,:,0].repeat(2**(len(ls)-num_levels),1).repeat(2**(len(ls)-num_levels),2)
                    Kvar_ML.update(K_eta, K_hu, K_hv)

                    runningKgains[n] = np.array([K_eta, K_hu, K_hv])
                    np.save(output_path+"/runningResults/Kgains_height"+str(d)+"_tau"+str(tau_idx)+".npy", runningKgains)

                    fig, axs = imshow3([K_eta,K_hu,K_hv], eta_vlim=eta_vlim,huv_vlim=huv_lim)
                    fig.savefig(output_path+"/GainFigs/ML_height"+str(d)+"_work"+str(tau_idx)+"_"+str(n))
                    plt.close("all")

                    currentKerr = np.array([np.linalg.norm(err_field) for err_field in [refK_eta-K_eta, refK_hu-K_hu, refK_hv-K_hv] ])
                    Kerr_ML[tau_idx] += 1/N_exp * currentKerr

                    runningKerrs[n] = currentKerr
                    np.savetxt(output_path+"/runningResults/Kerrs_height"+str(d)+"_tau"+str(tau_idx)+".txt", runningKerrs)

                    os.remove(os.path.join(os.path.join(base_path,"tmpMLGain"), f))
        
                # State eval
                if os.path.isdir(os.path.join(base_path,"tmpMLmean")):
                    assert len(os.listdir(os.path.join(base_path,"tmpMLmean"))) == 1, "Please remove old files"
                    f = os.listdir(os.path.join(base_path,"tmpMLmean"))[0]
                    eta, hu, hv = np.load(os.path.join(os.path.join(base_path,"tmpMLmean"), f)).repeat(2**(len(ls)-num_levels),1).repeat(2**(len(ls)-num_levels),2)

                    currentTerr = np.array([np.linalg.norm(err_field[center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N]) for err_field in [true_eta-eta, true_hu-hu, true_hv-hv] ])
                    Terr_ML[tau_idx] += 1/N_exp * currentTerr
                    
                    runningTerrs[n] = currentTerr
                    np.savetxt(output_path+"/runningResults/Terrs_height"+str(d)+"_tau"+str(tau_idx)+".txt", runningTerrs)

                    os.remove(os.path.join(os.path.join(base_path,"tmpMLmean"), f))

                n = n+1
                signal.alarm(0)

            except Exception as exc:
                print(exc)
                signal.alarm(0)
                pass

            # end: n-loop 
        
        eta_varML, hu_varML, hv_varML = Kvar_ML.finalize()
        fig, axs = imshow3var([eta_varML, hu_varML, hv_varML], eta_vlim=var_eta_vlim, huv_vlim=var_huv_vlim)
        fig.savefig(output_path+"/GainFigs/MLvar_height"+str(d)+"_work"+str(tau_idx))
        plt.close("all")

        KvarNorm_ML[tau_idx] = [np.linalg.norm(var_field) for var_field in [eta_varML, hu_varML, hv_varML] ]

        # Saving results (backup after every tau)
        np.savetxt(output_path+"/Kvar_HighML"+str(len(ML_Nes_height[d][0]))+".txt", np.c_[ML_works_height[d],KvarNorm_ML])
        np.savetxt(output_path+"/KerrHighML"+str(len(ML_Nes_height[d][0]))+"Gain.txt", np.c_[ML_works_height[d],Kerr_ML])
        np.savetxt(output_path+"/TrmseHighML"+str(len(ML_Nes_depth[d][0]))+".txt", np.c_[ML_works_height[d],Terr_ML])

    # end: tau-loop


















# %%
print("-----------------------------")
print("Multi level deep")

os.makedirs(output_path+"/runningResults", exist_ok=True)

for d in range(len(taus_depth)):
    print("ML with " + str(len(ML_Nes_depth[d][0])) + "levels depth")
    start_l_idx = len(ls)-1-len(ML_Nes_depth[d][0])

    KvarNorm_ML = np.zeros((len(ML_Nes_depth[d]),3))
    Kerr_ML = np.zeros((len(ML_Nes_depth[d]),3))
    Terr_ML = np.zeros((len(ML_Nes_depth[d]),3))

    ###########################################
    # tau-LOOP
    for tau_idx, ML_Nes in enumerate(ML_Nes_depth[d]): 
        print("-----------------------------")
        print("Ne = ", ML_Nes)

        Kvar_ML = VS.WelfordsVariance3((args_list[-1]["ny"], args_list[-1]["nx"]))

        runningKgains = np.zeros((N_exp,3,args_list[-1]["ny"], args_list[-1]["nx"]))
        runningKerrs = np.zeros((N_exp,3))
        runningTerrs = np.zeros((N_exp,3))

        ###########################################
        # N-LOOP 
        n = 0
        while (n < N_exp):
            
            try:
                signal.alarm(20*60)

                # NEW ML Ensemble
                print("ML experiment number ", n)

                if os.path.isdir(os.path.join(base_path,"tmpMLGain")):
                    for f in os.listdir(os.path.join(base_path,"tmpMLGain")):
                        os.remove(os.path.join(os.path.join(base_path,"tmpMLGain"), f))
                if os.path.isdir(os.path.join(base_path,"tmpMLmean")):
                    for f in os.listdir(os.path.join(base_path,"tmpMLmean")):
                        os.remove(os.path.join(os.path.join(base_path,"tmpMLmean"), f))

                MLlog = open(os.path.join(base_path, "logMLEnKF.txt"), "a")
                MLlog.write("ML depth " +str(d)+ ", work " +str(tau_idx)+ ", Experiment "+str(n)+"\n")
                MLlog.close()

                os.system("python BasinKGainConvSingle.py -m ML -ls " + " ".join(str(l) for l in ls[start_l_idx:-1]) + " -Ne " + " ".join(str(Ne) for Ne in ML_Nes))


                if os.path.isdir(os.path.join(base_path,"tmpMLGain")):
                    assert len(os.listdir(os.path.join(base_path,"tmpMLGain"))) == 1, "Please remove old files"
                    f = os.listdir(os.path.join(base_path,"tmpMLGain"))[0]
                    K_eta, K_hu, K_hv = np.load(os.path.join(os.path.join(base_path,"tmpMLGain"), f))[:,:,:,0].repeat(2,1).repeat(2,2)
                    Kvar_ML.update(K_eta, K_hu, K_hv)

                    runningKgains[n] = np.array([K_eta, K_hu, K_hv])
                    np.save(output_path+"/runningResults/Kgains_depth"+str(d)+"_tau"+str(tau_idx)+".npy", runningKgains)

                    fig, axs = imshow3([K_eta,K_hu,K_hv], eta_vlim=eta_vlim,huv_vlim=huv_lim)
                    fig.savefig(output_path+"/GainFigs/ML_depth"+str(d)+"_work"+str(tau_idx)+"_"+str(n))
                    plt.close("all")

                    currentKerr = np.array([np.linalg.norm(err_field) for err_field in [refK_eta-K_eta, refK_hu-K_hu, refK_hv-K_hv] ])
                    Kerr_ML[tau_idx] += 1/N_exp * currentKerr

                    runningKerrs[n] = currentKerr
                    np.savetxt(output_path+"/runningResults/Kerrs_depth"+str(d)+"_tau"+str(tau_idx)+".txt", runningKerrs)

                    os.remove(os.path.join(os.path.join(base_path,"tmpMLGain"), f))
        
                if os.path.isdir(os.path.join(base_path,"tmpMLmean")):
                    assert len(os.listdir(os.path.join(base_path,"tmpMLmean"))) == 1, "Please remove old files"
                    f = os.listdir(os.path.join(base_path,"tmpMLmean"))[0]
                    eta, hu, hv = np.load(os.path.join(os.path.join(base_path,"tmpMLmean"), f)).repeat(2,1).repeat(2,2)

                    currentTerr = np.array([np.linalg.norm(err_field[center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N]) for err_field in [true_eta-eta, true_hu-hu, true_hv-hv] ])
                    Terr_ML[tau_idx] += 1/N_exp * currentTerr
                    
                    runningTerrs[n] = currentTerr
                    np.savetxt(output_path+"/runningResults/Terrs_depth"+str(d)+"_tau"+str(tau_idx)+".txt", runningTerrs)

                    os.remove(os.path.join(os.path.join(base_path,"tmpMLmean"), f))

                n = n+1
                signal.alarm(0)

            except Exception as exc:
                print(exc)
                signal.alarm(0)
                pass

            # end: n-loop 
        
        eta_varML, hu_varML, hv_varML = Kvar_ML.finalize()
        fig, axs = imshow3var([eta_varML, hu_varML, hv_varML], eta_vlim=var_eta_vlim, huv_vlim=var_huv_vlim)
        fig.savefig(output_path+"/GainFigs/MLvar"+str(tau_idx))
        plt.close("all")

        KvarNorm_ML[tau_idx] = [np.linalg.norm(var_field) for var_field in [eta_varML, hu_varML, hv_varML] ]

        # Saving results (backup after every tau)
        np.savetxt(output_path+"/Kvar_ML"+str(len(ML_Nes_depth[d][0]))+".txt", np.c_[ML_works_depth[d],KvarNorm_ML])
        np.savetxt(output_path+"/KerrML"+str(len(ML_Nes_depth[d][0]))+"Gain.txt", np.c_[ML_works_depth[d],Kerr_ML])
        np.savetxt(output_path+"/TrmseML"+str(len(ML_Nes_depth[d][0]))+".txt", np.c_[ML_works_depth[d],Terr_ML])

    # end: tau-loop












# %%
print("-----------------------------")
print("Monte Carlo")

os.makedirs(output_path+"/runningResults", exist_ok=True)

d = -1

Terr_ML = np.zeros((len(ML_Nes_depth[d]),3))

###########################################
# tau-LOOP
for tau_idx, ML_Nes in enumerate(ML_Nes_depth[d]): 
    print("-----------------------------")
    print("Ne = ", ML_Nes)

    runningTerrs = np.zeros((N_exp,3))

    ###########################################
    # N-LOOP 
    n = 0
    while (n < N_exp):
        
        try:
            signal.alarm(15*60)

            # NEW ML Ensemble
            print("MC experiment number ", n)

            if os.path.isdir(os.path.join(base_path,"tmpMLmean")):
                for f in os.listdir(os.path.join(base_path,"tmpMLmean")):
                    os.remove(os.path.join(os.path.join(base_path,"tmpMLmean"), f))


            os.system("python BasinKGainConvSingle.py -m MC -Ne "+" ".join(str(Ne) for Ne in ML_Nes))

    
            if os.path.isdir(os.path.join(base_path,"tmpMLmean")):
                assert len(os.listdir(os.path.join(base_path,"tmpMLmean"))) == 1, "Please remove old files"
                f = os.listdir(os.path.join(base_path,"tmpMLmean"))[0]
                eta, hu, hv = np.load(os.path.join(os.path.join(base_path,"tmpMLmean"), f)).repeat(2,1).repeat(2,2)

                currentTerr = np.array([np.linalg.norm(err_field[center_y-center_N:center_y+center_N, center_x-center_N:center_x+center_N]) for err_field in [true_eta-eta, true_hu-hu, true_hv-hv] ])
                Terr_ML[tau_idx] += 1/N_exp * currentTerr
                
                runningTerrs[n] = currentTerr
                np.save(output_path+"/runningResults/Terrs_MC_tau"+str(tau_idx), runningTerrs)

                os.remove(os.path.join(os.path.join(base_path,"tmpMLmean"), f))

            n = n+1
            signal.alarm(0)

        except Exception as exc:
            print(exc)
            signal.alarm(0)
            pass

        # end: n-loop 

    # Saving results (backup after every tau)
    np.savetxt(output_path+"/TrmseMC"+str(len(ML_Nes_depth[d][0]))+".txt", np.c_[ML_works_depth[d],Terr_ML])

# end: tau-loop


# %% 
# Cleaning Up 
os.rename(os.path.join(base_path, "logMLEnKF.txt"), os.path.join(output_path, "logMLEnKF.txt"))

def clean_tmp(dirname):
    dirpath = os.path.join(base_path,dirname)
    if os.path.isdir(dirpath):
        for f in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, f))
    if os.path.isdir(dirpath):
        os.rmdir(dirpath)

# clean_tmp("tmpTruth")
# clean_tmp("tmpRefGain")

clean_tmp("tmpSLGain")
clean_tmp("tmpSLmean")

clean_tmp("tmpMLGain")
clean_tmp("tmpMLmean")

# %%
