from utils.BasinInit import *

from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

#####################################
# TODO: Implement in framework of `ÒceanModelEnsemble`!
#
# This script mostly consists of duplicates 
# But for convenience in the Rossby example all functionalities are collected here

def initSLensemble(Ne, args, data_args, sample_args, 
                    init_model_error_basis_args=None, 
                    sim_model_error_basis_args=None, sim_model_error_time_step=60.0):
    
    sim_args = {
        "gpu_ctx" : args["gpu_ctx"],
        "nx" : args["nx"],
        "ny" : args["ny"],
        "dx" : args["dx"],
        "dy" : args["dy"],
        "f"  : sample_args["f"],
        "g"  : sample_args["g"],
        "r"  : 0,
        "dt" : 0,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
        "eta0" : data_args["eta"],
        "hu0"  : data_args["hu"],
        "hv0"  : data_args["hv"],
        "H"    : data_args["Hi"],
    }


    SL_ensemble = []

    if init_model_error_basis_args is not None:
        init_mekl = ModelErrorKL.ModelErrorKL(**args, **init_model_error_basis_args)

    if sim_model_error_basis_args is not None:
        sim_mekl = ModelErrorKL.ModelErrorKL(**args, **sim_model_error_basis_args)

    for e in range(Ne):
        sim = CDKLM16.CDKLM16(**sim_args) 
        if init_model_error_basis_args is not None:
            init_mekl.perturbSim(sim)
        if sim_model_error_basis_args is not None:
            sim.model_error = sim_mekl
            sim.model_time_step = sim_model_error_time_step
        SL_ensemble.append( sim )

    return SL_ensemble


def SLstep(SL_ensemble, t):
    for e in range(len(SL_ensemble)):
        SL_ensemble[e].step(t)

    
def SLstepToObservation(SL_ensemble, T):
    for e in range(len(SL_ensemble)):
        SL_ensemble[e].dataAssimilationStep(T)


def SLdownload(SL_ensemble, interior_domain_only=True):
    SL_state = []
    for e in range(len(SL_ensemble)):
        eta, hu, hv = SL_ensemble[e].download(interior_domain_only=interior_domain_only)
        SL_state.append(np.array([eta, hu, hv]))
    SL_state = np.moveaxis(SL_state, 0, -1)
    return SL_state


def SLupload(SL_ensemble, SL_state):
    for e in range(len(SL_ensemble)):
        SL_ensemble[e].upload(*np.pad(SL_state[:,:,:,e], ((0,0),(2,2),(2,2))))


def SLestimate(SL_ensemble, func):
    SL_state = SLdownload(SL_ensemble)
    return func(SL_state, axis=-1)


def GCweights(SL_ensemble, Hx, Hy, r):
    Xs = np.linspace(0, SL_ensemble[0].nx * SL_ensemble[0].dx, SL_ensemble[0].nx)
    Ys = np.linspace(0, SL_ensemble[0].ny * SL_ensemble[0].dy, SL_ensemble[0].ny)
    X, Y = np.meshgrid(Xs, Ys)

    obs_loc = np.zeros(2)
    obs_loc[0] = X[0,Hx]
    obs_loc[1] = Y[Hy,0]
    dists = np.sqrt((X - obs_loc[0])**2 + (Y - obs_loc[1])**2)

    GC = np.zeros_like(dists)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            dist = dists[i,j]
            if dist/r < 1: 
                GC[i,j] = 1 - 5/3*(dist/r)**2 + 5/8*(dist/r)**3 + 1/2*(dist/r)**4 - 1/4*(dist/r)**5
            elif dist/r >= 1 and dist/r < 2:
                GC[i,j] = 4 - 5*(dist/r) + 5/3*(dist/r)**2 + 5/8*(dist/r)**3 -1/2*(dist/r)**4 + 1/12*(dist/r)**5 - 2/(3*(dist/r))

    return GC


def SLEnKF(SL_ensemble, obs, Hx, Hy, R, obs_var, 
           relax_factor=1.0, localisation_weights=None,
           return_perts=False, perts=None):
    
    ## Prior
    SL_state = SLdownload(SL_ensemble) 
    SL_Ne = len(SL_ensemble)

    # Variables
    if obs_var.step is None:
        obs_varN = (obs_var.stop - obs_var.start) 
    else: 
        obs_varN = (obs_var.stop - obs_var.start)/obs_var.step

    ## Localisation
    if localisation_weights is None:
        localisation_weights = np.ones((SL_ensemble[0].ny, SL_ensemble[0].nx))
    
    ## Perturbations
    if perts is not None:
        SL_perts = perts
    else:
        SL_perts = np.random.multivariate_normal(np.zeros(3)[obs_var], np.diag(R[obs_var]), size=SL_Ne)

    ## Analysis
    obs_idxs = [Hy, Hx]

    X0 = SL_state
    X0mean = np.average(X0, axis=-1)

    Y0 = SL_state[obs_var,obs_idxs[0],obs_idxs[1]] + SL_perts.T
    Y0mean = np.average(Y0, axis=-1)

    SL_XY = (relax_factor*np.tile(localisation_weights.flatten(),3)[:,np.newaxis]
             *1/SL_Ne*((X0-X0mean[:,:,:,np.newaxis]).reshape(-1,X0.shape[-1]) @ (Y0 - Y0mean[:,np.newaxis]).T)
             ).reshape(X0mean.shape + (obs_varN,))

    SL_HXY = SL_XY[obs_var,obs_idxs[0],obs_idxs[1],:]
    SL_YY  = SL_HXY + np.diag(R[obs_var])

    SL_K = SL_XY @ np.linalg.inv(SL_YY)

    ## Update
    SL_state = SL_state + (SL_K @ (obs[obs_var,np.newaxis] - SL_state[obs_var,obs_idxs[0],obs_idxs[1]] - SL_perts.T))

    SLupload(SL_ensemble, SL_state)

    if return_perts:
        return SL_K, SL_perts
    else:
        return SL_K


def SLrank(SL_ensemble, truth, obs_locations, R=None):

    assert truth.nx == SL_ensemble[0].nx, "Truth doesnt match finest level"
    assert truth.ny == SL_ensemble[0].ny, "Truth doesnt match finest level"
    assert truth.dx == SL_ensemble[0].dx, "Truth doesnt match finest level"
    assert truth.dy == SL_ensemble[0].dy, "Truth doesnt match finest level"

    SL_state = SLdownload(SL_ensemble)
    true_eta, true_hu, true_hv = truth.download(interior_domain_only=True)

    SL_Fys = []
    for [Hx, Hy] in obs_locations:
        true_values = np.array([true_eta[Hy, Hx], true_hu[Hy, Hx], true_hv[Hy, Hx]]) 
        if R is not None:
            true_values += np.random.multivariate_normal(mean=np.zeros(3),cov=np.diag(R))

        state_values = SL_state[:,Hy,Hx,:]
        if R is not None:
            state_values += np.random.multivariate_normal(mean=np.zeros(3),cov=np.diag(R), size=len(SL_ensemble)).T

        SL_Fys.append( 1/len(SL_ensemble) * np.sum(state_values < true_values[:,np.newaxis], axis=1) )

    return SL_Fys