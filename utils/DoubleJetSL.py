import numpy as np

from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL
from gpuocean.utils import Common

from scipy.spatial.distance import cdist

import pycuda.driver as cuda

#####################################
# TODO: Implement in framework of `ÒceanModelEnsemble`!
#
# This script mostly consists of duplicates 
# But for convenience in the Rossby example all functionalities are collected here

def initSLensemble(Ne, doubleJetCase_args, doubleJetCase_init,  
                    doubleJetCase_meargs=None, sim_model_error_time_step=60.0):
    
    # Letting simulator set dt
    doubleJetCase_args["dt"] = 0.0

    # Init a single model error with own stream
    mekl_stream = cuda.Stream()
    grid_args = {key: doubleJetCase_args[key] for key in ('nx', 'ny', 'dx', 'dy', 'gpu_ctx', 'boundary_conditions')}
    mekl = ModelErrorKL.ModelErrorKL(gpu_stream=mekl_stream, **grid_args, **doubleJetCase_meargs)

    # Generate ensemble
    SL_ensemble = []
    for e in range(Ne):
        sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init) 
        
        sim.model_error = mekl
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


def GCweights(SL_ensemble, obs_x, obs_y, r):
    Xs = np.linspace(0.5*SL_ensemble[0].dx, (SL_ensemble[0].nx-0.5) * SL_ensemble[0].dx, SL_ensemble[0].nx)
    Ys = np.linspace(0.5*SL_ensemble[0].dy, (SL_ensemble[0].ny-0.5) * SL_ensemble[0].dy, SL_ensemble[0].ny)
    X, Y = np.meshgrid(Xs, Ys)

    obs_loc = [obs_x, obs_y]

    def _calculate_distance(coord1, coord2, xdim, ydim):
        distx = np.abs(coord1[0] - coord2[0])
        disty = np.abs(coord1[1] - coord2[1])
        distx = np.minimum(distx, xdim - distx)
        disty = np.minimum(disty, ydim - disty)
        return np.sqrt(distx**2 + disty**2)
    
    xdim = SL_ensemble[0].dx * SL_ensemble[0].nx
    ydim = SL_ensemble[0].dy * SL_ensemble[0].ny
    
    grid_coordinates = np.vstack((X.flatten(), Y.flatten())).T
    dists = cdist(grid_coordinates, np.atleast_2d([obs_x, obs_y]),
                    lambda u, v: _calculate_distance(u, v, xdim, ydim))
    dists = dists.reshape(X.shape)

    # dists = np.sqrt((X - obs_loc[0])**2 + (Y - obs_loc[1])**2)

    GC = np.zeros_like(dists)
    GC = np.where(dists/r < 1, 1 - 5/3*(dists/r)**2 + 5/8*(dists/r)**3 + 1/2*(dists/r)**4 - 1/4*(dists/r)**5, GC)
    GC = np.where(np.logical_and((dists/r >= 1), (dists/r < 2)), 4 - 5*(dists/r) + 5/3*(dists/r)**2 + 5/8*(dists/r)**3 -1/2*(dists/r)**4 + 1/12*(dists/r)**5 - 2/np.maximum(1e-6,(3*(dists/r))), GC)

    return GC


def SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y):

    if not isinstance(SL_ensemble, list):
        SL_ensemble = [SL_ensemble]

    Xs = np.linspace(SL_ensemble[0].dx/2, (SL_ensemble[0].nx - 1/2) * SL_ensemble[0].dx, SL_ensemble[0].nx)
    Ys = np.linspace(SL_ensemble[0].dy/2, (SL_ensemble[0].ny - 1/2) * SL_ensemble[0].dy, SL_ensemble[0].ny)

    Hx = ((Xs - obs_x)**2).argmin()
    Hy = ((Ys - obs_y)**2).argmin()

    return Hx, Hy



def SLEnKF(SL_ensemble, obs, obs_x, obs_y, R, obs_var, 
           relax_factor=1.0, localisation_weights=None,
           return_perts=False, perts=None, dx=None, dy=None):
    """
    SL_ensemble     - list of CDKLM16 instances
    obs             - ndarray of size (3,) with truth observation (eta, hu, hv)
    obs_x           - float for physical observation location
    obs_y           - float for physical observation location
    R               - ndarray of size (3,) with observation noise covariance diagonal
    obs_var         - slice with observed variables
    relax_factor    - float between 0 and 1
    location_weights- ndarray of size (ny, nx) otherwise no localisation
    perts           - ndarray of size (Ne, Ny) - as output
    """

    # Check that obs_x and obs_y are NOT integer types 
    # (as this is indicator that those are indices as in earlier implementation)
    assert not isinstance(obs_x, (np.integer, int)), "This should be physical distance, not index"
    assert not isinstance(obs_y, (np.integer, int)), "This should be physical distance, not index"


    
    ## Prior
    if not isinstance(SL_ensemble[0], np.ndarray):
        SL_state = SLdownload(SL_ensemble) 
        SL_Ne = len(SL_ensemble)

        # From observation location to observation indices
        Hx, Hy = SLobsCoord2obsIdx(SL_ensemble, obs_x, obs_y)

    else: 
        SL_state = SL_ensemble
        SL_Ne = SL_state.shape[-1]

        assert dx is not None, "dx has to be given, when SL_state is an np.ndarray"
        assert dy is not None, "dy has to be given, when SL_state is an np.ndarray"

        ny, nx = SL_state.shape[1:3]

        Xs = np.linspace(dx/2, (nx - 1/2) * dx, nx)
        Ys = np.linspace(dy/2, (ny - 1/2) * dy, ny)

        Hx = ((Xs - obs_x)**2).argmin()
        Hy = ((Ys - obs_y)**2).argmin()


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
    Y0mean = np.average(Y0, axis=-1)[:,np.newaxis]

    SL_XY = (relax_factor*np.tile(localisation_weights.flatten(),3)[:,np.newaxis]
             *1/SL_Ne*((X0-X0mean[:,:,:,np.newaxis]).reshape(-1,X0.shape[-1]) @ (Y0 - Y0mean).T)
             ).reshape(X0mean.shape + (obs_varN,))

    SL_YY = 1/SL_Ne * (Y0 - Y0mean) @ (Y0 - Y0mean).T 

    SL_K = SL_XY @ np.linalg.inv(SL_YY)

    ## Update
    SL_state = SL_state + (SL_K @ (obs[obs_var,np.newaxis] - SL_state[obs_var,obs_idxs[0],obs_idxs[1]] - SL_perts.T))

    if not isinstance(SL_ensemble[0], np.ndarray):
        SLupload(SL_ensemble, SL_state)

        if return_perts:
            return SL_K, SL_perts
        else:
            return SL_K

    else:
        return SL_state


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