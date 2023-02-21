from utils.RossbyInit import *
from utils.WindPerturb import *

from gpuocean.SWEsimulators import CDKLM16

#####################################
# TODO: Implement in framework of `ÒceanModelEnsemble`!
#
# This script mostly consists of duplicates 
# But for convenience in the Rossby example all functionalities are collected here

def initSLensemble(gpu_ctx, ls, Ne, KLSampler, wind_weight, wind_T):
    SL_ensemble = []

    data_args = initLevel(ls[-1])
    for e in range(Ne):
        wind = wind_sample(KLSampler, wind_T, wind_weight=wind_weight)
        SL_ensemble.append(CDKLM16.CDKLM16(gpu_ctx, **data_args, wind=wind))

    return SL_ensemble


def SLstep(SL_ensemble, t):
    for e in range(len(SL_ensemble)):
        SL_ensemble[e].step(t)


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


def GCweights(SL_ensemble, Hx, Hy, r):
    Xs = np.linspace(0, SL_ensemble[0].nx * SL_ensemble[0].dx, SL_ensemble[0].nx)
    Ys = np.linspace(0, SL_ensemble[0].ny * SL_ensemble[0].dy, SL_ensemble[0].ny)
    X, Y = np.meshgrid(Xs, Ys)

    obs_loc = np.zeros(2)
    obs_loc[0] = X[0,Hx]
    obs_loc[1] = Y[Hy,0]
    dists = np.sqrt((X - obs_loc[0])**2 + (Y - obs_loc[1])**2)

    r = 2.5*1e7

    GC = np.zeros_like(dists)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            dist = dists[i,j]
            if dist/r < 1: 
                GC[i,j] = 1 - 5/3*(dist/r)**2 + 5/8*(dist/r)**3 + 1/2*(dist/r)**4 - 1/4*(dist/r)**5
            elif dist/r >= 1 and dist/r < 2:
                GC[i,j] = 4 - 5*(dist/r) + 5/3*(dist/r)**2 + 5/8*(dist/r)**3 -1/2*(dist/r)**4 + 1/12*(dist/r)**5 - 2/(3*(dist/r))

    return GC


def SLEnKF(SL_ensemble, obs, Hx, Hy, R, r = 2.5*1e7):
    
    ## Prior
    SL_state = SLdownload(SL_ensemble) 
    SL_Ne = len(SL_ensemble)

    ## Localisation
    GC = GCweights(SL_ensemble, Hx, Hy, r)
    
    ## Perturbations
    SL_perts = np.random.multivariate_normal(np.zeros(3), np.diag(R), size=SL_Ne)

    ## Analysis
    obs_idxs = [Hy, Hx]

    X0 = SL_state
    X0mean = np.average(X0, axis=-1)

    Y0 = SL_state[:,obs_idxs[0],obs_idxs[1]] + SL_perts.T
    Y0mean = np.average(Y0, axis=-1)

    SL_XY = (np.tile(GC.flatten(),3)[:,np.newaxis]*1/SL_Ne*((X0-X0mean[:,:,:,np.newaxis]).reshape(-1,X0.shape[-1]) @ (Y0 - Y0mean[:,np.newaxis]).T)).reshape(X0mean.shape + (3,))

    SL_HXY = SL_XY[:,obs_idxs[0],obs_idxs[1],:]
    SL_YY  = SL_HXY + np.diag(R)

    SL_K = SL_XY @ np.linalg.inv(SL_YY)

    ## Update
    SL_state = SL_state + (SL_K @ (obs[:,np.newaxis] - SL_state[:,obs_idxs[0],obs_idxs[1]] - SL_perts.T))

    SLupload(SL_ensemble, SL_state)

    return SL_state