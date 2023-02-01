import numpy as np

from gpuocean.utils import WindStress


class KarhunenLoeve_Sampler():
    """ Sampling random field based on Karhunen-Loeve expansions """
    
    def __init__(self, t_splits, N):
        """
        t_splits (int) - number of how many KL-fields are generated
        KL_N (int)     - spatial resolution of KL-fields
        """

        self.KL_DECAY=1.05
        self.KL_SCALING=0.15

        self.KL_bases_N = 10

        self.t_splits = t_splits
        self.N        = N

        self.KL_ref_fields = np.zeros((self.KL_bases_N, self.KL_bases_N, self.N, self.N))

        for n in range(1, self.KL_bases_N+1):
            for m in range(1, self.KL_bases_N+1):
                self.KL_ref_fields[n-1, m-1, :, :] = self.KL_SCALING * m**(-self.KL_DECAY) * n**(-self.KL_DECAY) * np.outer(np.sin(m*np.pi*np.linspace(0,1,self.N)), np.sin(n*np.pi*np.linspace(0,1,self.N))) 
        
        self.KL_ref_fields = np.reshape(self.KL_ref_fields, (self.KL_bases_N * self.KL_bases_N, self.N, self.N))


    def perturbations(self):
        """ Output: size=(t_splits, N, N) with t_splits-times a KL-field """

        KL_fields = np.zeros((self.t_splits, self.N, self.N))

        rns = np.random.normal(size=(self.KL_bases_N * self.KL_bases_N,self.t_splits))
        
        for t in range(self.t_splits):
            KL_fields[t,:,:] = np.sum(self.KL_ref_fields*rns[:,t][:,np.newaxis, np.newaxis], axis=0)

        return KL_fields


def wind_bump(ny, nx, sig = None):
    dataShape = (ny, nx )
    w = np.zeros(dataShape, dtype=np.float32, order='C')

    x_center = 0.5*nx
    y_center = 0.5*ny

    if sig is None:
        sig = nx**2/10

    for j in range(ny):
        for i in range(nx):
            x = i - x_center
            y = j - y_center

            d = x**2 + y**2
            
            w[j, i] = np.exp(-1/2*d/sig)    
    
    return w


def wind_sample(KLSampler, wind_weight=None, wind_speed=5.0):
    ## KL perturbed wind fields (much faster!)
        
    wind_degree = np.deg2rad(np.random.uniform(0,360))

    init_wind_u = wind_speed * np.sin(wind_degree) * np.ones((KLSampler.N,KLSampler.N))
    init_wind_v = wind_speed * np.cos(wind_degree) * np.ones((KLSampler.N,KLSampler.N))

    KL_fields_u = KLSampler.perturbations()
    KL_fields_v = KLSampler.perturbations()

    wind_u = np.repeat(init_wind_u[np.newaxis,:,:], KLSampler.t_splits, axis=0) + np.cumsum(KL_fields_u, axis=0)
    wind_v = np.repeat(init_wind_v[np.newaxis,:,:], KLSampler.t_splits, axis=0) + np.cumsum(KL_fields_v, axis=0)

    if wind_weight is None:
        wind_weight = wind_bump(KLSampler.N, KLSampler.N)

    wind_u = wind_u *np.repeat(wind_weight[np.newaxis,:,:], KLSampler.t_splits, axis=0)
    wind_v = wind_v *np.repeat(wind_weight[np.newaxis,:,:], KLSampler.t_splits, axis=0)

    ts = np.linspace(0,250000,KLSampler.t_splits)

    return WindStress.WindStress(t=ts, wind_u=wind_u.astype(np.float32), wind_v=wind_v.astype(np.float32))
