from utils.BasinInit import *
from utils.WindPerturb import *

from gpuocean.SWEsimulators import CDKLM16

def initMLensemble(gpu_ctx, ls, Nes, KLSampler, wind_weight, wind_T, wind_speed):
    ML_ensemble = []

    lvl_ensemble = []
    data_args = initLevel(ls[0])
    for e in range(Nes[0]):
        wind = wind_sample(KLSampler, wind_T, wind_weight=wind_weight, wind_speed=wind_speed)
        lvl_ensemble.append(CDKLM16.CDKLM16(gpu_ctx, **data_args, wind=wind))

    ML_ensemble.append(lvl_ensemble)
        

    for l_idx in range(1,len(Nes)):
        lvl_ensemble0 = []
        lvl_ensemble1 = []
        
        data_args0 = initLevel(ls[l_idx])
        data_args1 = initLevel(ls[l_idx-1])
        
        for e in range(Nes[l_idx]):
            wind = wind_sample(KLSampler, wind_T, wind_weight=wind_weight, wind_speed=wind_speed)
            
            lvl_ensemble0.append(CDKLM16.CDKLM16(gpu_ctx, **data_args0, wind=wind))
            lvl_ensemble1.append(CDKLM16.CDKLM16(gpu_ctx, **data_args1, wind=wind))
        
        ML_ensemble.append([lvl_ensemble0,lvl_ensemble1])

    return ML_ensemble