from utils.BasinInit import *

from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

def initMLensemble(ls, ML_Nes, model_error_args_list, model_error_basis_args, sample_args):

    # Model errors
    mekls = []
    for l_idx in range(len(ls)): 
        mekls.append( ModelErrorKL.ModelErrorKL(**model_error_args_list[l_idx], **model_error_basis_args) )

    ## MultiLevel ensemble
    ML_ensemble = []

    # 0-level
    lvl_ensemble = []
    data_args = make_init_fields(model_error_args_list[0])
    for i in range(ML_Nes[0]):
        if i % 100 == 0: print(i)
        sim = make_sim(model_error_args_list[0], sample_args, init_fields=data_args)
        mekls[0].perturbSim(sim)
        lvl_ensemble.append( sim )

    ML_ensemble.append(lvl_ensemble)

    # diff-levels
    for l_idx in range(1,len(ML_Nes)):
        lvl_ensemble0 = []
        lvl_ensemble1 = []
        
        data_args0 = make_init_fields(model_error_args_list[l_idx])
        data_args1 = make_init_fields(model_error_args_list[l_idx-1])
        
        for e in range(ML_Nes[l_idx]):
            sim0 = make_sim(model_error_args_list[l_idx], sample_args, init_fields=data_args0)
            mekls[l_idx].perturbSim(sim0)

            sim1 = make_sim(model_error_args_list[l_idx-1], sample_args, init_fields=data_args1)
            mekls[l_idx-1].perturbSimSimilarAs(sim1, modelError=mekls[l_idx])

            lvl_ensemble0.append(sim0)
            lvl_ensemble1.append(sim1)
        
        ML_ensemble.append([lvl_ensemble0,lvl_ensemble1])

    return ML_ensemble