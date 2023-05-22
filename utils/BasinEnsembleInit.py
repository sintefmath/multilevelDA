from utils.BasinInit import *

from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

def initMLensemble(ML_Nes, args_list, make_data_args, sample_args, 
                   init_model_error_basis_args=None, sim_model_error_basis_args=None, sim_model_error_time_step=None):
    """
    Utility function for the generation of multi-level ensembles of specific kind

    ML_Nes - list with ensemble sizes per level
    args_list - list of dicts with grid details per level, required keys per level: ny, ny, dx, dy, gpu_ctx, gpu_stream, boundary_condtions
    make_data_args - function that takes one element of args_list and returns the initial fields (see e.g. `BasinInit.py`), required output keys: eta0, hu0, hv0, Hi
    sample_args - dict with additional arguments to the Simulator
    init_model_error_basis_args - dict with arguments for ModelErrorKL-class used for initial perturbations. `None` means no initial model error
    sim_model_error_basis_args - dict with arguments for ModelErrorKL-class used for temporal perturbations. `None` means no temporal model error
    sim_model_error_time_step - float>0 as CDKLM16.model_time_step (aka. the intervals in which sim_model_error is applied)
    """

    assert len(ML_Nes) == len(args_list), "Number of levels in args and level sizes do not match"

    data_args_list = []
    if isinstance(make_data_args, list):
        assert len(ML_Nes) == len(make_data_args), "Number of levels in data_args and level sizes do not match"
        data_args_list = make_data_args
    else: 
        for l_idx in range(len(ML_Nes)):
            data_args_list.append( make_data_args(args_list[l_idx]) )

    # Model errors
    if init_model_error_basis_args is not None: 
        init_mekls = []
        for l_idx in range(len(args_list)): 
            init_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **init_model_error_basis_args) )

    if sim_model_error_basis_args is not None: 
        sim_mekls = []
        for l_idx in range(len(args_list)): 
            sim_mekls.append( ModelErrorKL.ModelErrorKL(**args_list[l_idx], **sim_model_error_basis_args) )



    ## MultiLevel ensemble
    ML_ensemble = []

    # 0-level
    lvl_ensemble = []
    for i in range(ML_Nes[0]):
        if i % 100 == 0: print(i)
        sim = make_sim(args_list[0], sample_args, init_fields=data_args_list[0])
        if init_model_error_basis_args is not None:
            init_mekls[0].perturbSim(sim)
        if sim_model_error_basis_args is not None:
            # sim.setKLModelError(**sim_model_error_basis_args)
            sim.model_error = sim_mekls[0]
        sim.model_time_step = sim_model_error_time_step
        lvl_ensemble.append( sim )

    ML_ensemble.append(lvl_ensemble)

    # diff-levels
    for l_idx in range(1,len(ML_Nes)):
        print(l_idx)
        lvl_ensemble0 = []
        lvl_ensemble1 = []
        
        for e in range(ML_Nes[l_idx]):
            sim0 = make_sim(args_list[l_idx], sample_args, init_fields=data_args_list[l_idx])
            sim1 = make_sim(args_list[l_idx-1], sample_args, init_fields=data_args_list[l_idx-1])
            
            if init_model_error_basis_args is not None:
                init_mekls[l_idx].perturbSim(sim0)
                init_mekls[l_idx-1].perturbSimSimilarAs(sim1, modelError=init_mekls[l_idx])

            if sim_model_error_basis_args is not None:
                sim0.model_error = sim_mekls[l_idx]
                sim1.model_error = sim_mekls[l_idx-1]

            sim0.model_time_step = sim_model_error_time_step
            sim1.model_time_step = sim_model_error_time_step

            lvl_ensemble0.append(sim0)
            lvl_ensemble1.append(sim1)
        
        ML_ensemble.append([lvl_ensemble0,lvl_ensemble1])

    return ML_ensemble