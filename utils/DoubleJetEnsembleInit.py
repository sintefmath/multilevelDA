from gpuocean.SWEsimulators import CDKLM16, ModelErrorKL

import pycuda.driver as cuda

def initMLensemble(ML_Nes, args_list, init_list, 
                   sim_model_error_basis_args=None, sim_model_error_time_step=None):
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
    assert len(args_list) == len(init_list), "Number of levels in args and level sizes do not match"

    # Model errors
    mekl_stream = cuda.Stream()
    mekls = []
    for l_idx in range(len(args_list)): 
        grid_args = {key: args_list[l_idx][key] for key in ('nx', 'ny', 'dx', 'dy', 'gpu_ctx', 'boundary_conditions')}
        mekls.append( ModelErrorKL.ModelErrorKL(gpu_stream=mekl_stream, **grid_args, **sim_model_error_basis_args) )


    ## MultiLevel ensemble
    ML_ensemble = []

    # 0-level
    lvl_ensemble = []
    for i in range(ML_Nes[0]):
        if i % 100 == 0: print(i)
        sim = CDKLM16.CDKLM16(**args_list[0], **init_list[0]) 

        sim.model_error = mekls[0]
        sim.model_time_step = sim_model_error_time_step
        lvl_ensemble.append( sim )

    ML_ensemble.append(lvl_ensemble)

    # diff-levels
    for l_idx in range(1,len(ML_Nes)):
        print(l_idx)
        lvl_ensemble0 = []
        lvl_ensemble1 = []
        
        for e in range(ML_Nes[l_idx]):
            sim0 = CDKLM16.CDKLM16(**args_list[l_idx], **init_list[l_idx]) 
            sim1 = CDKLM16.CDKLM16(**args_list[l_idx-1], **init_list[l_idx-1])

            sim0.model_error = mekls[l_idx]
            sim1.model_error = mekls[l_idx-1]

            sim0.model_time_step = sim_model_error_time_step
            sim1.model_time_step = sim_model_error_time_step

            lvl_ensemble0.append(sim0)
            lvl_ensemble1.append(sim1)
        
        ML_ensemble.append([lvl_ensemble0,lvl_ensemble1])

    return ML_ensemble