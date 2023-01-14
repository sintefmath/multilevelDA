import numpy as np
import copy

from skimage.measure import block_reduce

def initGridSpecs(l):
    data_args = {}
    data_args["nx"] = 2**l
    data_args["ny"] = 2**(l+1) 

    data_args["dx"] = 2**(19-l)*100
    data_args["dy"] = 2**(19-l)*100

    return data_args


def initBump(l, l_max, d_shift=1e6,D=0.5*1e6):

    data_args_max = initGridSpecs(l_max)
    nx, ny = data_args_max["nx"], data_args_max["ny"]

    dx, dy = data_args_max["dx"], data_args_max["dy"]

    eta0  = np.zeros((ny,nx), dtype=np.float32, order='C')
    x_center = dx*nx*0.5
    y_center = dy*ny*0.5

    for j in range(ny):
        for i in range(nx):
            x = dx*i - x_center
            y = dy*j - y_center

            d = np.sqrt(x**2 + y**2)
            
            eta0[j, i] += 0.1*(1.0+np.tanh(-(d-d_shift)/D))

    eta0 = block_reduce(eta0, block_size=(2**(l_max-l),2**(l_max-l)), func=np.nanmean)
    eta0 = np.pad(eta0, ((2,2),(2,2)))

    hu0 = np.zeros_like(eta0, dtype=np.float32)

    hv0 = np.zeros_like(eta0, dtype=np.float32)

    return eta0, hu0, hv0


def initLevel(l, l_max=None):
    if l_max is None:
        l_max = l 

    data_args = initGridSpecs(l)
    dataShape = (data_args["ny"] + 4, data_args["nx"] + 4)

    data_args["dt"] = 0.0
    data_args["g"] = 9.81
    data_args["f"] = 1.2e-4
    data_args["r"] = 0.0
    # data_args["boundary_conditions"] = Common.BoundaryConditions(1,1,1,1)

    data_args["eta0"], data_args["hu0"], data_args["hv0"] = initBump(l, l_max)
    
    H0 = 1000.0
    data_args["H"] = np.ma.array(np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')*H0, mask=False)

    return data_args