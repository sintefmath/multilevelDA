import numpy as np
import copy

from skimage.measure import block_reduce

from gpuocean.SWEsimulators import CDKLM16
from gpuocean.utils import Common

def initGridSpecs(l):
    data_args = {}
    data_args["nx"] = 2**l
    data_args["ny"] = 2**(l+1) 

    data_args["dx"] = 2**(9-l)*160
    data_args["dy"] = 2**(9-l)*160

    return data_args


def make_init_fields(args):
    dataShape = (args["ny"] + 4, args["nx"] + 4)
    dataShapeH = (args["ny"] + 5, args["nx"] + 5)

    eta = np.zeros(dataShape, dtype=np.float32)
    hu  = np.zeros(dataShape, dtype=np.float32)
    hv  = np.zeros(dataShape, dtype=np.float32)
    Hi = np.ones(dataShapeH, dtype=np.float32)*60

    return {"eta": eta, "hu": hu, "hv": hv, "Hi": Hi}


def make_init_steady_state(args, a=1, b = 2.5e-9, sample_args={"g":9.81, "f":0.0012}, bump_fractal_dist=6):
    dataShape = (args["ny"] + 4, args["nx"] + 4)
    dataShapeH = (args["ny"] + 5, args["nx"] + 5)

    eta0 = np.zeros(dataShape, dtype=np.float32)
    hu0  = np.zeros(dataShape, dtype=np.float32)
    hv0  = np.zeros(dataShape, dtype=np.float32)
    H_const = 60
    Hi = np.ones(dataShapeH, dtype=np.float32)*H_const

    nx = args["nx"]
    ny = args["ny"]
    dx = args["dx"]
    dy = args["dy"]


    def add_bump(x_bump, y_bump, a, b):
        X = np.arange(start=0.5*dx, stop=nx*dx , step=dx) - x_bump
        Y = np.arange(start=0.5*dy, stop=ny*dy , step=dy) - y_bump
        YY, XX = np.meshgrid(X, Y)

        etaBump = a*np.exp(-b*(XX*XX + YY*YY))
        eta0[2:-2, 2:-2] += etaBump

        DetaDX = -a*2*b*XX*np.exp(-b*(XX*XX + YY*YY))
        DetaDY = -a*2*b*YY*np.exp(-b*(XX*XX + YY*YY))

        hu0[2:-2,2:-2] += -(sample_args["g"]*(etaBump + H_const)/sample_args["f"])*DetaDX
        hv0[2:-2,2:-2] += (sample_args["g"]*(etaBump + H_const)/sample_args["f"])*DetaDY


    add_bump(nx*dx/2, ny*dy/2 + ny*dy/bump_fractal_dist, a, b)
    add_bump(nx*dx/2, ny*dy/2 - ny*dy/bump_fractal_dist, -a, b)

    return {"eta": eta0, "hu": hu0, "hv": hv0, "Hi": Hi}


def make_sim(me_args, sample_args={"g":9.81, "f":0.0012}, init_fields = None):
    if init_fields is None:
        init_fields = make_init_fields(me_args)

    sim_args = {
        "gpu_ctx" : me_args["gpu_ctx"],
        "nx" : me_args["nx"],
        "ny" : me_args["ny"],
        "dx" : me_args["dx"],
        "dy" : me_args["dy"],
        "f"  : sample_args["f"],
        "g"  : sample_args["g"],
        "r"  : 0,
        "dt" : 0,
        "boundary_conditions": Common.BoundaryConditions(2,2,2,2),
        "eta0" : init_fields["eta"],
        "hu0"  : init_fields["hu"],
        "hv0"  : init_fields["hv"],
        "H"    : init_fields["Hi"],
    }

    return CDKLM16.CDKLM16(**sim_args)

