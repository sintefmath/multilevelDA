from matplotlib import pyplot as plt
import numpy as np

def imshow3(etahuhv, negative_vlim=True,  eta_vlim=3, huv_vlim=100, cmap="coolwarm", title=None, ax_title_fontsize=15, **kwargs):
    fig, axs = plt.subplots(1,3, figsize=(15,10))
    fig.suptitle(title)

    im = axs[0].imshow(etahuhv[0], vmin=-negative_vlim*eta_vlim, vmax=eta_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[0], shrink=0.5)
    axs[0].set_title("$\eta$", fontsize=ax_title_fontsize)

    im = axs[1].imshow(etahuhv[1], vmin=-negative_vlim*huv_vlim, vmax=huv_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[1], shrink=0.5)
    axs[1].set_title("$hu$", fontsize=ax_title_fontsize)

    im = axs[2].imshow(etahuhv[2], vmin=-negative_vlim*huv_vlim, vmax=huv_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[2], shrink=0.5)
    axs[2].set_title("$hv$", fontsize=ax_title_fontsize)

    return fig, axs

def imshow2(uv, negative_vlim=True, uv_vlim=1, cmap="coolwarm", title=None, **kwargs):
    fig, axs = plt.subplots(1,2, figsize=(10,10))
    fig.suptitle(title)

    im = axs[0].imshow(uv[0], vmin=-negative_vlim*uv_vlim, vmax=uv_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[0], shrink=0.5)
    axs[0].set_title("$u$", fontsize=15)

    im = axs[1].imshow(uv[1], vmin=-negative_vlim*uv_vlim, vmax=uv_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[1], shrink=0.5)
    axs[1].set_title("$v$", fontsize=15)

    return fig, axs



def imshow3var(est_var, eta_vlim=0.025, huv_vlim=100, title=None, **kvargs):
    return imshow3(est_var, negative_vlim=False, eta_vlim=eta_vlim, huv_vlim=huv_vlim, cmap="Reds", title=title, **kvargs)

def imshow2var(est_var, uv_vlim=0.5, title=None):
    return imshow2(est_var, negative_vlim=False, uv_vlim=uv_vlim, cmap="Reds", title=None)


def imshowSim(sim, **kwargs):
    eta, hu, hv = sim.download(interior_domain_only=False)
    return imshow3(np.array([eta, hu, hv]), **kwargs)