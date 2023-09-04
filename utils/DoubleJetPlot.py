from matplotlib import pyplot as plt
import numpy as np

def imshow3(etahuhv, negative_vlim=True,  eta_vlim=3, huv_vlim=750, cmap="coolwarm", title=None, **kwargs):
    fig, axs = plt.subplots(1,3, figsize=(15,10))
    fig.suptitle(title)

    im = axs[0].imshow(etahuhv[0], vmin=-negative_vlim*eta_vlim, vmax=eta_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[0], shrink=0.25)
    axs[0].set_title("$\eta$", fontsize=15)

    im = axs[1].imshow(etahuhv[1], vmin=-negative_vlim*huv_vlim, vmax=huv_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[1], shrink=0.25)
    axs[1].set_title("$hu$", fontsize=15)

    im = axs[2].imshow(etahuhv[2], vmin=-negative_vlim*huv_vlim, vmax=huv_vlim, cmap=cmap, origin="lower", **kwargs)
    plt.colorbar(im, ax=axs[2], shrink=0.25)
    axs[2].set_title("$hv$", fontsize=15)

    return fig, axs


def imshow3var(est_var, eta_vlim=0.5, huv_vlim=100, title=None):
    return imshow3(est_var, negative_vlim=False, eta_vlim=eta_vlim, huv_vlim=huv_vlim, cmap="Reds", title=None)


def imshowSim(sim, **kwargs):
    eta, hu, hv = sim.download(interior_domain_only=False)
    return imshow3(np.array([eta, hu, hv]), **kwargs)




def crossSection(etahuv, eta_lim=3.5, huv_lim=900, x_idx=None, **kwargs):

    fig, axs = plt.subplots(1, 3, figsize=(15,3))

    if x_idx is None:
        x_idx = int(etahuv[0].shape[1]/2)
    ny = etahuv[0].shape[0]

    axs[0].plot(etahuv[0][:,x_idx])
    axs[0].set_xlim((0, ny))
    axs[0].set_ylim((-3.5,3.5))

    axs[1].plot(etahuv[1][:,x_idx])
    axs[1].set_xlim((0, ny))
    axs[1].set_ylim((-900,900))

    axs[2].plot(etahuv[2][:,x_idx])
    axs[2].set_xlim((0, ny))
    axs[2].set_ylim((-900,900))

    return fig, axs


def crossSectionSim(sim, **kwargs):
    eta, hu, hv = sim.download(interior_domain_only=True)
    return crossSection(np.array([eta, hu, hv]), **kwargs)

