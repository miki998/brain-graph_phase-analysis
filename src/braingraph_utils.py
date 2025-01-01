"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *
from src.operations import *

from joblib import Parallel, delayed
import netplotbrain

def volcoord2mnicoord(arrays: np.ndarray, affine: np.ndarray):
    """
    Compute volume coordinates to MNI coordinates transform.

    Transforms a set of 3D volume coordinates to MNI coordinates using 
    a provided affine transform matrix. The affine matrix maps between 
    volume voxel indices and MNI coordinates.

    Parameters
    ----------
    arrays : np.ndarray
        The volume coordinate arrays to transform. Each coordinate is a row.
    affine : np.ndarray 
        The affine transform matrix mapping volume to MNI coordinates.

    Returns
    -------
    ret : np.ndarray
        The MNI coordinates corresponding to the input volume coordinates.
    """

    tmp = np.concatenate([arrays, np.ones((arrays.shape[0], 1))], axis=1)
    ret = np.matmul(affine, tmp.T)[:3].T

    return np.array(ret).astype(float)

def mnicoord2volcoord(arrays: np.ndarray, affine: np.ndarray):
    """
    Compute volume coordinates from MNI coordinates.

    Transforms MNI coordinates to equivalent volume coordinates using
    the provided affine transform.

    Parameters
    ----------
    arrays : np.ndarray
        Array of MNI coordinates to transform.
    affine : np.ndarray
        Affine transform mapping from MNI space to volume space.

    Returns
    -------
    np.ndarray
        Array of transformed volume coordinates.
    """

    inv_affine = np.linalg.inv(affine)
    tmp = np.concatenate([arrays, np.ones((arrays.shape[0], 1))], axis=1)
    ret = np.matmul(inv_affine, tmp.T)[:3].T

    return np.array(ret).astype(int)

def visualize_braingraph(signal, A, coords, figsize=(5, 5), axview=None, 
                         phasegrain=12, cmap="bwr", nscale=100, hlv=0.85, nsizelegend=None):
    """
    Visualizing braingraphs and associated signals in MNI
    """
    slocation = deepcopy(coords)
    
    assert np.sum(np.abs(signal - signal.mean())) != 0, "Signal mut not be constant"
    nodes = {"x": [], "y": [], "z": [], "signal": [], "phase": []}
    edges = {"i": [], "j": [], "weight": []}

    grain = np.linspace(0, np.pi, phasegrain)
    p = plt.get_cmap(cmap)
    # populate nodes
    for sidx, s in enumerate(slocation):
        nodes["x"].append(s[0])
        nodes["y"].append(s[1])
        nodes["z"].append(s[2])

        angleencode = np.where((np.angle(signal[sidx]) - grain) <= 0)[0][0]
        nodes["phase"].append(p(angleencode / (phasegrain - 1)))
        # nodes["signal"].append(np.abs(scaled_signal[sidx]))
        nodes["signal"].append(np.abs(signal[sidx]))

    nodes = pd.DataFrame.from_dict(nodes)

    # populate edges
    for idx, s1 in enumerate(slocation):
        for jdx, s2 in enumerate(slocation):
            if idx > jdx:
                continue
            if A[idx, jdx] != 0:
                edges["i"].append(idx)
                edges["j"].append(jdx)
                edges["weight"].append(A[idx, jdx] / 50)

    edges = pd.DataFrame.from_dict(edges)

    if nsizelegend is None:
        nsizelegend = [np.round(np.min(nodes["signal"]), 2), np.round(np.max(nodes["signal"]),2)]
    if axview is None:
        view = ['LSR', 'AIP']
    else:
        view = 'L'

    fig, ax = netplotbrain.plot(
        template="MNI152NLin2009cAsym",
        templatestyle="glass",
        nodes=nodes,
        node_size='signal',
        node_color="phase",
        edges=edges,
        #   highlight_edges = adj,
        node_scale=nscale,
        # node_sizevminvmax=nodevminvmax,
        highlight_level=hlv,
        node_sizelegend=nsizelegend,
        view=view
    )
    if not (axview is None):
        ax[0].view_init(elev=axview[1], azim=axview[0])
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    return fig, ax[0]

def visualize_braingraph_dynamic(signals, A, coords, figsize=(5, 5), axview=None, 
                         phasegrain=12, cmap="bwr", nscale=100, hlv=0.85):
    """
    Visualizing braingraphs and associated signals in MNI
    """
    nsizelegend = [np.round(np.abs(signals).min(),2), np.round(np.abs(signals).max(),2)]
    def save_braingraph(k):
        fig, ax = visualize_braingraph(signals[k], A, coords, figsize=figsize, axview=axview, 
                                       phasegrain=phasegrain, cmap=cmap, nscale=nscale, hlv=hlv, nsizelegend=nsizelegend)
        fig.savefig(f'../figure_resources/braingraph_{k}.png')
        plt.close(fig)

    Parallel(n_jobs=-1)(delayed(save_braingraph)(k) for k in tqdm(range(len(signals))))