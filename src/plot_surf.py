import os.path as op
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import expand_labels
import nibabel as nib

from nilearn import datasets
from nilearn.surface import vol_to_surf
from nilearn.plotting import plot_surf_stat_map

def plot_node_signal_on_surf(
    node_signal: np.ndarray,
    path_to_atlas: str,
    labels: Optional[list],
    surf_template: str = "fsaverage6",
    expand_vols: bool = True,
    inflate: bool = True,
    figsize: tuple = (20, 15),
    saveloc: Optional[str] = None,
    **kwargs,
) -> tuple:
    """Plot a node signal on the a template surface.

    Parameters
    ----------
    node_signal : np.ndarray
        Signal values for each node in the atlas
    path_to_atlas : str
        Path to the atlas file (has to be in MNI space)
    labels : Optional[list[str]]
        Labels for each node in the atlas
    surf_template : str, optional
        Template surface to use for plotting, by default "fsaverage6"
    expand_vols : bool, optional
        Condition to expand the atlas volumes (better visualization in case the alignment is not perfect), by default True
    inflate : bool, optional
        Condition to inflate the surface, by default True
    figsize : tuple, optional
        Figure size for the plot, by default (20, 15)
    saveloc : Optional[str], optional
        Path to save the figure (won't save if `None`), by default None

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axes for the plot
    """
    if isinstance(path_to_atlas, nib.nifti1.Nifti1Image):
        roi_atlas = path_to_atlas
        atlas_data = roi_atlas.get_fdata()
    else:
        roi_atlas = nib.load(path_to_atlas)
        atlas_data = roi_atlas.get_fdata()

    stat_map = np.zeros_like(atlas_data, dtype=float)
    if labels is None:
        labels = [f"ROI {i}" for i in range(atlas_data.max())]

    # This implies that the atlas ROIs indices start from 1
    for i, _ in enumerate(labels):
        stat_map[atlas_data == i + 1] = node_signal[i]

    if expand_vols:
        stat_map = expand_labels(stat_map, 3)

    surf_map = nib.Nifti1Image(stat_map, roi_atlas.affine)

    fsaverage = datasets.fetch_surf_fsaverage(surf_template)

    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        figsize=figsize,
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    surf_def = [fsaverage.pial_left, fsaverage.pial_right]
    if inflate:
        hemis_def = [fsaverage.infl_left, fsaverage.infl_right]
    else:
        hemis_def = [fsaverage.pial_left, fsaverage.pial_right]
    bgs = [fsaverage.sulc_left, fsaverage.sulc_right]

    hemis = ["left", "right"]
    views = ["lateral", "medial"]

    for ax_i, ax in enumerate(axes.flatten()):
        texture = vol_to_surf(surf_map, surf_def[ax_i % 2], interpolation="nearest")
        plot_surf_stat_map(
            hemis_def[ax_i % 2],
            texture,
            hemi=hemis[ax_i % 2],
            view=views[ax_i // 2],
            bg_map=bgs[ax_i % 2],
            axes=ax,
            **kwargs,
        )

    if saveloc is not None:
        fig.savefig(saveloc, dpi=300, bbox_inches="tight")

    return fig, axes
