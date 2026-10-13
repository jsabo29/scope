"""Cilia detection on a 3-D brightness volume.

Two entry points:

``load_cilia_volume``
    Pull the cilia channel off disk one plane at a time into a (Z, Y, X) array.

``detect_cilia``
    Group bright voxels into connected cilia and keep the physically large
    enough ones.

Detection semantics (identical for both back-ends)
--------------------------------------------------
Two brightness cut-offs split the volume into three classes:

    bright      vol >= bright_threshold      seeds and grows a component
    dim         dim_threshold <= vol < bright_threshold
                                             may be absorbed near a bright voxel
    background  vol <  dim_threshold         a wall; never absorbed

A *cilium* is a connected set of bright voxels, where two bright stretches
count as connected if the dim voxels separating them can be crossed within
``dim_bridge`` growth steps from each side. Dim voxels reached that way join
the component (so the measured object includes its faint tapering tip) but a
run of dim voxels on its own can never form or extend a component — that is
what stops the fill leaking into the background haze that surrounds every
bright voxel.

The default back-end (``method="dilate"``) implements this as a geodesic
(mask-constrained) binary dilation of the bright mask into the signal mask,
followed by connected-component labelling. It runs in C, and — unlike the
original recursive fill — its output does not depend on voxel visit order.

The original explicit-stack flood fill is kept as ``method="floodfill"`` so
previous results can be reproduced. It is order-dependent by construction: a
dim voxel first reached along a dim path is permanently marked visited and
will not expand even if a bright voxel would have reached it a moment later.
"""

from __future__ import annotations

import numpy as np


def load_cilia_volume(reader, channel, n_z: int, *,
                      as_float: bool = True,
                      progress=None) -> np.ndarray:
    """Read the whole cilia channel into a 3-D brightness array.

    Parameters
    ----------
    reader
        Any object exposing ``_raw_slice(channel_index, z) -> 2-D (Y, X)`` — i.e.
        the app's ``SliceReader``. Each call pulls one plane from disk.
    channel
        The cilia channel to read. Either the absolute channel index (an int), or
        the mapping dict from the channel-map dialog, in which case
        ``mapping["Cilia"]`` is used.
    n_z
        Number of Z slices to read (``czi_data["n_z"]`` in the app).
    as_float
        If True (default), return float32 scaled to [0, 1] (divided by the dtype's
        max, or by the volume max for float inputs). If False, keep the file's
        native dtype and true pixel values.
    progress
        Optional ``callback(fraction_0_to_1, message)`` for a progress bar.

    Returns
    -------
    numpy.ndarray
        A ``(Z, Y, X)`` array of voxel brightness. ``vol[z, y, x]`` is one voxel.
    """
    c_idx = int(channel["Cilia"]) if isinstance(channel, dict) else int(channel)

    if n_z is None or int(n_z) <= 0:
        raise ValueError(f"n_z must be a positive integer, got {n_z!r}")
    n_z = int(n_z)

    # Read the first plane to learn the height, width and dtype, then allocate the
    # whole volume once and fill it slice by slice (RAM ≈ one full volume).
    first = np.squeeze(np.asarray(reader._raw_slice(c_idx, 0)))
    if first.ndim != 2:
        raise ValueError(
            f"expected a 2-D (Y, X) plane from the reader, got shape {first.shape}")
    H, W = int(first.shape[0]), int(first.shape[1])

    vol = np.empty((n_z, H, W), dtype=first.dtype)
    vol[0] = first
    for z in range(1, n_z):
        if progress:
            progress(z / n_z, f"reading cilia slice {z + 1}/{n_z}")
        plane = np.squeeze(np.asarray(reader._raw_slice(c_idx, z)))
        vol[z] = plane.reshape(H, W)
    if progress:
        progress(1.0, "done")

    if as_float:
        vol = _to_unit_float(vol)
    return vol


def _to_unit_float(vol: np.ndarray) -> np.ndarray:
    """Scale a brightness array to float32 in [0, 1]."""
    a = vol.astype(np.float32)
    if np.issubdtype(vol.dtype, np.integer):
        info = np.iinfo(vol.dtype)
        span = float(info.max - info.min)
        if span > 0:
            return (a - float(info.min)) / span
        return np.zeros_like(a)
    hi = float(a.max())
    return a / hi if hi > 0 else a


# 6-connectivity: only the voxels sharing a face (no edge/corner diagonals).
_NEIGHBOURS_6 = (
    (-1, 0, 0), (1, 0, 0),
    (0, -1, 0), (0, 1, 0),
    (0, 0, -1), (0, 0, 1),
)

# 26-connectivity: every neighbour including edge/corner diagonals. Needed to keep
# a thin cilium that runs diagonally (through z and across the plane) connected —
# under 6-connectivity such a line touches only at corners and fragments apart.
_NEIGHBOURS_26 = tuple(
    (dz, dy, dx)
    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    if not (dz == dy == dx == 0)
)


def _structure(connectivity: int) -> np.ndarray:
    """Binary structuring element matching 6- or 26-connectivity."""
    if int(connectivity) >= 26:
        return np.ones((3, 3, 3), dtype=bool)
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1, 1, :] = True
    st[1, :, 1] = True
    st[:, 1, 1] = True
    return st


def detect_cilia(vol: np.ndarray, min_cilia_vol, spacing=(1.0, 1.0, 1.0),
                 bright_threshold: float = 0.99, dim_threshold: float | None = None,
                 connectivity: int = 26, dim_bridge: int = 1,
                 method: str = "dilate"):
    """Group bright voxels into cilia and keep the physically large enough ones.

    Parameters
    ----------
    vol
        (Z, Y, X) brightness array, expected normalised to [0, 1].
    min_cilia_vol
        Minimum physical volume in µm³ for a component to be kept. With the
        default unit `spacing` this is simply a voxel count.
    spacing
        Physical voxel size (z_um, y_um, x_um) — pass
        ``(czi_data["z_um"], czi_data["um_per_px"], czi_data["um_per_px"])``.
    bright_threshold, dim_threshold
        The two brightness cut-offs described in the module docstring.
        ``dim_threshold=None`` disables bridging (plain bright components).
    connectivity
        6 = faces only (compact blobs) or 26 = include edge/corner diagonals.
        Use 26 for thin cilia running diagonally through the stack, otherwise a
        diagonal line touches only at corners and fragments into sub-threshold
        pieces that the minimum-volume filter then discards.
    dim_bridge
        Growth steps into dim signal from each side of a bright stretch. Two
        bright stretches separated by up to ``2 * dim_bridge`` dim voxels are
        joined. 0 disables dim absorption entirely.
    method
        ``"dilate"`` (default) — deterministic geodesic dilation + labelling.
        ``"floodfill"`` — the original order-dependent explicit-stack fill,
        kept so earlier results can be reproduced.

    Returns
    -------
    list[dict]
        One dict per detected cilium, largest first, with keys:
          ``voxels``      – (N, 3) int array of member (z, y, x) coordinates
          ``n_voxels``    – number of member voxels
          ``volume_um3``  – physical volume (n_voxels × voxel volume)
          ``centroid``    – (z, y, x) mean coordinate
    """
    vol = np.asarray(vol)
    if dim_threshold is None:
        dim_threshold = bright_threshold
    voxvol_um3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])

    if str(method).lower().startswith("flood"):
        cilia = _detect_floodfill(vol, min_cilia_vol, voxvol_um3,
                                  bright_threshold, dim_threshold, connectivity)
    else:
        cilia = _detect_dilate(vol, min_cilia_vol, voxvol_um3,
                               bright_threshold, dim_threshold, connectivity,
                               int(max(0, dim_bridge)))

    # Largest first. The centroid tie-break keeps the order reproducible when two
    # components have exactly the same voxel count — which matters because the
    # downstream cilium-to-cell matching is greedy in this order.
    cilia.sort(key=lambda c: (-c["n_voxels"], c["centroid"]))
    return cilia


# ---------------------------------------------------------------------------
# Default back-end: geodesic dilation + connected components (deterministic)
# ---------------------------------------------------------------------------

def _detect_dilate(vol, min_cilia_vol, voxvol_um3, bright_threshold,
                   dim_threshold, connectivity, dim_bridge):
    from scipy import ndimage as ndi

    struct = _structure(connectivity)
    bright = vol >= bright_threshold
    if not bright.any():
        return []

    if dim_bridge > 0 and dim_threshold < bright_threshold:
        signal = vol >= dim_threshold
        # Mask-constrained ("geodesic") dilation: bright grows outward, but only
        # through dim signal and never into background. Each side of a gap grows
        # dim_bridge steps, so gaps of up to 2*dim_bridge dim voxels close up.
        grown = ndi.binary_dilation(bright, structure=struct,
                                    iterations=dim_bridge, mask=signal)
        del signal
    else:
        grown = bright

    labels, n = ndi.label(grown, structure=struct) #type: ignore
    del grown
    if n == 0:
        return []

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = np.flatnonzero(sizes * voxvol_um3 > min_cilia_vol)
    if keep.size == 0:
        return []

    # Pull the member coordinates for every surviving label in one pass, then
    # split them by label — cheaper than masking the volume once per component.
    coords = np.argwhere(np.isin(labels, keep))
    if coords.size == 0:
        return []
    lab_of = labels[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(lab_of, kind="stable")
    coords = coords[order]
    lab_of = lab_of[order]
    starts = np.searchsorted(lab_of, keep, side="left")
    stops = np.searchsorted(lab_of, keep, side="right")

    cilia = []
    for a, b in zip(starts, stops):
        if b <= a:
            continue
        arr = np.ascontiguousarray(coords[a:b], dtype=np.intp)
        cilia.append({
            "voxels": arr,
            "n_voxels": int(arr.shape[0]),
            "volume_um3": float(arr.shape[0]) * voxvol_um3,
            "centroid": tuple(arr.mean(axis=0)),
        })
    return cilia


# ---------------------------------------------------------------------------
# Legacy back-end: explicit-stack flood fill (order-dependent, kept for parity)
# ---------------------------------------------------------------------------

def _detect_floodfill(vol, min_cilia_vol, voxvol_um3, bright_threshold,
                      dim_threshold, connectivity):
    Z, Y, X = vol.shape
    neighbours = _NEIGHBOURS_26 if int(connectivity) >= 26 else _NEIGHBOURS_6
    bright = vol >= bright_threshold
    signal = vol >= dim_threshold          # bright OR bridgeable dim (not background)
    visited = np.zeros(vol.shape, dtype=bool)

    cilia = []
    # Only bright voxels can seed a region, and they are sparse, so iterate those
    # directly instead of walking the whole volume in Python.
    for seed in np.argwhere(bright):
        sz, sy, sx = int(seed[0]), int(seed[1]), int(seed[2])
        if visited[sz, sy, sx]:
            continue

        members = []
        stack = [(sz, sy, sx, True)]
        while stack:
            z, y, x, from_bright = stack.pop()
            if visited[z, y, x]:
                continue
            visited[z, y, x] = True
            members.append((z, y, x))

            is_bright = bool(bright[z, y, x])
            # A dim voxel not reached from a bright one is a dead end: count it
            # but don't grow through it. Bright voxels always grow.
            if not is_bright and not from_bright:
                continue
            # Grow only into signal voxels — background (< dim_threshold) is a wall.
            for dz, dy, dx in neighbours:
                nz, ny, nx = z + dz, y + dy, x + dx
                if (0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X
                        and not visited[nz, ny, nx] and signal[nz, ny, nx]):
                    stack.append((nz, ny, nx, is_bright))

        if len(members) * voxvol_um3 > min_cilia_vol:
            arr = np.asarray(members, dtype=np.intp)
            cilia.append({
                "voxels":     arr,
                "n_voxels":   len(members),
                "volume_um3": len(members) * voxvol_um3,
                "centroid":   tuple(arr.mean(axis=0)),
            })
    return cilia
