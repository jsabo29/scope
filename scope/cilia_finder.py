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


def detect_cilia(vol: np.ndarray, min_cilia_vol, spacing=(1.0, 1.0, 1.0),
                 bright_threshold: float = 0.99, dim_threshold: float | None = None):
    """Group bright voxels into cilia by flood fill and keep the large enough ones.

    `vol` is the (Z, Y, X) brightness array from `load_cilia_volume` (expected
    normalised to [0, 1]; pass ``as_float=True`` when loading). Starting from each
    unvisited bright voxel, the region grows over its 6 face-connected neighbours:
    a bright voxel always keeps growing, while a dim voxel (brightness in
    ``[dim_threshold, bright_threshold)``) is absorbed only when it was reached
    directly from a bright voxel — so a single dim voxel can bridge a gap between
    bright stretches of one cilium, but chains of dim voxels cannot. Voxels below
    `dim_threshold` are background: they are never absorbed and act as walls.

    `bright_threshold` / `dim_threshold` set those two brightness cut-offs.
    `dim_threshold` defaults to `bright_threshold`, which disables bridging
    entirely (plain 6-connected bright components); lower it to let dim voxels
    bridge gaps. This split is what stops the fill from swallowing the background
    shell around every bright voxel.

    `spacing` is the physical voxel size (z_um, y_um, x_um) — pass
    ``(czi_data["z_um"], czi_data["um_per_px"], czi_data["um_per_px"])`` to work in
    microns. A region is kept only if its physical volume exceeds `min_cilia_vol`
    (in µm³); with the default unit spacing that is simply a voxel count.

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
    Z, Y, X = vol.shape
    if dim_threshold is None:
        dim_threshold = bright_threshold
    bright = vol >= bright_threshold
    signal = vol >= dim_threshold          # bright OR bridgeable dim (not background)
    visited = np.zeros(vol.shape, dtype=bool)
    voxvol_um3 = float(spacing[0]) * float(spacing[1]) * float(spacing[2])

    cilia = []
    # Only bright voxels can seed a region, and they are sparse, so iterate those
    # directly instead of walking the whole volume in Python.
    for seed in np.argwhere(bright):
        sz, sy, sx = int(seed[0]), int(seed[1]), int(seed[2])
        if visited[sz, sy, sx]:
            continue

        # Iterative flood fill (an explicit stack avoids Python's recursion limit
        # on long cilia). Each entry carries whether it was reached from a bright
        # voxel, which is what lets a lone dim voxel bridge but not a dim chain.
        members = []
        stack = [(sz, sy, sx, True)]
        while stack:
            z, y, x, from_bright = stack.pop()
            if visited[z, y, x]:
                continue
            visited[z, y, x] = True
            members.append((z, y, x))

            is_bright = bool(bright[z, y, x])
            # A dim voxel not reached from a bright one is a dead end: count it but
            # don't grow through it. Bright voxels always grow.
            if not is_bright and not from_bright:
                continue
            # Grow only into signal voxels — background (< dim_threshold) is a wall.
            for dz, dy, dx in _NEIGHBOURS_6:
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

    cilia.sort(key=lambda c: c["n_voxels"], reverse=True)
    return cilia
