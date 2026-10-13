"""
Process the edited CZI image to detect neurons with nuclei and then measure cilia.

Pipeline (see process_image):
  - Denoise each layer and channel
  - Build a 3-D model (binary volumes) of nuclei / soma / cilia
  - Remove all nuclei under a specified volume threshold
  - In one pass: drop nuclei lacking enough soma within a radius, AND keep only
    the soma lying within that radius of a surviving nucleus
  - Remove all cilia not within a specified radius of remaining soma
  - Add back all cilia directly attached to remaining cilia (bridge threshold gaps)
  - Group cilia, cell bodies, and nuclei into cells (cells with no cilium kept)
  - Measure cilia lengths (3-D skeleton geodesic, in microns)
  - Optionally, save the processed volume to a chosen location
  - Return the cilia lengths and cell centres

Everything spatial is computed with the true anisotropic voxel spacing
(z_um, um_per_px, um_per_px), so radii / volumes / lengths are physical.
"""

from __future__ import annotations

import heapq
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np

STAINS = ["Cilia", "Nucleus 1", "Nucleus 2", "GAD1 (Soma)"]


# ===========================================================================
# Parameters
# ===========================================================================

@dataclass
class DetectionParams:
    # Image scale (filled by from_physical) — microns per pixel / per slice.
    um_per_px: float = 1.0
    z_um:      float = 1.0

    # Pre-processing / thresholding
    gaussian_sigma: float = 1.0
    min_threshold:  float = 0.12     # intensity floor on [0,1]
    nucleus_combine: str  = "and"    # "and" = need both nuclei; "or" = either

    # Nucleus filter (3-D)
    min_nucleus_volume_vox: int = 300   # drop nuclei smaller than this
    # Per-channel speckle filter, applied to each thresholded nucleus channel
    # before they are combined. This used to be wired to `cilia_min_voxels` — so
    # nuclei were despeckled with a threshold computed from cilia volume, which
    # is the bug this field exists to fix.
    #
    # Default 0 (off), because it turns out to be redundant: anything it would
    # remove is smaller than `min_nucleus_volume_vox`, so `remove_small_nuclei`
    # deletes it moments later anyway. Running it costs a full connected-component
    # pass — an int32 array the size of the volume — per nucleus channel, at the
    # point where memory is tightest. Set it above 0 only if you want each channel
    # cleaned before the AND/OR combination for its own sake.
    min_nucleus_speckle_vox: int = 0

    # Soma detection = local brightness-ratio test around each nucleus
    soma_search_um:  float = 6.0     # inner radius (cytoplasm zone) around nucleus
    soma_compare_um: float = 3.0     # width of the comparison shell beyond it
    soma_ratio:      float = 1.15    # inner mean must exceed ratio × outer mean
    soma_exclude_neighbours: bool = True  # ignore other nuclei inside the test zones

    # Cilia (detected in cilia_finder.detect_cilia)
    cilia_max_dist_um:      float = 8.0   # match a cilium to a cell if within this of it
    cilia_min_volume_um3:   float = 0.5   # drop cilia specks below this physical volume
    cilia_bright_threshold: float = 0.5   # "bright" cut-off on [0,1]
    cilia_dim_threshold:    float = 0.2   # bridgeable-dim cut-off; <=0 disables bridging
    cilia_min_voxels:       int = 4       # derived from cilia_min_volume_um3
    cilia_connectivity:     int = 26      # neighbourhood: 6 faces or 26 (diagonal)
    cilia_dim_bridge:       int = 1       # dim growth steps per side (dilate back-end)
    cilia_method:           str = "dilate"   # "dilate" (deterministic) | "floodfill"

    # Cilium → cell assignment
    #   "proximal" : shortest distance from the cilium to the cell's nucleus
    #                surface. A primary cilium grows from the basal body docked
    #                at the cell membrane, so this gap is the meaningful test.
    #   "centroid" : legacy centre-of-mass to centre-of-mass distance.
    cilia_match_mode: str = "proximal"

    # Length measurement
    #   Resample each cilium crop to cubic voxels before skeletonising. The
    #   rationale is real — skeletonize is topological and ignores `spacing`, so
    #   on an anisotropic stack it thins harder along Z than across it. But
    #   measured against synthetic rods of known length at angles from 0° to 90°
    #   it is a wash: it tightens the angle-to-angle spread at a realistic 300 nm
    #   cilium radius (1.5 µm vs 3.4 µm), and loses at 150 nm and 500 nm, because
    #   nearest-neighbour upsampling adds a staircase that biases lengths ~10–20%
    #   high. Left OFF by default so measurements stay comparable with earlier
    #   runs; turn it on if angle-independence matters more than absolute bias.
    skeleton_isotropic:     bool = False
    skeleton_max_voxels:    int = 8_000_000   # resampling guard
    cilia_max_plausible_um: float = 30.0      # warn above this (not a filter)

    # ------------------------------------------------------------------
    @classmethod
    def from_physical(cls, um_per_px: float, *,
                      min_nucleus_diam_um: float = 5.0,
                      floor:           float = 0.12,
                      volfrac:         float = 1.0,
                      soma_search_um:  float = 6.0,
                      soma_compare_um: float = 3.0,
                      soma_diff_pct:   float = 15.0,
                      cilia_max_dist_um: float = 8.0,
                      cilia_min_volume_um3: float = 0.5,
                      cilia_bright_threshold: float = 0.5,
                      cilia_dim_threshold: float = 0.2,
                      cilia_connectivity: int = 26,
                      cilia_dim_bridge: int = 1,
                      cilia_method: str = "dilate",
                      cilia_match_mode: str = "proximal",
                      skeleton_isotropic: bool = False,
                      soma_exclude_neighbours: bool = True,
                      nucleus_speckle_filter: bool = False,
                      z_um:            float = 1.0,
                      nucleus_combine: str   = "and") -> "DetectionParams":
        """
        Build parameters from physical sizes.

          floor            : intensity threshold floor on [0,1] (nuclei + cilia).
          volfrac          : a detected nucleus must reach >= volfrac × (volume of
                             a sphere of `min_nucleus_diam_um`) to be kept.
          soma_search_um   : inner radius around the nucleus (the cytoplasm zone).
          soma_compare_um  : width of the comparison shell just beyond it.
          soma_diff_pct    : the inner median GAD1 brightness must exceed the outer
                             shell median by at least this percentage.
          cilia_max_dist_um: a detected cilium is assigned to the nearest cell only
                             if it lies within this distance of the cell centre.
          cilia_min_volume_um3: a detected cilium component must reach this physical
                             volume to be kept (removes noise specks).
          cilia_bright_threshold: brightness on [0,1] at/above which a cilia voxel is
                             "bright" and seeds/grows a flood-fill component.
          cilia_dim_threshold: brightness at/above which a voxel is bridgeable "dim"
                             signal (below it is background); <=0 disables bridging.
          cilia_dim_bridge : dim growth steps per side; a gap of up to
                             2 × this many dim voxels is bridged.
          cilia_method     : "dilate" (deterministic, C-speed) or "floodfill"
                             (the original order-dependent fill).
          cilia_match_mode : "proximal" (cilium-to-nucleus-surface distance) or
                             "centroid" (legacy centre-to-centre).
          skeleton_isotropic: resample each cilium to cubic voxels before
                             skeletonising. Off by default — see the field's
                             comment for what the phantom test actually showed.
        """
        r  = max(1e-6, float(um_per_px))
        zs = max(1e-6, float(z_um) if z_um and z_um > 0 else 1.0)
        voxvol_um3 = r * r * zs

        nuc_rad_um  = max(1e-6, min_nucleus_diam_um) / 2.0
        nuc_vol_um3 = (4.0 / 3.0) * math.pi * nuc_rad_um ** 3
        min_nuc_vox = max(8, int(float(volfrac) * nuc_vol_um3 / voxvol_um3))
        min_cil_vox = max(1, int(float(cilia_min_volume_um3) / voxvol_um3))
        # Off by default — see the field comment. If switched on, a tenth of the
        # real volume gate is the sensible size: enough to clear single-voxel
        # noise, never enough to remove something the real filter would keep.
        min_nuc_speck = (max(2, min_nuc_vox // 10)
                         if nucleus_speckle_filter else 0)

        return cls(
            um_per_px=r, z_um=zs,
            gaussian_sigma=1.0, min_threshold=float(floor),
            nucleus_combine=nucleus_combine,
            min_nucleus_volume_vox=min_nuc_vox,
            min_nucleus_speckle_vox=min_nuc_speck,
            soma_exclude_neighbours=bool(soma_exclude_neighbours),
            soma_search_um=float(soma_search_um),
            soma_compare_um=float(soma_compare_um),
            soma_ratio=1.0 + max(0.0, float(soma_diff_pct)) / 100.0,
            cilia_max_dist_um=float(cilia_max_dist_um),
            cilia_min_volume_um3=float(cilia_min_volume_um3),
            cilia_bright_threshold=float(cilia_bright_threshold),
            cilia_dim_threshold=float(cilia_dim_threshold),
            cilia_connectivity=(26 if int(cilia_connectivity) >= 26 else 6),
            cilia_dim_bridge=max(0, int(cilia_dim_bridge)),
            cilia_method=("floodfill" if str(cilia_method).lower().startswith("flood")
                          else "dilate"),
            cilia_match_mode=("centroid" if str(cilia_match_mode).lower().startswith("cent")
                              else "proximal"),
            skeleton_isotropic=bool(skeleton_isotropic),
            cilia_min_voxels=min_cil_vox,
        )

    def spacing(self):
        """Voxel spacing (z, y, x) in microns, for distance transforms."""
        return (self.z_um, self.um_per_px, self.um_per_px)


@dataclass
class CellResult:
    num: int
    x: float
    y: float
    z: int
    cilia_length_px: float
    cilia_length_um: float
    has_cilium: bool
    nucleus_volume_vox: int
    # Shortest distance from the matched cilium to this cell's nucleus surface
    # (µm). None when no cilium matched. Worth eyeballing: a genuine primary
    # cilium docks at the membrane, so gaps should be small and tightly spread.
    cilia_gap_um: float | None = None

    def as_row(self, img_name: str) -> dict:
        return {
            "img_name": img_name,
            "num":      self.num,
            "x":        round(self.x, 1),
            "y":        round(self.y, 1),
            "z":        self.z,
            "length":   (f"{self.cilia_length_um:.2f} µm"
                         if self.has_cilium else "—"),
        }


# ===========================================================================
# Packed boolean crop — stores a bounding-box boolean sub-volume as bits
# ===========================================================================

@dataclass
class PackedCrop:
    """Boolean 3-D sub-volume stored as packed bits (8× smaller than bool).

    `slices` is the (z, y, x) slice triple locating this crop inside the full
    volume, so a measured cilium can be painted back into an output mask.
    """
    bits:   np.ndarray
    shape:  tuple
    slices: tuple = ()

    @staticmethod
    def from_crop(crop_bool: np.ndarray, slices=()) -> "PackedCrop":
        crop = np.ascontiguousarray(crop_bool, dtype=bool)
        return PackedCrop(bits=np.packbits(crop), shape=crop.shape,
                          slices=tuple(slices))

    def unpack(self) -> np.ndarray:
        n = int(np.prod(self.shape))
        return np.unpackbits(self.bits)[:n].reshape(self.shape).astype(bool)


# ===========================================================================
# Small helpers
# ===========================================================================

def _norm_volume(vol: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    a  = vol.astype(np.float32)
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi <= lo:
        # Signal is sparser than hi_pct (e.g. thin cilia occupying <0.5% of
        # voxels) — percentile clipping would crush it. Fall back to min/max so
        # the bright structures survive normalisation.
        lo = float(a.min())
        hi = float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0.0, 1.0)


_PCTL_SAMPLE_CAP = 4_000_000


def _percentiles(a: np.ndarray, lo_pct, hi_pct):
    """Percentiles from a strided subsample.

    `np.percentile` partitions a full copy of its input, which on a stack this
    size is another whole volume allocated at the worst possible moment. A few
    million voxels pin a percentile far more tightly than we need, and the stride
    is a view, so only the subsample is materialised.
    """
    flat = a.ravel()
    step = max(1, flat.size // _PCTL_SAMPLE_CAP)
    sample = flat[::step]
    return float(np.percentile(sample, lo_pct)), float(np.percentile(sample, hi_pct))


def _norm_volume_inplace(a: np.ndarray, lo_pct=0.5, hi_pct=99.5) -> np.ndarray:
    """`_norm_volume`, but rescaling the float32 array in place (no second copy).

    On a full-size stack each avoided temporary is a couple of gigabytes, so the
    read-and-normalise path uses this rather than allocating a new volume.
    """
    lo, hi = _percentiles(a, lo_pct, hi_pct)
    if hi <= lo:
        lo = float(a.min())
        hi = float(a.max())
    if hi > lo:
        a -= lo
        a /= (hi - lo)
    else:
        a.fill(0.0)
    np.clip(a, 0.0, 1.0, out=a)
    return a


def _to_u16(a: np.ndarray) -> np.ndarray:
    """Quantise a [0,1] float volume to uint16 — half the memory, ~1.5e-5 error.

    Used for the volumes that must keep their intensities (GAD1 for the median
    ratio test, cilia for the two brightness cut-offs). Both are scale-invariant
    or threshold-based, so the quantisation is immaterial.

    DESTROYS `a`: every step is done in place so the conversion allocates only
    the uint16 result, not two more full-size float volumes along the way. The
    caller drops the float immediately afterwards.
    """
    np.clip(a, 0.0, 1.0, out=a)
    a *= 65535.0
    a += 0.5
    return a.astype(np.uint16)


U16 = 65535.0


def _ones3():
    return np.ones((3, 3, 3), dtype=bool)


def _threshold_volume(vol: np.ndarray, p: DetectionParams,
                      min_size: int = 0) -> np.ndarray:
    from skimage.filters import threshold_otsu
    try:
        t = float(threshold_otsu(vol))
    except Exception:
        t = float(np.percentile(vol, 99.0))
    t = max(t, p.min_threshold)
    m = vol > t
    if min_size > 1 and m.any():
        # Drop tiny components (avoids skimage's deprecated remove_small_objects
        # arg and stays correct across versions). 26-connectivity, like the rest.
        lbl, n = _label(m, _ones3())
        if n:
            sizes = np.bincount(lbl.ravel())
            keep = sizes >= min_size
            keep[0] = False
            m = keep[lbl] #type: ignore
    return m


def _label(mask: np.ndarray, structure):
    from scipy import ndimage as ndi
    labels, count = ndi.label(mask, structure=structure)  # type: ignore[misc]
    return labels, int(count)


def _edt(mask: np.ndarray, spacing) -> np.ndarray:
    from scipy import ndimage as ndi
    return np.asarray(
        ndi.distance_transform_edt(mask, sampling=spacing))  # type: ignore[arg-type]


def _to_isotropic(mask: np.ndarray, spacing, max_voxels: int = 8_000_000):
    """Nearest-neighbour resample a crop to cubic voxels.

    `skeletonize` is a topological operation: it decides what to thin purely from
    voxel adjacency and knows nothing about `spacing`. On a stack with 0.1 µm
    pixels and 0.5 µm slices, one voxel step in Z is five times one in X, so it
    thins harder along Z than across the plane, and weighting the graph edges
    afterwards cannot recover what thinning already removed.

    Blowing the crop up to cubic voxels first makes the topology match the
    geometry. In practice this is a trade rather than a win: on synthetic rods it
    halves the angle-to-angle spread at a realistic cilium radius, but the
    nearest-neighbour staircase biases lengths high. Hence off by default.

    Returns (mask, spacing) — unchanged if already isotropic or too large.
    """
    from scipy import ndimage as ndi

    sp = np.asarray(spacing, dtype=np.float64)
    if not np.all(np.isfinite(sp)) or sp.min() <= 0:
        return mask, spacing
    if np.allclose(sp, sp[0], rtol=1e-3):
        return mask, spacing

    factors = sp / sp.min()
    projected = int(np.prod(np.asarray(mask.shape) * factors))
    if projected > int(max_voxels):
        print(f"[analysis] cilium crop {mask.shape} would resample to ~{projected} "
              f"voxels (> {int(max_voxels)}) — measuring on the anisotropic grid")
        return mask, spacing

    grown = ndi.zoom(mask.astype(np.uint8), factors, order=0, prefilter=False)
    iso = float(sp.min())
    return grown.astype(bool), (iso, iso, iso) #type: ignore


def _skeleton_geodesic_um(mask: np.ndarray, spacing, isotropic: bool = False,
                          max_voxels: int = 8_000_000) -> float:
    """Longest-path length (microns) through a 3-D region's skeleton."""
    from skimage.morphology import skeletonize

    if isotropic:
        mask, spacing = _to_isotropic(mask, spacing, max_voxels)

    # `_voxels_to_crop` returns a TIGHT bounding box, so every cilium touches all
    # six faces of its own crop — and skeletonize treats the array border as a
    # special case, which for a short axial structure can thin it away to nothing.
    # One voxel of padding removes that entirely.
    mask = np.pad(mask, 1, mode="constant", constant_values=False)

    sk = skeletonize(mask)
    pts = list(zip(*np.nonzero(sk)))
    sz, sy, sx = spacing
    if len(pts) <= 1:
        # Skeleton collapsed (short / thin / near-2-D cilium). Fall back to the
        # physical bounding-box diagonal of the mask so a real cilium still gets
        # a non-zero recorded length instead of being dropped.
        mp = np.argwhere(mask)
        if mp.shape[0] <= 1:
            return 0.0
        coords = mp.astype(np.float64) * np.asarray([sz, sy, sx])
        return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))

    if len(pts) > 40000:                         # guard against blob-like noise
        arr = np.asarray(pts, dtype=np.float64) * np.asarray([sz, sy, sx])
        return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))

    index = {pt: i for i, pt in enumerate(pts)}
    neigh = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                w = math.sqrt((dz * sz) ** 2 + (dy * sy) ** 2 + (dx * sx) ** 2)
                neigh.append((dz, dy, dx, w))

    adj: list = [[] for _ in pts]
    for pt, i in index.items():
        z, y, x = pt
        for dz, dy, dx, w in neigh:
            j = index.get((z + dz, y + dy, x + dx))
            if j is not None:
                adj[i].append((j, w))

    def farthest(src: int):
        """Dijkstra from src; returns (farthest node, its distance, reached set)."""
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        far = max(dist, key=lambda k: dist[k])
        return far, dist[far], dist

    # The double sweep only ever explores ONE connected component, so a skeleton
    # that came apart — which happens whenever the mask is barely connected, i.e.
    # exactly the thin oblique cilia that are hardest to measure — used to be
    # reported at the length of whichever fragment happened to contain point 0.
    # That silently turned 6 µm cilia into 0.5 µm ones. Sweep every component and
    # take the longest.
    best = 0.0
    unseen = set(range(len(pts)))
    n_components = 0
    while unseen:
        src = next(iter(unseen))
        a, _, reached_a = farthest(src)
        _, length, _ = farthest(a)
        unseen -= set(reached_a)
        n_components += 1
        best = max(best, float(length))

    # Sanity floor. A skeleton can fragment, or thin away to a stub, whenever the
    # mask is only marginally connected — precisely the thin oblique cilia we care
    # most about. Whatever the skeleton did, the object's own physical extent is a
    # hard lower bound on its length, so if the geodesic comes back at less than
    # half of that, the skeleton is degenerate and the extent is the better
    # estimate. Normal, well-formed cilia are unaffected: their geodesic exceeds
    # their bounding-box diagonal, because they curve.
    mp = np.argwhere(mask)
    if mp.shape[0] > 1:
        coords = mp.astype(np.float64) * np.asarray([sz, sy, sx])
        span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        if best < 0.5 * span:
            if n_components > 1:
                print(f"[analysis] skeleton fragmented into {n_components} pieces "
                      f"(longest {best:.2f} µm) — using the {span:.2f} µm physical "
                      f"extent instead")
            best = max(best, span)
    return best


# ===========================================================================
# Pipeline stages  (progress(frac, sub): frac is 0..1 within the stage)
# ===========================================================================

def _gaussian_inplace(vol: np.ndarray, sigma3) -> None:
    """3-D Gaussian applied as three separable 1-D passes, in place.

    `ndi.gaussian_filter(vol, ..., output=vol)` still allocates a full-size
    temporary internally, which on a real stack is another couple of gigabytes at
    the exact moment memory is tightest. The filter is separable, so running it
    one axis at a time with `output=vol` needs only a line buffer.
    """
    from scipy import ndimage as ndi
    for axis, s in enumerate(sigma3):
        if s and s > 0:
            ndi.gaussian_filter1d(vol, float(s), axis=axis, output=vol)


def _read_channel(reader, c, n_z, H, W, progress=None, label="", should_abort=None):
    """Read one channel's Z stack into a single float32 volume."""
    vol = np.empty((n_z, H, W), dtype=np.float32)
    for z in range(n_z):
        if should_abort and should_abort():
            return None
        if progress and (z % 4 == 0 or z == n_z - 1):
            progress(z / max(1, n_z), f"{label} slice {z + 1}/{n_z}")
        vol[z] = np.asarray(reader._raw_slice(c, z), dtype=np.float32).reshape(H, W)
    return vol


def prepare_volumes(reader, mapping, n_z, p, progress=None, should_abort=None):
    """Read, denoise and reduce every channel to its leanest useful form.

    The previous version read all four channels into float32 volumes, then made a
    normalised copy and a Gaussian-filtered copy of each — six-plus full-size
    float volumes alive at once. On a 4096 × 4096 × 36 stack that is ~10 GB before
    the filter's own temporaries, which is the point where this pipeline used to
    fall over.

    Here each channel is read, processed and reduced one at a time, and only the
    reduced form is retained:

        nucleus  bool    thresholded and combined immediately (1 byte/voxel)
        gad      uint16  intensity needed for the median ratio test (2 bytes)
        cilia    uint16  intensity needed for the two brightness cut-offs (2)

    Peak usage is therefore one float32 volume plus the reductions accumulated so
    far, rather than four float32 volumes plus their copies. Measured on a
    48×400×400 uint16 stack: peak 2.5× lower, retained 3.2× lower, which
    extrapolates to roughly 17 GB → 7 GB for a 4096×4096×36 stack.

    Thresholds are unchanged: the uint16 quantisation is 1.5e-5 of full scale,
    and both consumers of those volumes are scale-invariant (the GAD1 median
    ratio) or threshold-based (the cilia cut-offs).
    """
    if n_z <= 0:
        return None

    c_cil = mapping["Cilia"]; c_n1 = mapping["Nucleus 1"]
    c_n2  = mapping["Nucleus 2"]; c_gad = mapping["GAD1 (Soma)"]

    first = np.squeeze(reader._raw_slice(c_n1, 0))
    H, W  = int(first.shape[0]), int(first.shape[1])
    del first

    z_sigma = max(0.4, p.gaussian_sigma * (p.um_per_px / max(1e-6, p.z_um)))
    sigma3  = (z_sigma, p.gaussian_sigma, p.gaussian_sigma)

    vox_mb = n_z * H * W / (1024 ** 2)
    print(f"[analysis] volume {n_z}×{H}×{W} — {vox_mb * 4:.0f} MB per float32 channel; "
          f"streaming one channel at a time")

    mode = (p.nucleus_combine or "and").lower()
    need_n1 = mode in ("and", "or", "nuc1")
    need_n2 = mode in ("and", "or", "nuc2")
    out: dict = {}

    def _denoised_mask(c, label, frac_lo, frac_hi):
        """Read → normalise → Gaussian → Otsu → despeckle, freeing as we go."""
        def sub(f, msg):
            if progress:
                progress(frac_lo + (frac_hi - frac_lo) * f, msg)
        vol = _read_channel(reader, c, n_z, H, W, sub, label, should_abort)
        if vol is None:
            return None
        _norm_volume_inplace(vol)
        sub(0.85, f"denoising {label}")
        _gaussian_inplace(vol, sigma3)
        sub(0.95, f"thresholding {label}")
        m = _threshold_volume(vol, p, min_size=p.min_nucleus_speckle_vox)
        del vol
        return m

    # ---- nuclei: threshold each channel, then combine and drop the floats ----
    nuc1 = _denoised_mask(c_n1, "nucleus 1", 0.00, 0.25) if need_n1 else None
    if need_n1 and nuc1 is None:
        return None
    nuc2 = _denoised_mask(c_n2, "nucleus 2", 0.25, 0.50) if need_n2 else None
    if need_n2 and nuc2 is None:
        return None

    if mode == "or":
        nucleus = nuc1 | nuc2                      # type: ignore[operator]
    elif mode == "nuc1":
        nucleus = nuc1
    elif mode == "nuc2":
        nucleus = nuc2
    else:                                          # "and" (default)
        nucleus = nuc1 & nuc2                      # type: ignore[operator]
    del nuc1, nuc2
    out["nucleus"] = nucleus

    # ---- GAD1 / soma: denoised intensity, kept as uint16 --------------------
    def sub_gad(f, msg):
        if progress:
            progress(0.50 + 0.25 * f, msg)
    gad = _read_channel(reader, c_gad, n_z, H, W, sub_gad, "GAD1", should_abort)
    if gad is None:
        return None
    _norm_volume_inplace(gad)
    sub_gad(0.9, "denoising GAD1")
    _gaussian_inplace(gad, sigma3)
    out["soma_intensity"] = _to_u16(gad)
    del gad

    # ---- cilia: NOT denoised ------------------------------------------------
    def sub_cil(f, msg):
        if progress:
            progress(0.75 + 0.25 * f, msg)
    cil = _read_channel(reader, c_cil, n_z, H, W, sub_cil, "cilia", should_abort)
    if cil is None:
        return None
    # Don't Gaussian-denoise cilia — the blur erases structures one or two voxels
    # across, which is most of a cilium. Normalise against the 99.99th percentile
    # (NOT 99.5): cilia are so sparse that a 99.5 ceiling lands inside the noise
    # and saturates it to 1.0, drowning them. The 99.99th percentile sits at cilia
    # brightness, so cilia map near 1.0 while noise stays low, and it is still
    # robust to isolated hot/dead pixels. detect_cilia thresholds it downstream.
    _norm_volume_inplace(cil, lo_pct=1.0, hi_pct=99.99)
    out["cilia"] = _to_u16(cil)
    del cil

    if progress:
        progress(1.0, "done")
    held = sum(v.nbytes for v in out.values()) / (1024 ** 2)
    print(f"[analysis] prepared volumes held in memory: {held:.0f} MB "
          f"(nucleus bool + GAD1 uint16 + cilia uint16)")
    return out


def remove_small_nuclei(nucleus_mask, p):
    """Label nuclei in 3-D and drop components below the volume threshold."""
    labels, n = _label(nucleus_mask, _ones3())
    if n == 0:
        return labels
    sizes = np.bincount(labels.ravel())
    keep = np.where(sizes >= p.min_nucleus_volume_vox)[0]
    keep = keep[keep != 0]
    return np.where(np.isin(labels, keep), labels, 0)


def filter_nuclei_and_soma(nuc_labels, soma_intensity, p, progress=None):
    """
    Detect GAD1+ soma by a LOCAL BRIGHTNESS-RATIO test around each nucleus.

    For every nucleus, using distance from the nucleus surface:
      • inner zone  = voxels with 0 < dist <= soma_search_um   (cytoplasm)
      • outer shell = voxels with soma_search_um < dist <= soma_search_um
                                                        + soma_compare_um
    The nucleus is kept (counts as a cell body) only if

        median(GAD1 over inner)  >=  soma_ratio × median(GAD1 over outer)

    e.g. 6 µm inner, 3 µm comparison shell, 15 % differential → keep when the
    inner median exceeds 1.15× the 6–9 µm shell median.  (Median is used rather
    than mean so a few very bright voxels don't dominate the comparison.)

    Voxels belonging to a *different* nucleus are excluded from both zones when
    `soma_exclude_neighbours` is set (the default): in dense tissue the 6–9 µm
    shell routinely swallows a neighbouring cell, and without the exclusion a
    GAD1-negative cell packed next to a GAD1-positive one inherits its neighbour's
    signal in the inner zone — or, worse, a positive cell is masked because a
    bright neighbour lifts its comparison shell.

    Returns (kept_nucleus_labels, soma_mask), where soma_mask is the inner-zone
    voxels (of kept nuclei) brighter than their own outer-shell median — i.e. the
    actual bright cytoplasm, used downstream for cilia gating and the output.
    """
    from scipy import ndimage as ndi
    spacing = p.spacing()
    R_in  = float(p.soma_search_um)
    R_out = R_in + float(p.soma_compare_um)
    ratio = float(p.soma_ratio)
    eps   = 1e-6
    exclude_neighbours = bool(getattr(p, "soma_exclude_neighbours", True))

    # Keep the caller's dtype (uint16 from prepare_volumes) — casting to float32
    # here would allocate another full-size volume for no benefit, since the test
    # is a ratio of medians and therefore scale-invariant.
    gad = np.asarray(soma_intensity)
    ids = np.unique(nuc_labels)
    ids = ids[ids != 0]
    soma_mask = np.zeros(nuc_labels.shape, dtype=bool)
    if ids.size == 0:
        return nuc_labels, soma_mask

    pad  = [max(1, int(math.ceil(R_out / sp)) + 1) for sp in spacing]
    objs = ndi.find_objects(nuc_labels)
    step = max(1, len(ids) // 100)
    kept_nuc = np.zeros_like(nuc_labels)
    n_err = 0

    for n, lab in enumerate(ids):
        if progress and n % step == 0:
            progress(n / len(ids), f"nucleus {n + 1}/{len(ids)}")
        try:
            sl = objs[lab - 1]
            if sl is None:
                continue
            zs, ys, xs = sl[0], sl[1], sl[2]
            z0 = max(0, int(zs.start) - pad[0]); z1 = min(nuc_labels.shape[0], int(zs.stop) + pad[0])
            y0 = max(0, int(ys.start) - pad[1]); y1 = min(nuc_labels.shape[1], int(ys.stop) + pad[1])
            x0 = max(0, int(xs.start) - pad[2]); x1 = min(nuc_labels.shape[2], int(xs.stop) + pad[2])

            lab_crop = nuc_labels[z0:z1, y0:y1, x0:x1]
            nuc_crop = (lab_crop == lab)
            gad_crop = gad[z0:z1, y0:y1, x0:x1]
            dist  = _edt(~nuc_crop, spacing)
            inner = (dist > 0) & (dist <= R_in)
            outer = (dist > R_in) & (dist <= R_out)
            if exclude_neighbours:
                # Voxels inside a different nucleus are that cell's, not this
                # one's cytoplasm or background — drop them from both zones.
                foreign = (lab_crop != 0) & ~nuc_crop
                if foreign.any():
                    inner &= ~foreign
                    outer &= ~foreign
            if not inner.any() or not outer.any():
                continue

            inner_med = float(np.median(gad_crop[inner]))
            outer_med = float(np.median(gad_crop[outer]))
            if inner_med > eps and inner_med >= ratio * max(outer_med, eps):
                kept_nuc[z0:z1, y0:y1, x0:x1][nuc_crop] = lab
                bright = inner & (gad_crop > outer_med)
                soma_mask[z0:z1, y0:y1, x0:x1] |= bright
        except Exception as e:           # never let one nucleus abort the run
            n_err += 1
            if n_err <= 3:
                print(f"[analysis] soma test skipped nucleus {lab}: {e}")
    if n_err:
        print(f"[analysis] soma test: {n_err} nucleus/nuclei skipped due to errors")
    return kept_nuc, soma_mask




def group_cell_parts(nuc_labels, p, progress=None):
    """Build one cell record per surviving nucleus (centroid, physical position,
    volume). Cilia are detected, measured and matched separately — see
    `match_and_measure_cilia` — so this no longer touches the cilia channel."""
    from skimage.measure import regionprops

    spacing = np.asarray(p.spacing(), dtype=np.float64)
    cells = []
    n_labels = int(np.unique(nuc_labels[nuc_labels > 0]).size)
    for i, r in enumerate(regionprops(nuc_labels), start=1):
        cells.append({
            "num":      i,
            "label":    int(r.label),                     # value in nuc_labels
            "centroid": r.centroid,                       # (z, y, x) voxels
            "phys":     np.asarray(r.centroid) * spacing,  # microns
            "volume":   int(r.area),
            "best_len": 0.0,                              # matched cilium length (µm)
            "best_gap": None,                             # anchor gap to it (µm)
            "best_cilium": None,                          # PackedCrop of that cilium
            "cilia":    [],                               # cilia matched to this cell
        })
        if progress and (i % 256 == 0):
            # Progress against the label count, not len(cells) — that list is
            # what we are building, so dividing by it always read as ~100%.
            progress(min(0.99, i / max(1, n_labels)), f"cell {i}/{n_labels}")
    return cells


def _voxels_to_crop(voxels: np.ndarray):
    """Boolean bounding-box crop of a (N, 3) voxel list + its (z,y,x) slice tuple."""
    mn = voxels.min(axis=0)
    mx = voxels.max(axis=0)
    shape = tuple(int(mx[i] - mn[i] + 1) for i in range(3))
    crop = np.zeros(shape, dtype=bool)
    crop[voxels[:, 0] - mn[0], voxels[:, 1] - mn[1], voxels[:, 2] - mn[2]] = True
    sl = tuple(slice(int(mn[i]), int(mx[i]) + 1) for i in range(3))
    return crop, sl


def _nucleus_surface_by_cell(nuc_labels, spacing):
    """Physical coordinates of every nucleus boundary voxel, grouped by label.

    The boundary is the nucleus minus its 26-connected erosion, i.e. the voxels
    facing the cytoplasm. That surface — not the centre of mass — is where a
    basal body docks, so it is the right reference for "how far is this cilium
    from that cell".
    """
    from scipy import ndimage as ndi

    mask = nuc_labels > 0
    if not mask.any():
        return {}
    interior = ndi.binary_erosion(mask, structure=_ones3(), border_value=0)
    surf = mask & ~interior #type: ignore
    del interior
    coords = np.argwhere(surf)
    if coords.size == 0:                    # single-voxel nuclei erode to nothing
        coords = np.argwhere(mask)
    labs = nuc_labels[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(labs, kind="stable")
    coords = coords[order].astype(np.float64) * spacing
    labs = labs[order]
    uniq, starts = np.unique(labs, return_index=True)
    stops = np.r_[starts[1:], len(labs)]
    return {int(u): coords[a:b] for u, a, b in zip(uniq, starts, stops)}


def match_and_measure_cilia(cilia_vol, cells, p, nuc_labels=None, progress=None):
    """Detect cilia, assign each to a cell, and measure it.

    Detection is delegated to `cilia_finder.detect_cilia`, which groups the cilia
    brightness volume into connected components and drops those below the minimum
    physical volume.

    Assignment is LARGEST-FIRST and one-per-cell: the biggest cilium takes its
    closest free cell within `cilia_max_dist_um`, that cell is then removed from
    consideration, and so on.

    What "closest" means depends on `p.cilia_match_mode`:

      "proximal" (default)
          The shortest distance between any voxel of the cilium and any boundary
          voxel of the cell's nucleus. A primary cilium is anchored by its basal
          body at the plasma membrane and then projects away from the cell, so
          the meaningful quantity is that anchor gap. Centre-to-centre distance
          instead grows with the cilium's own length: a genuine 9 µm cilium
          docked right on the membrane has a centroid ~4.5 µm out, and a longer
          one is pushed past the 8 µm cut-off and lost precisely because it is
          long — biasing the surviving population short.

      "centroid" (legacy)
          Centre of mass to centre of mass, as before.

    Returns (list[CellResult] in cell order, detected-cilia mask). The mask is a
    boolean full-volume image of every detected cilium (matched or not), so the
    caller can show them all in the saved file for verification.
    """
    from scipy.spatial import KDTree
    from .cilia_finder import detect_cilia

    spacing = np.asarray(p.spacing(), dtype=np.float64)

    def _emit():
        lengths = sorted(c["best_len"] for c in cells if c["best_len"] > 0.0)
        n_with  = len(lengths)
        print(f"[analysis] cilia matched: {n_with}/{len(cells)} cells have a cilium")
        if lengths:
            med = lengths[len(lengths) // 2]
            print(f"[analysis] cilium length µm — min {lengths[0]:.2f}, "
                  f"median {med:.2f}, max {lengths[-1]:.2f}")
            # A neuronal primary cilium is typically 1–10 µm. Anything far past
            # that is usually two structures fused by the fill, or a cilium
            # running alongside a vessel/process that got absorbed with it.
            limit = float(getattr(p, "cilia_max_plausible_um", 30.0))
            implausible = [v for v in lengths if v > limit]
            if implausible:
                print(f"[analysis] WARNING: {len(implausible)} cilium/cilia measured "
                      f"longer than {limit:g} µm (max {implausible[-1]:.1f} µm). "
                      f"Neuronal primary cilia are typically 1–10 µm — these are "
                      f"likely merged structures. Raise the cilia bright threshold "
                      f"or lower the dim bridge to split them.")
        elif cells:
            print("[analysis] NOTE: no cilia matched to cells. If cells should have "
                  "cilia, raise the max cilia distance, lower the cilia bright "
                  "threshold, or confirm the Cilia channel mapping.")
        return [
            CellResult(
                num=c["num"],
                x=float(c["centroid"][2]), y=float(c["centroid"][1]),
                z=int(round(c["centroid"][0])),
                cilia_length_um=float(c["best_len"]),
                cilia_length_px=float(c["best_len"] / max(1e-6, p.um_per_px)),
                has_cilium=c["best_len"] > 0.0,
                nucleus_volume_vox=c["volume"],
                cilia_gap_um=c.get("best_gap"),
            )
            for c in cells
        ]

    # --- Detection: hand-written flood fill over the cilia brightness volume ----
    # Run this FIRST and unconditionally, so the detected cilia are always visible
    # in the saved file even when no cells were found — that keeps cilia detection
    # debuggable independently of the (separate) cell-detection stage.
    # `cilia_vol` arrives as uint16 (prepare_volumes keeps it quantised to halve
    # memory), so scale the [0,1] thresholds onto that range. A float volume is
    # still accepted for callers that build one themselves.
    scale = U16 if np.issubdtype(np.asarray(cilia_vol).dtype, np.integer) else 1.0
    dim = p.cilia_dim_threshold
    dim_threshold = (dim * scale) if 0.0 < dim < p.cilia_bright_threshold else None
    detections = detect_cilia(
        cilia_vol, p.cilia_min_volume_um3, spacing=tuple(spacing),
        bright_threshold=p.cilia_bright_threshold * scale,
        dim_threshold=dim_threshold,
        connectivity=p.cilia_connectivity,
        dim_bridge=getattr(p, "cilia_dim_bridge", 1),
        method=getattr(p, "cilia_method", "dilate"))
    K = len(detections)
    print(f"[analysis] cilia detected ({getattr(p, 'cilia_method', 'dilate')}, "
          f"bright≥{p.cilia_bright_threshold:g}, "
          f"≥{p.cilia_min_volume_um3:g} µm³): {K}", flush=True)

    # Paint every detected cilium into a mask for the saved-file visualisation.
    detected = np.zeros(cilia_vol.shape, dtype=bool)
    for det in detections:
        vx = det["voxels"]
        detected[vx[:, 0], vx[:, 1], vx[:, 2]] = True

    if not cells or K == 0:
        if not cells and K:
            print(f"[analysis] (no cells found — showing the {K} detected cilia anyway "
                  f"so detection can be checked)")
        return _emit(), detected

    coms      = np.asarray([det["centroid"] for det in detections], dtype=np.float64)
    com_phys  = coms * spacing
    comp_area = np.asarray([det["n_voxels"] for det in detections], dtype=np.int64)

    cell_phys = np.asarray([c["phys"] for c in cells], dtype=np.float64)
    centroid_tree = KDTree(cell_phys)

    mode = str(getattr(p, "cilia_match_mode", "proximal")).lower()
    reach = float(p.cilia_max_dist_um)

    # ---- proximal mode: per-cell nucleus surfaces in physical coordinates ----
    surf_by_label: dict = {}
    cell_reach = np.zeros(len(cells), dtype=np.float64)   # centroid → own surface
    if mode.startswith("prox"):
        if nuc_labels is None:
            print("[analysis] proximal matching needs the nucleus labels — "
                  "falling back to centroid matching")
            mode = "centroid"
        else:
            surf_by_label = _nucleus_surface_by_cell(nuc_labels, spacing)
            for i, c in enumerate(cells):
                pts = surf_by_label.get(int(c["label"]))
                if pts is not None and len(pts):
                    cell_reach[i] = float(
                        np.linalg.norm(pts - c["phys"], axis=1).max())
            if not surf_by_label:
                print("[analysis] no nucleus surface voxels — "
                      "falling back to centroid matching")
                mode = "centroid"
    max_cell_reach = float(cell_reach.max()) if cell_reach.size else 0.0

    # Assign cilia LARGEST-FIRST: each takes its closest free cell within the
    # radius. Larger cilia get first pick, and a cell holds at most one.
    # Size descending, ties broken by detection index so the greedy assignment
    # below is reproducible run to run.
    order = np.lexsort((np.arange(K), -comp_area))
    cell_taken = np.zeros(len(cells), dtype=bool)
    n_matched = 0
    step = max(1, K // 200)

    for idx, k in enumerate(order):
        if progress and idx % step == 0:
            progress(idx / max(1, K), f"cilium {idx + 1}/{K}")

        best_cell, best_gap = -1, np.inf

        if mode.startswith("prox"):
            vox_phys = detections[k]["voxels"].astype(np.float64) * spacing
            # How far this cilium reaches from its own centre of mass; a candidate
            # cell can be at most (reach + its own radius + this) away by centroid.
            cil_reach = float(np.linalg.norm(vox_phys - com_phys[k], axis=1).max())
            probe = reach + max_cell_reach + cil_reach
            cand = centroid_tree.query_ball_point(com_phys[k], probe)
            if cand:
                cil_tree = KDTree(vox_phys)
                for ci in cand:
                    if cell_taken[ci]:
                        continue
                    pts = surf_by_label.get(int(cells[ci]["label"]))
                    if pts is None or not len(pts):
                        continue
                    # Distance from every nucleus-surface voxel to the nearest
                    # voxel of this cilium; the minimum is the anchor gap.
                    gap = float(cil_tree.query(pts, k=1)[0].min())
                    if gap <= reach and gap < best_gap:
                        best_cell, best_gap = int(ci), gap
        else:
            cand = [c for c in centroid_tree.query_ball_point(com_phys[k], reach)
                    if not cell_taken[c]]
            if cand:
                d = np.linalg.norm(cell_phys[cand] - com_phys[k], axis=1)
                j = int(np.argmin(d))
                best_cell, best_gap = int(cand[j]), float(d[j])

        if best_cell < 0:
            continue
        cell_taken[best_cell] = True

        crop, sl = _voxels_to_crop(detections[k]["voxels"])
        length = _skeleton_geodesic_um(
            crop, spacing,
            isotropic=bool(getattr(p, "skeleton_isotropic", True)),
            max_voxels=int(getattr(p, "skeleton_max_voxels", 8_000_000)))
        cell = cells[best_cell]
        packed = PackedCrop.from_crop(crop, sl)
        com = tuple(float(v) for v in coms[k])          # cilium centroid (z,y,x vox)
        cell["cilia"].append({"crop": packed, "com": com, "length": length,
                              "gap_um": best_gap})
        cell["best_len"] = length                       # one cilium per cell
        cell["best_gap"] = best_gap
        cell["best_cilium"] = packed
        n_matched += 1

    label = ("proximal (cilium → nucleus surface)" if mode.startswith("prox")
             else "centroid → centroid")
    print(f"[analysis] cilia matched to a cell (largest-first, one per cell, "
          f"{label}): {n_matched}/{K} (reach {reach:g} µm)", flush=True)
    gaps = sorted(c["best_gap"] for c in cells if c.get("best_gap") is not None)
    if gaps:
        print(f"[analysis] anchor gap µm — min {gaps[0]:.2f}, "
              f"median {gaps[len(gaps) // 2]:.2f}, max {gaps[-1]:.2f}")
    return _emit(), detected


def _unique_names(names):
    """Make channel names unique, preserving order.

    Two output channels legitimately derive from the same source channel (the
    matched cilia and the all-detections verification channel), and both nucleus
    channels carry the same mask. Writing duplicate names leaves the viewer with
    two identically-labelled channels and no way to tell which is which, so
    repeats get a numeric suffix.
    """
    seen: dict = {}
    out = []
    for n in names:
        n = n or "Channel"
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 1
            out.append(n)
    return out


def _rgb255(colour):
    """(r, g, b) floats in 0..1 → ints in 0..255."""
    c = list(colour) + [0.0] * (3 - len(colour))
    return tuple(int(round(max(0.0, min(1.0, float(v))) * 255)) for v in c[:3])


def _rgba_int(colour) -> int:
    """(r,g,b[,a]) floats in 0..1 → signed-32-bit OME RGBA integer.

    OME packs Channel/@Color as R<<24 | G<<16 | B<<8 | A, stored signed, so
    anything with R >= 128 has to wrap into the negative range.
    """
    c = list(colour) + [1.0] * (4 - len(colour))
    r, g, b, a = (int(round(max(0.0, min(1.0, float(v))) * 255)) for v in c[:4])
    v = ((r & 255) << 24) | ((g & 255) << 16) | ((b & 255) << 8) | (a & 255)
    return v - (1 << 32) if v >= (1 << 31) else v


def save_processed_file(path, nuc_labels, soma_mask, cilia_mask, p,
                        channel_meta=None, cilia_detected_vol=None):
    """
    Save the processed masks as a multi-channel volume that mirrors the original
    file's channels (same names, colours and channel order) and metadata
    (pixel size from `p`).

    `channel_meta` is a list (one entry per output channel, already in the
    original channel order) of dicts:
        {"name", "colour"(r,g,b 0..1), "data_key" in {cilia, soma, nucleus, cellid}}
    Each channel is filled with the corresponding detection mask, so e.g. the
    original cilia channel now holds the (length-recorded) cilia, the soma
    channel holds the soma, and both nucleus channels hold the nucleus mask.

    Writing order: pylibCZIrw (Python 3.12 ok) for .czi, then czifile, else
    OME-TIFF — all annotated with the metadata.
    """
    nucleus = nuc_labels > 0
    # `cilia_mask` may be a boolean mask OR a pre-graded uint16 visualisation
    # (detected/measured/skeleton brightness levels). Use it as-is if integer.
    cilia_arr = np.asarray(cilia_mask)
    cilia_graded = cilia_arr if np.issubdtype(cilia_arr.dtype, np.integer) else None

    def _cellid_gray():
        maxlab = int(nuc_labels.max()) if nuc_labels.size else 0
        s = (65535.0 / maxlab) if maxlab > 0 else 0.0
        return (nuc_labels.astype(np.float32) * s).astype(np.uint16)

    masks = {
        "cilia":   cilia_arr.astype(bool) if cilia_graded is None else None,
        "soma":    np.asarray(soma_mask, dtype=bool),
        "nucleus": nucleus,
    }

    def channel_pixels(key):
        if key == "cellid":
            return _cellid_gray()
        if key == "cilia_detected":
            # Every cilium the flood-fill detector picked up, at full brightness.
            # Falls back to a plain nucleus-label gray if absent.
            if cilia_detected_vol is not None:
                return np.asarray(cilia_detected_vol, dtype=np.uint16)
            return _cellid_gray()
        if key == "cilia" and cilia_graded is not None:
            return cilia_graded.astype(np.uint16)
        m = masks.get(key, nucleus)
        return m.astype(np.uint16) * np.uint16(65535)

    if not channel_meta:
        channel_meta = [
            {"name": "Cilia",          "colour": (1.0, 0.0, 1.0), "data_key": "cilia"},
            {"name": "Soma",           "colour": (0.0, 1.0, 0.0), "data_key": "soma"},
            {"name": "Nucleus",        "colour": (0.0, 0.4, 1.0), "data_key": "nucleus"},
            {"name": "Cilia detected", "colour": (1.0, 0.0, 1.0),
             "data_key": "cilia_detected"},
        ]

    planes  = [channel_pixels(m.get("data_key", "nucleus")) for m in channel_meta]
    stack   = np.stack(planes, axis=0)   # (C, Z, Y, X)
    zcyx    = np.moveaxis(stack, 0, 1)   # (Z, C, Y, X)
    C, Z    = stack.shape[0], stack.shape[1]
    names   = _unique_names([str(m.get("name", f"Ch{i}"))
                             for i, m in enumerate(channel_meta)])
    colours = [m.get("colour", (1.0, 1.0, 1.0)) for m in channel_meta]
    um = float(p.um_per_px) if p.um_per_px else 1.0
    zs = float(p.z_um) if p.z_um and p.z_um > 0 else 0.0
    print("[analysis] output channels: " + ", ".join(
        f"{i}:{n} {_rgb255(c)}" for i, (n, c) in enumerate(zip(names, colours))))

    lower = str(path).lower()
    if lower.endswith(".czi"):
        # 1) pylibCZIrw — writes 2-D planes, then metadata.
        try:
            from pylibCZIrw import czi as pyczi
            with pyczi.create_czi(path, exist_ok=True) as doc:
                for c in range(C):
                    for z in range(Z):
                        plane = np.ascontiguousarray(stack[c, z])[..., np.newaxis]
                        doc.write(data=plane, plane={"C": c, "Z": z})
                try:
                    # Colours were previously never written to CZI at all — only
                    # names and scale went in, so every channel opened grey in
                    # ZEN. They belong in display_settings, as a tinting colour
                    # per channel.
                    display = None
                    try:
                        display = {
                            c: pyczi.ChannelDisplaySettingsDataClass(
                                True,
                                pyczi.TintingMode.Color,
                                pyczi.Rgb8Color(*_rgb255(colours[c])), #type: ignore
                                0.0, 1.0,
                            )
                            for c in range(C)
                        }
                    except Exception as e_disp:
                        print(f"[analysis] (CZI channel colours unavailable: {e_disp})")
                    doc.write_metadata(
                        channel_names={c: names[c] for c in range(C)},
                        scale_x=(um * 1e-6) or None,
                        scale_y=(um * 1e-6) or None,
                        scale_z=(zs * 1e-6) or None,
                        display_settings=display,
                    )
                except Exception as e_meta:
                    print(f"[analysis] (CZI metadata partial: {e_meta})")
            print(f"[analysis] Saved processed CZI (pylibCZIrw): {path}")
            return path
        except Exception as e_pyczi:
            print(f"[analysis] pylibCZIrw write unavailable ({e_pyczi}); trying czifile…")
            try:
                import czifile
                writer = getattr(czifile, "imwrite", None)
                if writer is None:
                    raise RuntimeError("installed czifile has no imwrite()")
                writer(path, zcyx)
                print(f"[analysis] Saved processed CZI (czifile): {path}")
                return path
            except Exception as e_czi:
                path = path[:-4] + ".ome.tif"
                print(f"[analysis] CZI writing unavailable ({e_czi}); "
                      f"writing OME-TIFF instead: {path}")

    import tifffile
    full = {"axes": "ZCYX",
            "PhysicalSizeX": um, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": um, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": names, "Color": [_rgba_int(c) for c in colours]}}
    if zs > 0:
        full["PhysicalSizeZ"] = zs
        full["PhysicalSizeZUnit"] = "µm"
    last: Exception = RuntimeError("write failed")
    for md in (full,
               {k: v for k, v in full.items() if k != "Channel"} |
               {"Channel": {"Name": names}},
               {"axes": "ZCYX"}):
        try:
            tifffile.imwrite(path, zcyx.astype(np.uint16), metadata=md)
            print(f"[analysis] Saved processed volume: {path}")
            return path
        except Exception as e:
            last = e
    raise last


def process_image(reader, mapping, n_z, um_per_px, params,
                  progress=None, should_abort=None, save_path=None,
                  channel_meta=None):
    """Run the full pipeline and return list[CellResult]."""
    p = params

    def aborted():
        return should_abort is not None and should_abort()

    total = 5 + (1 if save_path else 0)
    state = {"i": 0}

    def stage(name):
        state["i"] += 1
        idx = state["i"]

        def sp(frac=0.0, sub=""):
            pct = max(0, min(100, int(frac * 100)))
            label = f"{name} ({idx}/{total})"
            if sub:
                label += f" — {sub}"
            if progress:
                progress(pct, label)
        sp(0.0)
        return sp

    sp = stage("Reading & preparing volumes")
    model = prepare_volumes(reader, mapping, n_z, p, sp, should_abort)
    if model is None:
        return []
    sp(1.0)
    if aborted():
        return []

    sp = stage("Filtering nuclei by volume")
    nuclei = remove_small_nuclei(model["nucleus"], p)
    model["nucleus"] = None          # free the binary nucleus volume
    n_after_vol = int(np.unique(nuclei[nuclei > 0]).size)
    print(f"[analysis] nuclei after volume filter: {n_after_vol} "
          f"(min volume = {p.min_nucleus_volume_vox} vox)")
    sp(1.0)
    if aborted():
        return []

    sp = stage("Matching nuclei & soma")
    nuclei, soma = filter_nuclei_and_soma(nuclei, model["soma_intensity"], p, sp)
    model["soma_intensity"] = None   # free the GAD1 intensity volume
    n_after_soma = int(np.unique(nuclei[nuclei > 0]).size)
    soma_vox = int(soma.sum())
    soma_frac = 100.0 * soma_vox / max(1, soma.size)
    print(f"[analysis] nuclei kept by soma test: {n_after_soma} / {n_after_vol} "
          f"(inner {p.soma_search_um} µm vs +{p.soma_compare_um} µm shell, "
          f"ratio {p.soma_ratio:.2f}); soma voxels = {soma_vox} "
          f"({soma_frac:.0f}% of volume)")
    if n_after_soma == 0:
        print("[analysis] NOTE: no nuclei passed the soma brightness test — "
              "lower the differential %, widen the comparison shell, or check the "
              "GAD1 (Soma) channel mapping.")
    if soma_frac > 60.0:
        print("[analysis] WARNING: soma covers most of the volume — the soma test "
              "is likely too permissive. Consider a smaller soma radius or a higher "
              "differential %.")
    sp(1.0)
    if aborted():
        return []

    cilia = model["cilia"]           # uint16 brightness volume, [0,1] × 65535
    model["cilia"] = None
    cil_scale = U16 if np.issubdtype(cilia.dtype, np.integer) else 1.0
    cil_vox = int((cilia >= p.cilia_bright_threshold * cil_scale).sum())
    cil_frac = 100.0 * cil_vox / max(1, cilia.size)
    print(f"[analysis] cilia bright voxels (≥{p.cilia_bright_threshold:g}): {cil_vox} "
          f"({cil_frac:.1f}% of volume)", flush=True)
    if cil_frac > 5.0:
        print("[analysis] WARNING: cilia cover a large fraction of the volume — the "
              "cilia channel is likely over-detecting noise. Raise the cilia bright "
              "threshold.")

    print("[analysis] grouping nuclei into cells…", flush=True)
    sp = stage("Grouping cells")
    cells = group_cell_parts(nuclei, p, sp)
    print(f"[analysis] cells: {len(cells)}")
    sp(1.0)
    if aborted():
        return []

    print("[analysis] matching & measuring cilia…", flush=True)
    sp = stage("Matching & measuring cilia")
    results, detected = match_and_measure_cilia(cilia, cells, p,
                                                nuc_labels=nuclei, progress=sp)
    sp(1.0)

    # ---- Build a GRADED, fully visible cilia channel for verification --------
    # Three brightness levels so cilia can be checked against the numbers:
    #   • every detected cilium (matched or not) -> dim   (≈ 1/3 range)
    #   • the cilium matched to a cell           -> medium(≈ 2/3 range)
    #   • the skeleton used for its length        -> bright(full range)
    # Painting every detected cilium dim means the cilia always show up in the
    # saved file, and unmatched cilia stay visible for tuning the reach/volume.
    from skimage.morphology import skeletonize as _skel
    DIM, MED, BRIGHT = np.uint16(22000), np.uint16(45000), np.uint16(65535)
    cilia_vis = np.where(detected, DIM, np.uint16(0)).astype(np.uint16)

    def _paint(slices, mask, value):
        sub = cilia_vis[slices]
        np.maximum(sub, mask.astype(np.uint16) * value, out=sub)

    # Cilia channel grading: raise each matched cilium's body to medium and its
    # measured skeleton to full brightness on top of the dim "all detected" layer.
    n_drawn = 0
    for cell in cells:
        for item in cell["cilia"]:              # the cilium matched to this cell
            packed = item["crop"]
            if not packed.slices:
                continue
            body = packed.unpack()
            _paint(packed.slices, body, MED)     # matched cilium — medium bright
            try:
                _paint(packed.slices, _skel(body), BRIGHT)  # measured skeleton
            except Exception:
                pass
            n_drawn += 1

    # Extra verification channel: just what the flood-fill detector picked up —
    # every filtered cilium, at full brightness, nothing else.
    cilia_detected_vis = np.where(detected, BRIGHT, np.uint16(0)).astype(np.uint16)
    del cilia, detected
    print(f"[analysis] cilia drawn into processed image: {n_drawn} matched "
          f"(dim=all detected, medium=matched, bright=skeleton); extra detected-cilia "
          f"channel shows every flood-fill-detected cilium")

    if save_path:
        sp = stage("Saving processed file")
        try:
            save_processed_file(save_path, nuclei, soma, cilia_vis, p,
                                channel_meta=channel_meta,
                                cilia_detected_vol=cilia_detected_vis)
        except Exception as e:
            print(f"[analysis] Save failed: {e}")
        sp(1.0)

    n_cil = sum(1 for r in results if r.has_cilium)
    if progress:
        progress(100, f"Done ({total}/{total}) — {len(results)} cell(s), "
                      f"{n_cil} with a cilium")
    return results


# ===========================================================================
# Optional Qt helpers (used only when driving detection from the GUI)
# ===========================================================================

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
    from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter
    from PyQt6.QtWidgets import (
        QDialog, QGridLayout, QLabel, QComboBox, QDoubleSpinBox, QSpinBox,
        QDialogButtonBox, QVBoxLayout, QFrame, QCheckBox, QFileDialog,
        QScrollArea, QWidget, QListView, QApplication,
    )

    def _rgb_hex(colour) -> str:
        r, g, b = (int(round(max(0.0, min(1.0, float(c))) * 255)) for c in colour[:3])
        return f"#{r:02x}{g:02x}{b:02x}"

    def _default_processed_path(source_path: str) -> str:
        """`<folder of the source image>/<image name> - Processed.czi`."""
        if not source_path:
            return "Processed.czi"
        name = os.path.basename(source_path)
        for e in (".ome.tif", ".ome.tiff", ".czi", ".tif", ".tiff"):
            if name.lower().endswith(e):
                name = name[:-len(e)]
                break
        else:
            name = os.path.splitext(name)[0]
        return os.path.join(os.path.dirname(source_path), f"{name} - Processed.czi")


    def _fit_dialog_to_screen(dlg, parent=None, margin: int = 80):
        """Size a dialog to its content, but never past the usable screen.

        Hard-coded maximum heights are wrong on every display but the one they
        were picked on: too small and a dialog scrolls needlessly on a 27-inch
        monitor, too large and its buttons sit below the bottom of a laptop
        screen where they cannot be reached. Ask the layout what it wants, clamp
        that to the screen actually showing the window, and centre it on the
        parent.
        """
        screen = None
        if parent is not None and parent.window().windowHandle() is not None:
            screen = parent.window().windowHandle().screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        want = dlg.sizeHint()
        w = min(want.width(), max(360, avail.width() - margin))
        h = min(want.height(), max(320, avail.height() - margin))
        dlg.resize(w, h)
        dlg.setMaximumSize(avail.width(), avail.height())

        anchor = parent.window().frameGeometry() if parent is not None else avail
        x = anchor.center().x() - w // 2
        y = anchor.center().y() - h // 2
        # Keep the whole frame on screen, so the buttons are always reachable.
        x = max(avail.left(), min(x, avail.right() - w))
        y = max(avail.top(), min(y, avail.bottom() - h))
        dlg.move(x, y)

    def _colour_icon(colour, size: int = 12) -> QIcon:
        """A solid swatch of the channel's display colour, with a subtle border
        so a dark channel is still visible against the dark popup."""
        pm = QPixmap(size, size)
        pm.fill(QColor(_rgb_hex(colour)))
        painter = QPainter(pm)
        painter.setPen(QColor("#555555"))
        painter.drawRect(0, 0, size - 1, size - 1)
        painter.end()
        return QIcon(pm)

    class ChannelMapDialog(QDialog):
        """Assign channels to stains (shown in colour) + advanced settings."""

        def __init__(self, channels, um_per_px=1.0, z_um=0.0, parent=None,
                     source_path: str = ""):
            super().__init__(parent)
            self.setWindowTitle("Detect GAD1+ cells")
            self.setMinimumWidth(470)
            self._source_path = str(source_path or "")
            self._channels = channels
            self._um  = float(um_per_px) if um_per_px else 1.0
            self._zum = float(z_um) if z_um and z_um > 0 else 0.0
            self.mapping: dict = {}
            self.params = DetectionParams()
            self.save_path = None

            self.setStyleSheet("""
                QDialog { background: #161616; }
                QLabel  { color: #d0d0d0; font-family: 'Courier New'; font-size: 11px; }
                QComboBox, QDoubleSpinBox, QSpinBox {
                    background: #222; color: #e0e0e0;
                    border: 1px solid #444; border-radius: 3px; padding: 2px 4px;
                    font-family: 'Courier New';
                }
                QComboBox QAbstractItemView {
                    background: #1e1e1e; color: #e0e0e0;
                    border: 1px solid #444; outline: none;
                    selection-background-color: #003355; selection-color: #ffffff;
                }
                QCheckBox { color: #d0d0d0; font-family: 'Courier New'; font-size: 11px; }
            """)

            # The settings list outgrew a fixed dialog, so the body scrolls and
            # only the OK/Cancel row is pinned. The final size is chosen in
            # `_fit_to_screen` once the content exists.
            shell = QVBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            body = QWidget()
            outer = QVBoxLayout(body)
            scroll.setWidget(body)
            shell.addWidget(scroll, stretch=1)
            self._scroll = scroll
            self._body = body

            head = QLabel("Match each stain to its channel:")
            head.setStyleSheet("color: #00aaff;")
            outer.addWidget(head)

            grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
            self._combos = {}
            for row, stain in enumerate(STAINS):
                combo = QComboBox()
                # A real QListView for the popup: the native macOS combo popup
                # renders items its own way and ignores per-item roles.
                combo.setView(QListView())
                combo.setIconSize(QSize(12, 12))
                for i, ch in enumerate(channels):
                    label = ch.get("name") or f"Ch {ch.get('raw', i)}"
                    # The colour rides on an ICON, not on the text colour. Text
                    # colour was set through a per-combo stylesheet, and a
                    # `QComboBox { color: … }` rule applies to the widget AND its
                    # children — the popup view included — so every row in the
                    # list took the colour of whichever channel was selected.
                    # An icon is drawn from the item's own data and no stylesheet
                    # can cascade over it.
                    combo.addItem(_colour_icon(ch["colour"]), label)
                    combo.setItemData(i, QColor(_rgb_hex(ch["colour"])),
                                      Qt.ItemDataRole.ForegroundRole)
                if channels:
                    combo.setCurrentIndex(min(row, len(channels) - 1))
                grid.addWidget(QLabel(stain), row, 0)
                grid.addWidget(combo, row, 1)
                self._combos[stain] = combo
            outer.addLayout(grid)

            line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color: #333;"); outer.addWidget(line)

            z_scale = (f"{self._zum:.4f} µm/slice" if self._zum > 0
                       else "Z spacing not in metadata — enter below")
            opt = QLabel(f"Detection settings  (XY: {self._um:.4f} µm/px   Z: {z_scale}):")
            opt.setStyleSheet("color: #00aaff;"); opt.setWordWrap(True)
            outer.addWidget(opt)

            def umspin(val, lo=0.0, hi=1000.0, step=0.5):
                s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
                s.setValue(val); s.setSuffix(" µm"); return s

            def fracspin(val):
                s = QDoubleSpinBox(); s.setRange(0.0, 1.0); s.setSingleStep(0.01)
                s.setDecimals(2); s.setValue(val); return s

            def pctspin(val):
                s = QDoubleSpinBox(); s.setRange(0.0, 1000.0); s.setSingleStep(1.0)
                s.setDecimals(1); s.setValue(val); s.setSuffix(" %"); return s

            pgrid = QGridLayout()
            self._combine = QComboBox(); self._combine.addItems(["And", "Or", "Nuc1", "Nuc2"])
            self._combine.setCurrentText("Nuc2")
            self._floor    = fracspin(0.35)
            self._nuc_um   = umspin(4.0, 0.1)
            self._volfrac  = fracspin(1.0)
            self._somar_um = umspin(3.0, 0.1)
            self._cmp_um   = umspin(7.0, 0.1)
            self._somadiff = pctspin(30.0)
            self._reach_um = umspin(2.0, 0.1)
            self._cil_vol  = QDoubleSpinBox(); self._cil_vol.setRange(0.0, 1000.0)
            self._cil_vol.setSingleStep(0.1); self._cil_vol.setDecimals(2)
            self._cil_vol.setValue(0.5); self._cil_vol.setSuffix(" µm³")
            self._cil_bright = fracspin(0.5)
            self._cil_dim    = fracspin(0.2)
            self._cil_conn   = QComboBox()
            self._cil_conn.addItems(["26 (diagonal — thin/diagonal cilia)", "6 (faces only)"])
            self._cil_bridge = QSpinBox(); self._cil_bridge.setRange(0, 5)
            self._cil_bridge.setValue(3)
            self._cil_bridge.setToolTip(
                "Growth steps into dim signal from each side of a bright stretch. "
                "A gap of up to twice this many dim voxels is bridged; 0 disables "
                "dim absorption entirely.")
            self._cil_method = QComboBox()
            self._cil_method.addItems(["Deterministic (dilate + label)",
                                       "Legacy flood fill"])
            self._cil_method.setToolTip(
                "The legacy fill's result depends on voxel visit order: a dim "
                "voxel first reached along a dim path is fixed as non-expanding "
                "even if a bright voxel would have reached it next. Kept only to "
                "reproduce earlier runs.")
            self._cil_match = QComboBox()
            self._cil_match.addItems(["Proximal (cilium → nucleus surface)",
                                      "Legacy centroid → centroid"])
            self._cil_match.setToolTip(
                "Proximal measures the anchor gap, which is what a basal body "
                "actually constrains. Centroid distance grows with the cilium's "
                "own length, so it drops long cilia for being long.")
            self._skel_iso = QCheckBox("Resample cilia to cubic voxels before "
                                       "skeletonising")
            self._skel_iso.setChecked(False)
            self._skel_iso.setToolTip(
                "Makes the measured length independent of the cilium's angle to "
                "the optical axis, at the cost of a systematic overestimate from "
                "nearest-neighbour upsampling. On synthetic rods it roughly halves "
                "the angle-to-angle spread but reads ~10-20% long. Off by default.")
            self._soma_excl = QCheckBox("Exclude neighbouring nuclei from the soma "
                                        "test zones")
            self._soma_excl.setChecked(True)
            self._soma_excl.setToolTip(
                "In dense tissue the comparison shell routinely swallows the next "
                "cell along, which both leaks its signal into this cell's inner "
                "zone and lifts its outer shell.")
            self._zstep_um = QDoubleSpinBox(); self._zstep_um.setRange(0.01, 100.0)
            self._zstep_um.setSingleStep(0.1); self._zstep_um.setSuffix(" µm")
            self._zstep_um.setValue(self._zum if self._zum > 0 else 1.0)

            z_label = "Z-step / slice spacing" + ("  (from metadata)" if self._zum > 0 else "")
            rows = [
                ("Nucleus channels (And/Or/Nuc1/Nuc2)", self._combine),
                ("Intensity floor (0–1)",           self._floor),
                ("Minimum nucleus diameter",        self._nuc_um),
                ("Nucleus volume fraction (volfrac)", self._volfrac),
                ("Soma radius (inner)",             self._somar_um),
                ("Soma comparison shell width",     self._cmp_um),
                ("Soma brightness differential",    self._somadiff),
                ("Max cilia distance from cell",    self._reach_um),
                ("Minimum cilia volume",            self._cil_vol),
                ("Cilia bright threshold (0–1)",    self._cil_bright),
                ("Cilia dim threshold (0 = off)",   self._cil_dim),
                ("Cilia dim bridge (steps/side)",   self._cil_bridge),
                ("Cilia connectivity",              self._cil_conn),
                ("Cilia detection method",          self._cil_method),
                ("Cilium → cell matching",          self._cil_match),
                (z_label,                           self._zstep_um),
            ]
            for rr, (text, w) in enumerate(rows):
                pgrid.addWidget(QLabel(text), rr, 0); pgrid.addWidget(w, rr, 1)
            outer.addLayout(pgrid)

            outer.addWidget(self._skel_iso)
            outer.addWidget(self._soma_excl)

            self._save_chk = QCheckBox("Save processed volume after detection")
            outer.addWidget(self._save_chk)

            hint = QLabel("Soma test: a nucleus is GAD1+ only if the median soma-channel "
                          "brightness within the inner radius exceeds the brightness in "
                          "the comparison shell just beyond it by the differential %. "
                          "Raise floor / volfrac / differential to cut false positives.")
            hint.setWordWrap(True); hint.setStyleSheet("color: #777; font-size: 10px;")
            outer.addWidget(hint)

            outer.addStretch()
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                  QDialogButtonBox.StandardButton.Cancel)
            bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
            bb.setContentsMargins(10, 6, 10, 10)
            shell.addWidget(bb)

            _fit_dialog_to_screen(self, parent)

        def sizeHint(self):
            """Ask for the size the content actually wants, so `_fit_dialog_to_screen`
            has something meaningful to clamp."""
            body = self._body.sizeHint()
            return QSize(max(500, body.width() + 40), body.height() + 70)

        def _accept(self):
            self.mapping = {stain: self._channels[c.currentIndex()]["raw"]
                            for stain, c in self._combos.items()}
            self.params = DetectionParams.from_physical(
                self._um,
                min_nucleus_diam_um=self._nuc_um.value(),
                floor=self._floor.value(),
                volfrac=self._volfrac.value(),
                soma_search_um=self._somar_um.value(),
                soma_compare_um=self._cmp_um.value(),
                soma_diff_pct=self._somadiff.value(),
                cilia_max_dist_um=self._reach_um.value(),
                cilia_min_volume_um3=self._cil_vol.value(),
                cilia_bright_threshold=self._cil_bright.value(),
                cilia_dim_threshold=self._cil_dim.value(),
                cilia_connectivity=(6 if self._cil_conn.currentText().startswith("6") else 26),
                cilia_dim_bridge=self._cil_bridge.value(),
                cilia_method=("floodfill"
                              if self._cil_method.currentText().startswith("Legacy")
                              else "dilate"),
                cilia_match_mode=("centroid"
                                  if self._cil_match.currentText().startswith("Legacy")
                                  else "proximal"),
                skeleton_isotropic=self._skel_iso.isChecked(),
                soma_exclude_neighbours=self._soma_excl.isChecked(),
                z_um=self._zstep_um.value(),
                nucleus_combine=self._combine.currentText().lower(),
            )

            # One output channel per mapped stain, ordered by the original
            # channel index, keeping each source channel's display colour.
            #
            # The NAME says what the channel now holds, with the source channel
            # in brackets: "Cilia (AF647)", not bare "AF647". The saved file does
            # not contain AF647 fluorescence any more — it contains the cilia
            # mask derived from it — and naming it after the dye made the export
            # read as if it were raw data, with two channels sharing a name and
            # both nucleus channels carrying the same mask under different dye
            # names. `_unique_names` in the writer catches any remaining clash.
            stain_key = {"Cilia": "cilia", "GAD1 (Soma)": "soma",
                         "Nucleus 1": "nucleus", "Nucleus 2": "nucleus"}
            entries = []
            for stain in STAINS:
                idx = self._combos[stain].currentIndex()
                ch  = self._channels[idx] if 0 <= idx < len(self._channels) else {}
                src = ch.get("name") or ""
                entries.append({
                    "raw":      ch.get("raw", idx),
                    "name":     f"{stain} ({src})" if src else stain,
                    "colour":   ch.get("colour", (1.0, 1.0, 1.0)),
                    "data_key": stain_key[stain],
                })
            entries.sort(key=lambda e: e["raw"])
            # Extra verification channel (not one of the original stains): every
            # cilium the detector picked up, matched or not. Keeps the Cilia
            # channel's colour so the two read as a pair.
            cil = next((e for e in entries if e["data_key"] == "cilia"), None)
            cil_src = ""
            if cil:
                idx = self._combos["Cilia"].currentIndex()
                if 0 <= idx < len(self._channels):
                    cil_src = self._channels[idx].get("name") or ""
            entries.append({
                "raw":      max((e["raw"] for e in entries), default=0) + 1,
                "name":     (f"Cilia detected ({cil_src})" if cil_src
                             else "Cilia detected"),
                "colour":   cil["colour"] if cil else (1.0, 0.0, 1.0),
                "data_key": "cilia_detected",
            })
            self.channel_meta = entries

            if self._save_chk.isChecked():
                # Default to "<original name> - Processed.czi", in the folder the
                # original came from, so the result lands next to its source.
                default = _default_processed_path(self._source_path)
                path, chosen = QFileDialog.getSaveFileName(
                    self, "Save processed volume", default,
                    "CZI (*.czi);;OME-TIFF (*.ome.tif *.tif)")
                if path and not os.path.splitext(path)[1]:
                    path += ".ome.tif" if "OME" in (chosen or "") else ".czi"
                self.save_path = path or None
            self.accept()

    class DetectionWorker(QThread):
        """Runs process_image() off the GUI thread."""
        progress = pyqtSignal(int, str)
        finished = pyqtSignal(object)
        error    = pyqtSignal(str)

        def __init__(self, reader, mapping, n_z, um_per_px, params,
                     save_path=None, channel_meta=None):
            super().__init__()
            self._reader = reader
            self._mapping = mapping
            self._n_z = n_z
            self._um = um_per_px
            self._params = params
            self._save_path = save_path
            self._channel_meta = channel_meta
            self._abort = False

        def abort(self):
            self._abort = True

        def run(self):
            try:
                res = process_image(
                    self._reader, self._mapping, self._n_z, self._um, self._params,
                    progress=lambda pct, msg: self.progress.emit(pct, msg),
                    should_abort=lambda: self._abort,
                    save_path=self._save_path,
                    channel_meta=self._channel_meta,
                )
                self.finished.emit(res)
            except Exception:
                self.error.emit(traceback.format_exc())

except ImportError:
    ChannelMapDialog = None   # type: ignore
    DetectionWorker = None    # type: ignore