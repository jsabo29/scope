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

    # Soma detection = local brightness-ratio test around each nucleus
    soma_search_um:  float = 6.0     # inner radius (cytoplasm zone) around nucleus
    soma_compare_um: float = 3.0     # width of the comparison shell beyond it
    soma_ratio:      float = 1.15    # inner mean must exceed ratio × outer mean

    # Cilia (detected by the flood-fill detector in cilia_finder.detect_cilia)
    cilia_max_dist_um:      float = 8.0   # match a cilium to a cell if within this of it
    cilia_min_volume_um3:   float = 0.5   # drop cilia specks below this physical volume
    cilia_bright_threshold: float = 0.5   # flood-fill "bright" cut-off on [0,1]
    cilia_dim_threshold:    float = 0.2   # bridgeable-dim cut-off; <=0 disables bridging
    cilia_min_voxels:       int = 4       # derived from cilia_min_volume_um3
    connectivity:           int = 1       # 1 = faces, 3 = full 26-neighbour

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
        """
        r  = max(1e-6, float(um_per_px))
        zs = max(1e-6, float(z_um) if z_um and z_um > 0 else 1.0)
        voxvol_um3 = r * r * zs

        nuc_rad_um  = max(1e-6, min_nucleus_diam_um) / 2.0
        nuc_vol_um3 = (4.0 / 3.0) * math.pi * nuc_rad_um ** 3
        min_nuc_vox = max(8, int(float(volfrac) * nuc_vol_um3 / voxvol_um3))
        min_cil_vox = max(1, int(float(cilia_min_volume_um3) / voxvol_um3))

        return cls(
            um_per_px=r, z_um=zs,
            gaussian_sigma=1.0, min_threshold=float(floor),
            nucleus_combine=nucleus_combine,
            min_nucleus_volume_vox=min_nuc_vox,
            soma_search_um=float(soma_search_um),
            soma_compare_um=float(soma_compare_um),
            soma_ratio=1.0 + max(0.0, float(soma_diff_pct)) / 100.0,
            cilia_max_dist_um=float(cilia_max_dist_um),
            cilia_min_volume_um3=float(cilia_min_volume_um3),
            cilia_bright_threshold=float(cilia_bright_threshold),
            cilia_dim_threshold=float(cilia_dim_threshold),
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


def _boost_contrast(vol: np.ndarray, min_lum: float = 0.99,
                    contrast: float = 3.0) -> np.ndarray:
    """Hard contrast stretch of a normalised [0,1] volume — the same operation as
    the app's Process dialog with min-luminance and contrast pushed to their max.

    Used on the cilia channel INSTEAD of Gaussian denoising: the blur erases thin
    cilia, whereas maxing the black point (min_lum) and the gain about mid-grey
    (contrast) saturates the faint cilia so they separate cleanly from background.
    The leftover specks are removed later by the minimum-volume filter.
    """
    a = np.asarray(vol, dtype=np.float32)
    if min_lum > 0.0:
        a = np.clip((a - min_lum) / max(1e-6, 1.0 - min_lum), 0.0, 1.0)
    if contrast != 1.0:
        a = (a - 0.5) * contrast + 0.5
    return np.clip(a, 0.0, 1.0)


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


def _skeleton_geodesic_um(mask: np.ndarray, spacing) -> float:
    """Longest-path length (microns) through a 3-D region's skeleton."""
    from skimage.morphology import skeletonize

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
        return far, dist[far]

    a, _      = farthest(0)
    _, length = farthest(a)
    return float(length)


# ===========================================================================
# Pipeline stages  (progress(frac, sub): frac is 0..1 within the stage)
# ===========================================================================

def denoise(reader, mapping, n_z, p, progress=None, should_abort=None):
    """Read every channel/slice, normalise, and 3-D Gaussian-denoise."""
    from skimage.filters import gaussian

    c_cil = mapping["Cilia"]; c_n1 = mapping["Nucleus 1"]
    c_n2  = mapping["Nucleus 2"]; c_gad = mapping["GAD1 (Soma)"]
    order = [("cilia", c_cil), ("nuc1", c_n1), ("nuc2", c_n2), ("gad", c_gad)]

    if n_z <= 0:
        return None
    first = np.squeeze(reader._raw_slice(c_n1, 0))
    H, W  = int(first.shape[0]), int(first.shape[1])

    stacks = {k: np.empty((n_z, H, W), dtype=np.float32) for k, _ in order}
    for z in range(n_z):
        if should_abort and should_abort():
            return None
        if progress:
            progress(0.8 * (z + 1) / n_z, f"reading slice {z + 1}/{n_z}")
        for k, c in order:
            plane = np.asarray(reader._raw_slice(c, z), dtype=np.float32)
            stacks[k][z] = plane.reshape(H, W)

    z_sigma = max(0.4, p.gaussian_sigma * (p.um_per_px / max(1e-6, p.z_um)))
    sigma3  = (z_sigma, p.gaussian_sigma, p.gaussian_sigma)
    for ci, (k, _) in enumerate(order):
        if should_abort and should_abort():
            return None
        if progress:
            progress(0.8 + 0.2 * ci / len(order), f"denoising {k}")
        if k == "cilia":
            # Don't denoise cilia — the Gaussian blurs thin cilia away. Instead
            # apply a hard contrast stretch (contrast + min-luminance maxed) so the
            # faint cilia saturate and separate from background; the minimum-volume
            # filter downstream removes the leftover specks.
            smoothed = _boost_contrast(_norm_volume(stacks[k]))
        else:
            smoothed = gaussian(_norm_volume(stacks[k]), sigma=sigma3)  # type: ignore[arg-type]
        stacks[k] = np.asarray(smoothed, dtype=np.float32)
    if progress:
        progress(1.0, "done")
    return stacks


def build_3d_model(volumes, p):
    """Threshold the nuclei; keep the cilia and GAD1 (soma) channels as intensity.

    Cilia are detected later by the hand-written flood-fill detector
    (`cilia_finder.detect_cilia`) directly on the (boosted, normalised) cilia
    brightness volume, so that channel is carried through as intensity rather than
    thresholded here. The soma is likewise detected later by a local
    brightness-ratio test, so the GAD1 channel is also carried through.
    """
    nuc1 = _threshold_volume(volumes["nuc1"], p, min_size=p.cilia_min_voxels)
    nuc2 = _threshold_volume(volumes["nuc2"], p, min_size=p.cilia_min_voxels)
    mode = (p.nucleus_combine or "and").lower()
    if mode == "or":
        nucleus = nuc1 | nuc2
    elif mode == "nuc1":
        nucleus = nuc1
    elif mode == "nuc2":
        nucleus = nuc2
    else:                       # "and" (default)
        nucleus = nuc1 & nuc2
    return {"nucleus": nucleus,
            "cilia": np.asarray(volumes["cilia"], dtype=np.float32),
            "soma_intensity": np.asarray(volumes["gad"], dtype=np.float32)}


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

    gad = np.asarray(soma_intensity, dtype=np.float32)
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

            nuc_crop = (nuc_labels[z0:z1, y0:y1, x0:x1] == lab)
            gad_crop = gad[z0:z1, y0:y1, x0:x1]
            dist  = _edt(~nuc_crop, spacing)
            inner = (dist > 0) & (dist <= R_in)
            outer = (dist > R_in) & (dist <= R_out)
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
    for i, r in enumerate(regionprops(nuc_labels), start=1):
        cells.append({
            "num":      i,
            "label":    int(r.label),                     # value in nuc_labels
            "centroid": r.centroid,                       # (z, y, x) voxels
            "phys":     np.asarray(r.centroid) * spacing,  # microns
            "volume":   int(r.area),
            "best_len": 0.0,                              # longest matched cilium (µm)
            "best_cilium": None,                         # PackedCrop of that cilium
            "cilia":    [],                              # all cilia matched to this cell
        })
        if progress and (i % 256 == 0):
            progress(min(0.99, i / max(1, len(cells))), f"cell {i}")
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


def match_and_measure_cilia(cilia_vol, cells, p, progress=None):
    """Detect cilia with the hand-written flood-fill detector, then assign each to
    a cell and measure it.

    Detection is delegated to `cilia_finder.detect_cilia`, which flood-fills the
    cilia brightness volume into components and drops those below the minimum
    volume. Matching is LARGEST-FIRST and one-per-cell: the biggest cilium takes
    its closest cell within `cilia_max_dist_um`, that cell is then removed from
    consideration, and so on (a cell never has more than one cilium).

    Returns (list[CellResult] in cell order, detected-cilia mask). The mask is a
    boolean full-volume image of every detected cilium (matched or not), so the
    caller can show them all in the saved file for verification.
    """
    from scipy.spatial import KDTree
    from .cilia_finder import detect_cilia

    spacing = np.asarray(p.spacing(), dtype=np.float64)
    empty   = np.zeros(cilia_vol.shape, dtype=bool)

    def _emit():
        lengths = sorted(c["best_len"] for c in cells if c["best_len"] > 0.0)
        n_with  = len(lengths)
        print(f"[analysis] cilia matched: {n_with}/{len(cells)} cells have a cilium")
        if lengths:
            med = lengths[len(lengths) // 2]
            print(f"[analysis] cilium length µm — min {lengths[0]:.2f}, "
                  f"median {med:.2f}, max {lengths[-1]:.2f}")
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
            )
            for c in cells
        ]

    if not cells:
        return _emit(), empty

    # --- Detection: hand-written flood fill over the cilia brightness volume ----
    dim = p.cilia_dim_threshold
    dim_threshold = dim if 0.0 < dim < p.cilia_bright_threshold else None
    detections = detect_cilia(
        cilia_vol, p.cilia_min_volume_um3, spacing=tuple(spacing),
        bright_threshold=p.cilia_bright_threshold, dim_threshold=dim_threshold)
    K = len(detections)
    print(f"[analysis] cilia detected (flood fill, bright≥{p.cilia_bright_threshold:g}, "
          f"≥{p.cilia_min_volume_um3:g} µm³): {K}", flush=True)
    if K == 0:
        return _emit(), empty

    # Paint every detected cilium into a mask for the saved-file visualisation.
    detected = np.zeros(cilia_vol.shape, dtype=bool)
    for det in detections:
        vx = det["voxels"]
        detected[vx[:, 0], vx[:, 1], vx[:, 2]] = True

    coms      = np.asarray([det["centroid"] for det in detections], dtype=np.float64)
    com_phys  = coms * spacing
    comp_area = np.asarray([det["n_voxels"] for det in detections], dtype=np.int64)

    cell_phys = np.asarray([c["phys"] for c in cells], dtype=np.float64)
    tree = KDTree(cell_phys)

    # Assign cilia LARGEST-FIRST: each cilium takes its closest cell that is
    # within the radius AND not already claimed. Larger cilia get first pick.
    order = np.argsort(-comp_area, kind="stable")   # detection indices, largest first
    cell_taken = np.zeros(len(cells), dtype=bool)
    reach = float(p.cilia_max_dist_um)
    n_matched = 0
    step = max(1, K // 200)
    for idx, k in enumerate(order):
        if progress and idx % step == 0:
            progress(idx / max(1, K), f"cilium {idx + 1}/{K}")
        cand = tree.query_ball_point(com_phys[k], reach)   # cells within radius
        cand = [c for c in cand if not cell_taken[c]]
        if not cand:
            continue
        d = np.linalg.norm(cell_phys[cand] - com_phys[k], axis=1)
        best = int(cand[int(np.argmin(d))])                # closest free cell
        cell_taken[best] = True

        crop, sl = _voxels_to_crop(detections[k]["voxels"])
        length = _skeleton_geodesic_um(crop, spacing)
        cell = cells[best]
        packed = PackedCrop.from_crop(crop, sl)
        com = tuple(float(v) for v in coms[k])          # cilium centroid (z,y,x vox)
        cell["cilia"].append({"crop": packed, "com": com, "length": length})
        cell["best_len"] = length                       # one cilium per cell
        cell["best_cilium"] = packed
        n_matched += 1

    print(f"[analysis] cilia matched to a cell (largest-first, one per cell): "
          f"{n_matched}/{K} (reach {reach:g} µm)", flush=True)
    return _emit(), detected


def _rgba_int(colour) -> int:
    """(r,g,b[,a]) floats in 0..1 → signed-32-bit OME RGBA integer."""
    c = list(colour) + [1.0] * (4 - len(colour))
    r, g, b, a = (int(round(max(0.0, min(1.0, v)) * 255)) for v in c[:4])
    v = ((r & 255) << 24) | ((g & 255) << 16) | ((b & 255) << 8) | (a & 255)
    return v - (1 << 32) if v >= (1 << 31) else v


def save_processed_file(path, model, nuc_labels, soma_mask, cilia_mask, results, p,
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
            {"name": "Cilia",   "colour": (1.0, 0.0, 1.0), "data_key": "cilia"},
            {"name": "Soma",    "colour": (0.0, 1.0, 0.0), "data_key": "soma"},
            {"name": "Nucleus", "colour": (0.0, 0.4, 1.0), "data_key": "nucleus"},
            # Detected cilia — keep the Cilia channel's name and colour.
            {"name": "Cilia",   "colour": (1.0, 0.0, 1.0), "data_key": "cilia_detected"},
        ]

    planes  = [channel_pixels(m.get("data_key", "nucleus")) for m in channel_meta]
    stack   = np.stack(planes, axis=0)   # (C, Z, Y, X)
    zcyx    = np.moveaxis(stack, 0, 1)   # (Z, C, Y, X)
    C, Z    = stack.shape[0], stack.shape[1]
    names   = [str(m.get("name", f"Ch{i}")) for i, m in enumerate(channel_meta)]
    colours = [m.get("colour", (1.0, 1.0, 1.0)) for m in channel_meta]
    um = float(p.um_per_px) if p.um_per_px else 1.0
    zs = float(p.z_um) if p.z_um and p.z_um > 0 else 0.0

    lower = str(path).lower()
    if lower.endswith(".czi"):
        # 1) pylibCZIrw — writes 2-D planes, then metadata (names + scale).
        try:
            from pylibCZIrw import czi as pyczi
            with pyczi.create_czi(path, exist_ok=True) as doc:
                for c in range(C):
                    for z in range(Z):
                        plane = np.ascontiguousarray(stack[c, z])[..., np.newaxis]
                        doc.write(data=plane, plane={"C": c, "Z": z})
                try:
                    doc.write_metadata(
                        channel_names={c: names[c] for c in range(C)},
                        scale_x=(um * 1e-6) or None,
                        scale_y=(um * 1e-6) or None,
                        scale_z=(zs * 1e-6) or None,
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

    total = 6 + (1 if save_path else 0)
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

    sp = stage("Reading & denoising")
    vols = denoise(reader, mapping, n_z, p, sp, should_abort)
    if vols is None:
        return []
    sp(1.0)
    if aborted():
        return []

    sp = stage("Building 3-D model")
    model = build_3d_model(vols, p)
    del vols
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

    cilia = model["cilia"]                       # float brightness volume [0,1]
    model["cilia"] = None
    cil_vox = int((cilia >= p.cilia_bright_threshold).sum())
    cil_frac = 100.0 * cil_vox / max(1, cilia.size)
    print(f"[analysis] cilia bright voxels (≥{p.cilia_bright_threshold:g}): {cil_vox} "
          f"({cil_frac:.1f}% of volume)", flush=True)
    if cil_frac > 5.0:
        print("[analysis] WARNING: cilia cover a large fraction of the volume — the "
              "cilia channel is likely over-detecting noise. Raise the cilia bright "
              "threshold.")

    print("[analysis] stage 5: grouping nuclei into cells…", flush=True)
    sp = stage("Grouping cells")
    cells = group_cell_parts(nuclei, p, sp)
    print(f"[analysis] cells: {len(cells)}")
    sp(1.0)
    if aborted():
        return []

    print("[analysis] stage 6: matching & measuring cilia…", flush=True)
    sp = stage("Matching & measuring cilia")
    results, detected = match_and_measure_cilia(cilia, cells, p, sp)
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
            save_processed_file(save_path, model, nuclei, soma, cilia_vis,
                                results, p, channel_meta=channel_meta,
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
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QDialog, QGridLayout, QLabel, QComboBox, QDoubleSpinBox,
        QDialogButtonBox, QVBoxLayout, QFrame, QCheckBox, QFileDialog,
    )

    def _rgb_hex(colour) -> str:
        r, g, b = (int(round(max(0.0, min(1.0, float(c))) * 255)) for c in colour[:3])
        return f"#{r:02x}{g:02x}{b:02x}"

    class ChannelMapDialog(QDialog):
        """Assign channels to stains (shown in colour) + advanced settings."""

        def __init__(self, channels, um_per_px=1.0, z_um=0.0, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Detect GAD1+ cells")
            self.setMinimumWidth(470)
            self._channels = channels
            self._um  = float(um_per_px) if um_per_px else 1.0
            self._zum = float(z_um) if z_um and z_um > 0 else 0.0
            self.mapping: dict = {}
            self.params = DetectionParams()
            self.save_path = None

            self.setStyleSheet("""
                QDialog { background: #161616; }
                QLabel  { color: #d0d0d0; font-family: 'Courier New'; font-size: 11px; }
                QComboBox, QDoubleSpinBox {
                    background: #222; color: #e0e0e0;
                    border: 1px solid #444; border-radius: 3px; padding: 2px 4px;
                    font-family: 'Courier New';
                }
                QCheckBox { color: #d0d0d0; font-family: 'Courier New'; font-size: 11px; }
            """)

            outer = QVBoxLayout(self)
            head = QLabel("Match each stain to its channel:")
            head.setStyleSheet("color: #00aaff;")
            outer.addWidget(head)

            grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
            self._combos = {}
            for row, stain in enumerate(STAINS):
                combo = QComboBox()
                for i, ch in enumerate(channels):
                    combo.addItem(ch.get("name", f"Ch {ch.get('raw', i)}"))
                    combo.setItemData(i, QColor(_rgb_hex(ch["colour"])),
                                      Qt.ItemDataRole.ForegroundRole)
                if channels:
                    combo.setCurrentIndex(min(row, len(channels) - 1))
                combo.currentIndexChanged.connect(lambda _i, c=combo: self._recolour(c))
                self._recolour(combo)
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
            self._combine.setCurrentText("And")
            self._floor    = fracspin(0.12)
            self._nuc_um   = umspin(5.0, 0.1)
            self._volfrac  = fracspin(1.0)
            self._somar_um = umspin(6.0, 0.1)
            self._cmp_um   = umspin(3.0, 0.1)
            self._somadiff = pctspin(15.0)
            self._reach_um = umspin(8.0, 0.1)
            self._cil_vol  = QDoubleSpinBox(); self._cil_vol.setRange(0.0, 1000.0)
            self._cil_vol.setSingleStep(0.1); self._cil_vol.setDecimals(2)
            self._cil_vol.setValue(0.5); self._cil_vol.setSuffix(" µm³")
            self._cil_bright = fracspin(0.5)
            self._cil_dim    = fracspin(0.2)
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
                (z_label,                           self._zstep_um),
            ]
            for rr, (text, w) in enumerate(rows):
                pgrid.addWidget(QLabel(text), rr, 0); pgrid.addWidget(w, rr, 1)
            outer.addLayout(pgrid)

            self._save_chk = QCheckBox("Save processed volume after detection")
            outer.addWidget(self._save_chk)

            hint = QLabel("Soma test: a nucleus is GAD1+ only if the median soma-channel "
                          "brightness within the inner radius exceeds the brightness in "
                          "the comparison shell just beyond it by the differential %. "
                          "Raise floor / volfrac / differential to cut false positives.")
            hint.setWordWrap(True); hint.setStyleSheet("color: #777; font-size: 10px;")
            outer.addWidget(hint)

            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                  QDialogButtonBox.StandardButton.Cancel)
            bb.accepted.connect(self._accept); bb.rejected.connect(self.reject)
            outer.addWidget(bb)

        def _recolour(self, combo):
            i = combo.currentIndex()
            if 0 <= i < len(self._channels):
                combo.setStyleSheet(
                    combo.styleSheet() +
                    f"\nQComboBox {{ color: {_rgb_hex(self._channels[i]['colour'])}; }}")

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
                z_um=self._zstep_um.value(),
                nucleus_combine=self._combine.currentText().lower(),
            )

            # One output channel per mapped stain, ordered by the original
            # channel index, so the saved file mirrors the original channels
            # (name, colour, order). Each carries that stain's detection mask.
            stain_key = {"Cilia": "cilia", "GAD1 (Soma)": "soma",
                         "Nucleus 1": "nucleus", "Nucleus 2": "nucleus"}
            entries = []
            for stain in STAINS:
                idx = self._combos[stain].currentIndex()
                ch  = self._channels[idx] if 0 <= idx < len(self._channels) else {}
                entries.append({
                    "raw":      ch.get("raw", idx),
                    "name":     ch.get("name") or stain,
                    "colour":   ch.get("colour", (1.0, 1.0, 1.0)),
                    "data_key": stain_key[stain],
                })
            entries.sort(key=lambda e: e["raw"])
            # Extra verification channel (not one of the original stains): every
            # cilium the flood-fill detector picked up. Appended after the real
            # channels, keeping the original Cilia channel's name and colour.
            cil = next((e for e in entries if e["data_key"] == "cilia"), None)
            entries.append({
                "raw":      max((e["raw"] for e in entries), default=0) + 1,
                "name":     cil["name"] if cil else "Cilia",
                "colour":   cil["colour"] if cil else (1.0, 0.0, 1.0),
                "data_key": "cilia_detected",
            })
            self.channel_meta = entries

            if self._save_chk.isChecked():
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save processed volume", "processed.ome.tif",
                    "OME-TIFF (*.ome.tif *.tif);;CZI (*.czi)")
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