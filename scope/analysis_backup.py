"""
analysis.py — locate GAD1+ ciliated neurons and measure their cilia.

This module is deliberately decoupled from the viewer (scope.py / app.py).  It
needs only:
  • a `reader` object exposing  `_raw_slice(channel: int, z: int) -> 2-D ndarray`
    (the app's SliceReader satisfies this), and
  • the absolute channel indices for each of the four stains.

Pipeline (see detect_cells):
  1.  Per Z slice, threshold the Nucleus 1 / Nucleus 2 / GAD1 channels.
  2.  A *cell* footprint on that slice = a nucleus region (Nucleus 1 combined
      with Nucleus 2) whose area lies sufficiently inside a GAD1/soma region.
  3.  All per-slice footprints are stacked into a 3-D volume and labelled with
      3-D connected components — so a cell that appears across many Z slices is
      ONE component, counted once.  Its 3-D centroid gives the (x, y, z) centre.
  4.  For each cell, a local max-projection of the cilia mask over the cell's
      Z-range is searched for a cilium reaching the soma; the cilium is
      skeletonised and its geodesic (longest-path) length measured in pixels,
      then converted to micrometres.

Nothing here imports Qt except the optional ChannelMapDialog / DetectionWorker
helpers at the bottom, which are only used when driving this from the GUI.

NOTE: thresholds and radii are heuristics.  They are exposed in DetectionParams
(and in the dialog) precisely because they must be tuned on real data.
"""

from __future__ import annotations

import heapq
import math
import traceback
from dataclasses import dataclass, field, asdict

import numpy as np

# ---------------------------------------------------------------------------
# The four stains the user maps channels to.
# ---------------------------------------------------------------------------
STAINS = ["Cilia", "Nucleus 1", "Nucleus 2", "GAD1 (Soma)"]


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

@dataclass
class DetectionParams:
    # Pre-processing / thresholding
    gaussian_sigma:   float = 1.0    # denoise before thresholding
    min_threshold:    float = 0.12   # absolute floor (on [0,1]) so blank slices
                                     # don't threshold pure noise
    min_object_px:    int   = 25     # drop tiny specks from each channel mask

    # Nucleus → cell logic
    nucleus_combine:  str   = "and"  # "and" = need both nuclei; "or" = either
    min_nucleus_area_px: int = 80    # ignore nucleus blobs smaller than this
    gad_overlap_frac: float = 0.6    # fraction of a nucleus that must fall
                                     # inside GAD1 to count as a soma cell

    # 3-D cell consolidation
    min_cell_volume_vox: int = 150   # drop 3-D components smaller than this
    z_connectivity:   int   = 1      # 1 = faces only, 3 = full 26-neighbour

    # Cilia search / measurement
    cilia_z_window:   int   = 2      # ± slices around the cell to project cilia
    cilia_search_radius_px: int = 40 # how far from the soma a cilium may reach
    cilia_min_area_px: int  = 8      # ignore cilia specks smaller than this

    # ------------------------------------------------------------------
    @classmethod
    def from_physical(cls, um_per_px: float, *,
                      nucleus_diam_um: float = 8.0,
                      min_cilium_um:   float = 1.0,
                      cilia_reach_um:  float = 8.0,
                      z_step_um:       float = 1.0,
                      nucleus_combine: str   = "and",
                      strictness:      str   = "balanced") -> "DetectionParams":
        """
        Build parameters from physical sizes you actually know, converting to
        pixels with the image's `um_per_px`.  This is the recommended way to set
        things up — pixel thresholds are resolution-dependent, micron sizes are
        not.

        Args:
            um_per_px        : microns per pixel (the file reports this).
            nucleus_diam_um  : typical nucleus diameter in microns.
            min_cilium_um    : shortest cilium length you'd accept as real.
            cilia_reach_um   : how far a cilium may extend from the soma.
            z_step_um        : axial spacing between Z slices, in microns.
            nucleus_combine  : "and" (both nuclei) or "or".
            strictness       : "lenient" | "balanced" | "strict".  Higher
                               strictness raises the noise floor, the
                               nucleus-size minimum, and the GAD-overlap
                               requirement — the three biggest false-positive
                               levers.
        """
        r = max(1e-6, float(um_per_px))
        presets = {
            "lenient":  dict(floor=0.08, area_frac=0.20, overlap=0.45),
            "balanced": dict(floor=0.12, area_frac=0.35, overlap=0.60),
            "strict":   dict(floor=0.18, area_frac=0.50, overlap=0.75),
        }
        s = presets.get(strictness, presets["balanced"])

        nuc_area_px = math.pi * (nucleus_diam_um / 2.0 / r) ** 2  # expected area
        min_nuc = max(10, int(nuc_area_px * s["area_frac"]))

        # A real cell appears with real area across several Z slices.
        z_span  = max(1.0, nucleus_diam_um / max(1e-6, z_step_um))
        min_vol = max(min_nuc, int(min_nuc * z_span * 0.5))

        search_px    = max(5, int(round(cilia_reach_um / r)))
        min_cil_area = max(4, int(round((min_cilium_um / r) * 2.0)))  # ~2 px wide
        min_obj      = max(8, int(nuc_area_px * 0.02))

        return cls(
            gaussian_sigma=1.0,
            min_threshold=s["floor"],
            min_object_px=min_obj,
            nucleus_combine=nucleus_combine,
            min_nucleus_area_px=min_nuc,
            gad_overlap_frac=s["overlap"],
            min_cell_volume_vox=min_vol,
            z_connectivity=1,
            cilia_z_window=2,
            cilia_search_radius_px=search_px,
            cilia_min_area_px=min_cil_area,
        )


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


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _norm(plane: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Percentile-stretch a raw plane to float32 in [0, 1] (matches the viewer)."""
    a  = plane.astype(np.float32)
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0.0, 1.0)


def _mask(plane_norm: np.ndarray, p: DetectionParams) -> np.ndarray:
    """Threshold a normalised plane into a cleaned boolean mask."""
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import remove_small_objects

    g = gaussian(plane_norm, sigma=p.gaussian_sigma)
    try:
        t = float(threshold_otsu(g))
    except Exception:
        t = float(np.percentile(g, 99.0))
    t = max(t, p.min_threshold)          # never threshold below the noise floor
    m = g > t
    if p.min_object_px > 1:
        m = remove_small_objects(m, min_size=p.min_object_px)
    return m


def _skeleton_geodesic_px(region_mask: np.ndarray) -> float:
    """
    Length of the longest path through a region's skeleton, in pixels.

    Skeletonise the (cropped) region, treat skeleton pixels as graph nodes with
    edges to 8-neighbours (weight 1 orthogonally, √2 diagonally), then take the
    graph "diameter" (two Dijkstra passes).  This measures a curved cilium's
    true length and ignores short side-spurs.
    """
    from skimage.morphology import skeletonize

    sk = skeletonize(region_mask)
    coords = list(zip(*np.nonzero(sk)))
    if len(coords) <= 1:
        return float(len(coords))

    index = {c: i for i, c in enumerate(coords)}
    SQRT2 = math.sqrt(2.0)
    steps = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
             (0, -1, 1.0),                  (0, 1, 1.0),
             (1, -1, SQRT2),  (1, 0, 1.0),  (1, 1, SQRT2)]
    adj: list[list[tuple[int, float]]] = [[] for _ in coords]
    for c, i in index.items():
        y, x = c
        for dy, dx, w in steps:
            j = index.get((y + dy, x + dx))
            if j is not None:
                adj[i].append((j, w))

    def farthest(src: int) -> tuple[int, float]:
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


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_cells(reader, mapping: dict, n_z: int, um_per_px: float,
                 params: DetectionParams | None = None,
                 progress=None, should_abort=None) -> list[CellResult]:
    """
    Locate GAD1+ cells whose nuclei are present, dedup across Z, and measure
    each cell's cilium.

    Args:
        reader     : object with _raw_slice(channel, z) -> 2-D ndarray.
        mapping    : {"Cilia": c, "Nucleus 1": c, "Nucleus 2": c, "GAD1 (Soma)": c}
                     where each value is an ABSOLUTE channel index.
        n_z        : number of Z slices.
        um_per_px  : microns per pixel (for length conversion).
        params     : DetectionParams (defaults used if None).
        progress   : optional callable(pct:int, message:str).
        should_abort: optional callable() -> bool to cancel early.

    Returns:
        list[CellResult], one per detected cell (deduplicated across Z).
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, disk  # type: ignore[attr-defined]

    p = params or DetectionParams()

    def _emit(pct, msg):
        if progress is not None:
            progress(int(pct), msg)

    def _aborted():
        return should_abort is not None and should_abort()

    c_cil = mapping["Cilia"]
    c_n1  = mapping["Nucleus 1"]
    c_n2  = mapping["Nucleus 2"]
    c_gad = mapping["GAD1 (Soma)"]

    # ---- Pass 1: build the 3-D cell-footprint volume + cilia mask volume ----
    if n_z <= 0:
        return []

    # Probe one slice to learn the plane shape, then allocate the volumes.
    first = np.squeeze(reader._raw_slice(c_n1, 0))
    H, W = int(first.shape[0]), int(first.shape[1])
    cell_vol  = np.zeros((n_z, H, W), dtype=bool)   # nucleus∩soma footprints
    cilia_vol = np.zeros((n_z, H, W), dtype=bool)   # cilia masks

    for z in range(n_z):
        if _aborted():
            return []
        _emit(5 + int(55 * z / max(1, n_z)), f"Segmenting slice {z + 1}/{n_z}…")

        n1 = _mask(_norm(reader._raw_slice(c_n1, z)), p)
        n2 = _mask(_norm(reader._raw_slice(c_n2, z)), p)
        gd = _mask(_norm(reader._raw_slice(c_gad, z)), p)
        cl = _mask(_norm(reader._raw_slice(c_cil, z)), p)

        cilia_vol[z] = cl

        nuc = (n1 & n2) if p.nucleus_combine == "and" else (n1 | n2)

        # Keep only nucleus regions that sit inside a GAD1 soma.
        nuc_lbl = label(nuc)
        slice_cells = np.zeros_like(nuc)
        for r in regionprops(nuc_lbl):
            if r.area < p.min_nucleus_area_px:
                continue
            ys = r.coords[:, 0]
            xs = r.coords[:, 1]
            inside = int(gd[ys, xs].sum())
            if inside / float(r.area) >= p.gad_overlap_frac:
                slice_cells[ys, xs] = True
        cell_vol[z] = slice_cells

    # ---- Consolidate across Z: 3-D connected components = unique cells ------
    if _aborted():
        return []
    _emit(65, "Consolidating cells across Z…")

    conn     = max(1, min(3, p.z_connectivity))
    labels3d = label(cell_vol, connectivity=conn)   # nD-aware; returns an array
    n_found  = int(labels3d.max()) # type: ignore

    if n_found == 0:
        _emit(100, "No GAD1+ nucleated cells found.")
        return []

    cell_regions = [r for r in regionprops(labels3d)
                    if r.area >= p.min_cell_volume_vox]

    # ---- Pass 2: per cell, find + measure the attached cilium ---------------
    results: list[CellResult] = []
    Zmax, Ymax, Xmax = cell_vol.shape
    R = int(p.cilia_search_radius_px)
    se = disk(R)

    for k, r in enumerate(cell_regions, start=1):
        if _aborted():
            return []
        _emit(65 + int(34 * k / max(1, len(cell_regions))),
              f"Measuring cilia {k}/{len(cell_regions)}…")

        cz, cy, cx = r.centroid                      # 3-D centroid (z, y, x)
        z0, y0, x0, z1, y1, x1 = r.bbox              # (zmin,ymin,xmin,zmax,ymax,xmax)

        # 2-D footprint of this cell (max-projected over its slices).
        coords = r.coords                            # (N, 3) z,y,x
        foot = np.zeros((Ymax, Xmax), dtype=bool)
        foot[coords[:, 1], coords[:, 2]] = True

        # Crop window around the footprint, expanded by the search radius.
        cy0 = max(0, y0 - R); cy1 = min(Ymax, y1 + R)
        cx0 = max(0, x0 - R); cx1 = min(Xmax, x1 + R)

        foot_crop = foot[cy0:cy1, cx0:cx1]
        attach_zone = binary_dilation(foot_crop, se)  # soma reach for cilia

        # Local max-projection of the cilia mask over the cell's Z-range.
        zlo = max(0, z0 - p.cilia_z_window)
        zhi = min(Zmax, z1 + p.cilia_z_window)
        cil_crop = cilia_vol[zlo:zhi, cy0:cy1, cx0:cx1].any(axis=0)
        cil_crop = cil_crop & ~foot_crop              # don't count the soma itself

        best_len_px = 0.0
        if cil_crop.any():
            cil_lbl = label(cil_crop)
            for cr in regionprops(cil_lbl):
                if cr.area < p.cilia_min_area_px:
                    continue
                region_mask = cil_lbl == cr.label
                if (region_mask & attach_zone).any():    # reaches the soma?
                    L = _skeleton_geodesic_px(region_mask)
                    if L > best_len_px:
                        best_len_px = L

        has_cil = best_len_px > 0.0
        results.append(CellResult(
            num=k,
            x=float(cx), y=float(cy), z=int(round(cz)),
            cilia_length_px=float(best_len_px),
            cilia_length_um=float(best_len_px * um_per_px),
            has_cilium=has_cil,
            nucleus_volume_vox=int(r.area),
        ))

    _emit(100, f"Done — {len(results)} cell(s) found.")
    return results


# ---------------------------------------------------------------------------
# Optional Qt helpers (used only when driving detection from the GUI)
# ---------------------------------------------------------------------------

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QDialog, QGridLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
        QDialogButtonBox, QVBoxLayout, QFrame, QHBoxLayout,
    )

    def _rgb_hex(colour) -> str:
        r, g, b = (int(round(max(0.0, min(1.0, float(c))) * 255)) for c in colour[:3])
        return f"#{r:02x}{g:02x}{b:02x}"

    class ChannelMapDialog(QDialog):
        """
        Lets the user assign one channel to each stain (channels are shown in
        their display colour) and tweak the key detection parameters.

        `channels` is the app's channel list: [{"raw", "name", "colour"}, ...].
        On accept, `.mapping` is {stain: raw_index} and `.params` a DetectionParams.
        """

        def __init__(self, channels: list, um_per_px: float = 1.0,
                     z_um: float = 0.0, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Detect ciliated GAD1+ cells")
            self.setMinimumWidth(460)
            self._channels = channels
            self._um  = float(um_per_px) if um_per_px else 1.0
            self._zum = float(z_um) if z_um and z_um > 0 else 0.0
            self.mapping: dict = {}
            self.params: DetectionParams = DetectionParams()

            self.setStyleSheet("""
                QDialog { background: #161616; }
                QLabel  { color: #d0d0d0; font-family: 'Courier New'; font-size: 11px; }
                QComboBox, QSpinBox, QDoubleSpinBox {
                    background: #222; color: #e0e0e0;
                    border: 1px solid #444; border-radius: 3px; padding: 2px 4px;
                    font-family: 'Courier New';
                }
            """)

            outer = QVBoxLayout(self)
            head = QLabel("Match each stain to its channel:")
            head.setStyleSheet("color: #00aaff;")
            outer.addWidget(head)

            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(6)
            self._combos: dict = {}
            for row, stain in enumerate(STAINS):
                lbl = QLabel(stain)
                combo = QComboBox()
                for i, ch in enumerate(channels):
                    combo.addItem(ch.get("name", f"Ch {ch.get('raw', i)}"))
                    combo.setItemData(i, QColor(_rgb_hex(ch["colour"])),
                                      Qt.ItemDataRole.ForegroundRole)
                # Sensible default: stain i -> channel i (user can change).
                if channels:
                    combo.setCurrentIndex(min(row, len(channels) - 1))
                combo.currentIndexChanged.connect(
                    lambda _i, c=combo: self._recolour(c))
                self._recolour(combo)
                grid.addWidget(lbl, row, 0)
                grid.addWidget(combo, row, 1)
                self._combos[stain] = combo
            outer.addLayout(grid)

            line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet("color: #333;")
            outer.addWidget(line)

            z_scale = (f"{self._zum:.4f} µm/slice" if self._zum > 0
                       else "Z spacing not in metadata — enter it below")
            opt = QLabel(f"Detection settings  (XY: {self._um:.4f} µm/px   "
                         f"Z: {z_scale}):")
            opt.setStyleSheet("color: #00aaff;")
            opt.setWordWrap(True)
            outer.addWidget(opt)

            pgrid = QGridLayout()
            self._strict = QComboBox()
            self._strict.addItems(["Lenient", "Balanced", "Strict"])
            self._strict.setCurrentText("Balanced")
            self._strict.setToolTip(
                "Higher strictness raises the noise floor, the nucleus-size "
                "minimum, and the nucleus-inside-GAD1 requirement — the main "
                "false-positive controls.")

            self._combine = QComboBox(); self._combine.addItems(["and", "or"])
            self._combine.setCurrentText("and")

            self._nuc_um = QDoubleSpinBox(); self._nuc_um.setRange(0.1, 1000.0)
            self._nuc_um.setSingleStep(0.5); self._nuc_um.setValue(8.0)
            self._nuc_um.setSuffix(" µm")
            self._zstep_um = QDoubleSpinBox(); self._zstep_um.setRange(0.01, 100.0)
            self._zstep_um.setSingleStep(0.1)
            self._zstep_um.setSuffix(" µm")
            # Prefill axial spacing from metadata when available.
            self._zstep_um.setValue(self._zum if self._zum > 0 else 1.0)
            if self._zum > 0:
                self._zstep_um.setToolTip("Read from file metadata; override only "
                                          "if you know it's wrong.")
            self._reach_um = QDoubleSpinBox(); self._reach_um.setRange(0.1, 1000.0)
            self._reach_um.setSingleStep(0.5); self._reach_um.setValue(8.0)
            self._reach_um.setSuffix(" µm")
            self._mincil_um = QDoubleSpinBox(); self._mincil_um.setRange(0.0, 1000.0)
            self._mincil_um.setSingleStep(0.5); self._mincil_um.setValue(1.0)
            self._mincil_um.setSuffix(" µm")

            z_label = ("Z-step / slice spacing"
                       + ("  (from metadata)" if self._zum > 0 else ""))
            rows = [
                ("Strictness (false-positive control)",  self._strict),
                ("Both nuclei required",                 self._combine),
                ("Typical nucleus diameter",             self._nuc_um),
                (z_label,                                self._zstep_um),
                ("Max cilia reach from soma",            self._reach_um),
                ("Shortest cilium to keep",              self._mincil_um),
            ]
            for rr, (text, w) in enumerate(rows):
                pgrid.addWidget(QLabel(text), rr, 0)
                pgrid.addWidget(w, rr, 1)
            outer.addLayout(pgrid)

            hint = QLabel("Tip: if you get too many false positives, raise "
                          "strictness; too few cells, lower it.")
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #777; font-size: 10px;")
            outer.addWidget(hint)

            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                  QDialogButtonBox.StandardButton.Cancel)
            bb.accepted.connect(self._accept)
            bb.rejected.connect(self.reject)
            outer.addWidget(bb)

        def _recolour(self, combo: QComboBox):
            i = combo.currentIndex()
            if 0 <= i < len(self._channels):
                combo.setStyleSheet(
                    combo.styleSheet() +
                    f"\nQComboBox {{ color: {_rgb_hex(self._channels[i]['colour'])}; }}")

        def _accept(self):
            self.mapping = {
                stain: self._channels[c.currentIndex()]["raw"]
                for stain, c in self._combos.items()
            }
            self.params = DetectionParams.from_physical(
                self._um,
                nucleus_diam_um=self._nuc_um.value(),
                min_cilium_um=self._mincil_um.value(),
                cilia_reach_um=self._reach_um.value(),
                z_step_um=self._zstep_um.value(),
                nucleus_combine=self._combine.currentText(),
                strictness=self._strict.currentText().lower(),
            )
            self.accept()

    class DetectionWorker(QThread):
        """Runs detect_cells() off the GUI thread."""
        progress = pyqtSignal(int, str)
        finished = pyqtSignal(object)     # list[CellResult]
        error    = pyqtSignal(str)

        def __init__(self, reader, mapping, n_z, um_per_px, params):
            super().__init__()
            self._reader = reader
            self._mapping = mapping
            self._n_z = n_z
            self._um = um_per_px
            self._params = params
            self._abort = False

        def abort(self):
            self._abort = True

        def run(self):
            try:
                res = detect_cells(
                    self._reader, self._mapping, self._n_z, self._um, self._params,
                    progress=lambda pct, msg: self.progress.emit(pct, msg),
                    should_abort=lambda: self._abort,
                )
                self.finished.emit(res)
            except Exception:
                self.error.emit(traceback.format_exc())

except ImportError:
    # PyQt6 not available — the pure detect_cells() API still works headless.
    ChannelMapDialog = None     # type: ignore
    DetectionWorker = None      # type: ignore