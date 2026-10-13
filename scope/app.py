"""
Scope — Main application window
  Top-left:    Z-projection overview + cell markers
               + = rect-draw to add a cell
               - = freehand lasso to delete cells
               drag = pan
  Top-right:   Z-slice viewer
               📏 = freehand draw to measure cilia length (releases to show µm)
               🔍+/🔍- = zoom
               drag = pan
  Bottom:      Data table (img_name, num, x, y, z, length) + Export

CZI loading strategy (handles up to ~10 GB):
  • Uses `czifile` (the 2026 pure-Python rewrite, no CMake/C++ build required).
  • The channel count and channel names are read directly from the pixel data
    via  img.sizes['C']  and  img.channels  — NOT guessed from a scene loop.
  • The fluorescence subset is the LAST 4 channels of the true C count.
  • Each Z slice is read on demand with  img(C=c, Z=z).asarray()  so peak RAM
    stays ≈ one slice regardless of total file size.

Install:
    pip install -U "czifile[all]" scikit-image numpy matplotlib PyQt6
"""

import sys
import os
import threading
from collections import OrderedDict
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QSlider, QLabel,
    QTableWidget, QTableWidgetItem, QFrame,
    QHeaderView, QSizePolicy, QToolButton,
    QFileDialog, QStatusBar, QProgressDialog,
    QComboBox, QMessageBox,
    QDialog, QCheckBox, QGridLayout, QScrollArea, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QObject, QSize, QSettings
from PyQt6.QtGui import QFont, QAction, QActionGroup

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath

# Cell-detection module (optional — app still runs if it's missing).
try:
    from .analysis import (  # type: ignore[import-not-found]
        ChannelMapDialog, DetectionWorker, _fit_dialog_to_screen,
    )
except Exception as _e:   # pragma: no cover
    ChannelMapDialog = None
    DetectionWorker  = None

    def _fit_dialog_to_screen(dlg, parent=None, margin: int = 80):  # type: ignore[misc]
        return None

    print(f"[Scope] analysis module unavailable: {_e}")


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------
MODE_NORMAL = "normal"
MODE_ADD    = "add"
MODE_DELETE = "delete"

# Default false-colour fallback palette
CHANNEL_COLOURS = [
    [1.0, 0.0,   1.0  ],   # magenta  (AF647)
    [1.0, 0.094, 0.0  ],   # red-orange (AF568)
    [0.0, 1.0,   0.2  ],   # green    (AF488)
    [0.0, 0.8,   1.0  ],   # cyan     (DAPI)
    [1.0, 1.0,   0.0  ],   # yellow
    [0.0, 0.0,   1.0  ],   # blue
]

# Known fluorophore display colours matched by channel name prefix (case-insensitive).
FLUORO_COLOURS: list = [
    ("af647", [1.0,  0.0,   1.0  ]),   # #FF00FF  magenta
    ("af568", [1.0,  0.094, 0.0  ]),   # #FF1800  red-orange
    ("af488", [0.0,  1.0,   0.2  ]),   # #00FF33  green
    ("dapi",  [0.0,  0.8,   1.0  ]),   # #00CCFF  cyan
]


def _colour_for_channel(name: str, color_hex: str, fallback_idx: int) -> list:
    name_lo = (name or "").lower()
    for prefix, rgb in FLUORO_COLOURS:
        if name_lo.startswith(prefix):
            return rgb
    parsed = _hex_to_rgb_float(color_hex) if color_hex else None
    if parsed is not None:
        return parsed
    return CHANNEL_COLOURS[fallback_idx % len(CHANNEL_COLOURS)]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_to_float(arr2d: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Per-plane percentile stretch — only for when no global range is known.

    Stretching each plane on its own makes every plane fill the display range, so
    an empty slice at the top of a stack shows its noise as brightly as a slice
    full of real signal. That both makes Z-scrubbing flicker and, when the planes
    are max-projected afterwards, pushes that amplified noise into the
    projection. Prefer `_norm_with_range` with a per-channel volume range.
    """
    a = arr2d.astype(np.float32)
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0.0, 1.0)


def _norm_with_range(arr2d: np.ndarray, rng) -> np.ndarray:
    """Normalise a plane against a fixed (lo, hi) intensity range.

    Every plane of a channel goes through the same transfer function, so
    brightness is comparable across Z and consistent with the projection.
    """
    if rng is None:
        return _norm_to_float(arr2d)
    lo, hi = float(rng[0]), float(rng[1])
    a = arr2d.astype(np.float32)
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0.0, 1.0)


class _RangeEstimator:
    """Accumulates an intensity histogram for one channel across a whole stack.

    Integer image data (what a CZI holds) has few enough distinct values to
    histogram exactly, so whole-volume percentiles come out exact for the cost of
    one bincount per plane. Float data falls back to a subsample.
    """

    def __init__(self, lo_pct: float = 0.5, hi_pct: float = 99.5):
        self.lo_pct, self.hi_pct = lo_pct, hi_pct
        self._hist = None
        self._offset = 0
        self._samples: list = []

    def add(self, plane: np.ndarray):
        plane = np.asarray(plane)
        if np.issubdtype(plane.dtype, np.integer):
            flat = plane.ravel()
            mn, mx = int(flat.min()), int(flat.max())
            if self._hist is None:
                self._offset = min(0, mn)
                self._hist = np.zeros(max(mx - self._offset + 1, 256), dtype=np.int64)
            need = mx - self._offset + 1
            if need > self._hist.size:
                grown = np.zeros(need, dtype=np.int64)
                grown[:self._hist.size] = self._hist
                self._hist = grown
            self._hist += np.bincount(flat - self._offset,
                                      minlength=self._hist.size)[:self._hist.size]
        else:
            step = max(1, plane.size // 20000)
            self._samples.append(plane.ravel()[::step].astype(np.float32))

    def range(self):
        if self._hist is not None:
            total = int(self._hist.sum())
            if total <= 0:
                return (0.0, 1.0)
            cdf = np.cumsum(self._hist)
            lo = float(int(np.searchsorted(cdf, total * self.lo_pct / 100.0)) + self._offset)
            hi = float(int(np.searchsorted(cdf, total * self.hi_pct / 100.0)) + self._offset)
            return (lo, hi) if hi > lo else (lo, lo + 1.0)
        if self._samples:
            allv = np.concatenate(self._samples)
            lo = float(np.percentile(allv, self.lo_pct))
            hi = float(np.percentile(allv, self.hi_pct))
            return (lo, hi) if hi > lo else (lo, lo + 1.0)
        return (0.0, 1.0)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (_norm_to_float(arr) * 255).astype(np.uint8)


def _process_slice(arr2d: np.ndarray) -> np.ndarray:
    from skimage import filters
    f = _norm_to_float(arr2d)
    f = filters.gaussian(f, sigma=0.8)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def _colourise(arr2d: np.ndarray, colour: list) -> np.ndarray:
    f = _norm_to_float(arr2d)
    rgb = np.stack([f * colour[0],
                    f * colour[1],
                    f * colour[2]], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plane-selection helper (works with the 2026 czifile API)
# ---------------------------------------------------------------------------

def _select_plane(img, dims, c_idx: int, z_idx: int) -> np.ndarray:
    """
    Read a single (Y, X) plane for absolute channel c_idx and Z z_idx from a
    czifile CziImage.

    The 2026 czifile API selects dimensions by CALLING the image object, e.g.
        img(C=20, Z=5).asarray()
    Selections are absolute coordinates.  Any non-spatial dimension we are not
    explicitly addressing (T, H, R, S, ...) is pinned to 0 so the result
    squeezes down to 2-D.
    """
    kwargs = {}
    for d in dims:
        if d in ("Y", "X"):
            continue
        if d == "C":
            kwargs["C"] = c_idx
        elif d == "Z":
            kwargs["Z"] = z_idx
        else:
            kwargs[d] = 0
    arr = img(**kwargs).asarray()
    return np.squeeze(arr)


# ---------------------------------------------------------------------------
# CZI metadata parsing
# ---------------------------------------------------------------------------

def _parse_czi_metadata(meta_xml: str):
    """Pull channel names, display colours and voxel size out of a CZI's XML.

    Returns ``(names, colours, um_per_px, z_um)``. `colours` are raw hex strings
    as stored (``#AARRGGBB``); `z_um` is 0.0 when the file does not record it.

    A CZI holds MORE THAN ONE ``<Channels>`` block: the acquisition block, which
    carries the names, and a DisplaySetting block, which carries the colours the
    microscope or writing software assigned. This used to take the first block it
    found and read both from there, so colours came back empty on any file whose
    colours live in the display block — including files Scope itself had written,
    which then reopened in palette fallback colours rather than their own. Names
    come from the first block; colours from whichever block actually has them.
    """
    import xml.etree.ElementTree as ET

    names: list = []
    colours: list = []
    um_per_px, z_um = 1.0, 0.0
    if not meta_xml:
        return names, colours, um_per_px, z_um

    root = ET.fromstring(meta_xml)
    containers = list(root.iter("Channels"))
    name_container = containers[0] if containers else None
    colour_container = next(
        (c for c in containers
         if any(ch.find("Color") is not None for ch in c.findall("Channel"))),
        None)

    if colour_container is not None:
        for ch_el in colour_container.findall("Channel"):
            col_el = ch_el.find("Color")
            colours.append(col_el.text.strip()
                           if col_el is not None and col_el.text else "")

    if name_container is not None:
        for ch_el in name_container.findall("Channel"):
            names.append(ch_el.get("Name", "") or "")
            if colour_container is None:
                col_el = ch_el.find("Color")
                colours.append(col_el.text.strip()
                               if col_el is not None and col_el.text else "")

    # Scaling/Items gives per-axis Distance elements, in metres.
    for dist in root.iter("Distance"):
        axis   = dist.get("Id", "")
        val_el = dist.find("Value")
        if val_el is None or not val_el.text:
            continue
        try:
            v_um = float(val_el.text) * 1e6
        except ValueError:
            continue
        if axis == "X":
            um_per_px = v_um
        elif axis == "Z":
            z_um = v_um

    return names, colours, um_per_px, z_um


# ---------------------------------------------------------------------------
# CZI loader  (runs in a background QThread)
# ---------------------------------------------------------------------------

class CziSignals(QObject):
    progress = pyqtSignal(str)
    pct      = pyqtSignal(int)
    finished = pyqtSignal(object)   # CziData dict
    error    = pyqtSignal(str)


class CziLoader(QThread):
    """
    Background thread:
      1. Opens .czi with czifile (2026 rewrite).
      2. Picks a scene image handle (img = czi.scenes[key]).
      3. Reads the TRUE channel count from img.sizes['C'] and the channel
         names from img.channels — the authoritative pixel-data values.
      4. Derives the fluorescence subset as the LAST 4 channels.
      5. Builds a max-Z composite projection over that subset, plane-by-plane.

    Why this is the fix:
      The previous version counted channels by flattening czi.scenes into a
      per-scene "channel_map" and using its length.  Channels are a dimension
      *inside* a scene (shared across scenes), not a per-scene concatenation,
      so that count collapsed to 4 and "last 4 of 4" became the first 4.
      We now read C straight from the pixel data, so a 24-channel file reports
      24 and the subset is [20, 21, 22, 23].
    """

    def __init__(self, path: str, channel: int = 0):
        super().__init__()
        self.path    = path
        self.channel = channel
        self.signals = CziSignals()
        self._abort  = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            import czifile
            import xml.etree.ElementTree as ET

            self.signals.progress.emit("Opening file…")
            self.signals.pct.emit(5)

            czi = czifile.CziFile(self.path)

            # ── Pick a scene image handle ────────────────────────────────
            self.signals.progress.emit("Reading file structure…")
            try:
                scene_keys = list(czi.scenes.keys())
            except Exception:
                scene_keys = []

            if scene_keys:
                scene_key = scene_keys[0]
                img = czi.scenes[scene_key]
            else:
                # File reports no explicit scenes — merge whatever exists.
                scene_key = None
                img = czi.scenes()

            if len(scene_keys) > 1:
                print(f"[Scope] NOTE: file has {len(scene_keys)} scenes "
                      f"{scene_keys}; using scene {scene_key}. "
                      "Tell me if you need them merged into one view.")

            dims  = tuple(img.dims)
            sizes = dict(img.sizes)
            ndim  = len(img.shape)

            # `_select_plane` pins every dimension it is not addressing to 0. If
            # any of those actually has extent — a mosaic (M), a time series (T),
            # multiple rotations/illuminations — then everything downstream is
            # looking at one tile / one timepoint, silently. Say so loudly rather
            # than letting it pass for the whole image.
            ignored = {d: int(sizes.get(d, 1)) for d in dims
                       if d not in ("Y", "X", "C", "Z") and int(sizes.get(d, 1)) > 1}
            if ignored:
                detail = ", ".join(f"{d}={n}" for d, n in ignored.items())
                print("=" * 64)
                print(f"[Scope] WARNING: unhandled dimensions with extent: {detail}")
                print("[Scope] Only index 0 of each is read, so the view and every")
                print("[Scope] measurement cover ONE tile/timepoint, not the whole")
                print("[Scope] image. For a mosaic, stitch it in ZEN and re-open.")
                print("=" * 64)
                self.signals.progress.emit(
                    f"WARNING: showing index 0 of {detail} only")

            # ── TRUE channel count + Z, straight from the pixel data ─────
            n_c_pixel = int(sizes.get("C", 1))
            n_z       = int(sizes.get("Z", 1))

            # ── Channel names from the pixel data (authoritative) ────────
            try:
                pixel_names = [str(n) for n in img.channels]
            except Exception:
                pixel_names = []
            while len(pixel_names) < n_c_pixel:
                pixel_names.append(f"Ch{len(pixel_names)}")

            # ── Cross-check + display colours + pixel size from XML ──────
            try:
                xml_names, xml_colours, um_per_px, z_um = _parse_czi_metadata(
                    czi.metadata())
            except Exception as e:
                print(f"[Scope] Metadata parse warning: {e}")
                xml_names, xml_colours, um_per_px, z_um = [], [], 1.0, 0.0

            # ── Build channel_info (length == n_c_pixel) ─────────────────
            # Prefer pixel-data names; enrich with XML colours where indices
            # line up; fall back to XML names only if a pixel name is blank.
            channel_info: list = []
            for i in range(n_c_pixel):
                name = pixel_names[i] if i < len(pixel_names) else f"Ch{i}"
                if not name and i < len(xml_names):
                    name = xml_names[i]
                colour_hex = xml_colours[i] if i < len(xml_colours) else ""
                channel_info.append({"name": name, "color_hex": colour_hex})

            # ── Fluorescence subset = the LAST 4 channels ────────────────
            subset_size    = min(4, n_c_pixel)
            channel_subset = list(range(n_c_pixel - subset_size, n_c_pixel))

            # ── Diagnostics (so the file's real structure is visible) ────
            print("\n" + "=" * 64)
            print(f"[Scope] File : {os.path.basename(self.path)}")
            print(f"[Scope] dims = {dims}")
            print(f"[Scope] sizes= {sizes}")
            print(f"[Scope] C (pixel data) = {n_c_pixel}   "
                  f"C (XML <Channel> count) = {len(xml_names)}")
            if xml_names and len(xml_names) != n_c_pixel:
                print("[Scope] WARNING: XML channel count != pixel channel count. "
                      "Indexing uses the pixel-data count (you can only read "
                      "channels that exist as pixels).")
            z_tag = f"{z_um:.4f} µm/slice" if z_um > 0 else "(Z spacing not in metadata)"
            print(f"[Scope] Z = {n_z}   {um_per_px:.4f} µm/px   {z_tag}")
            print(f"[Scope] All {n_c_pixel} channels:")
            for i, ch in enumerate(channel_info):
                marker     = " ◀ fluorescence (used)" if i in channel_subset else ""
                colour_tag = ch['color_hex'] if ch['color_hex'] else "(name-matched colour)"
                print(f"  [{i:2d}]  name='{ch['name']}'  display_colour={colour_tag}{marker}")
            print(f"[Scope] Fluorescence subset (last {subset_size}): {channel_subset}")
            print("=" * 64 + "\n")

            self.signals.progress.emit(
                f"dims={dims}  |  C={n_c_pixel}  Z={n_z}  |  {um_per_px:.4f} µm/px"
            )
            self.signals.pct.emit(15)

            # ── Determine plane height/width from a sample read ──────────
            if self._abort:
                czi.close()
                return
            s0   = _select_plane(img, dims, channel_subset[0], 0)
            H, W = int(s0.shape[0]), int(s0.shape[1])

            # ── Per-channel max-Z projections (normalised grayscale) ─────
            # We keep ONE normalised projection plane per fluorescence channel
            # instead of a single baked RGB composite.  Toggling channels or
            # adjusting brightness/contrast then becomes a pure in-memory
            # recombination of these cached planes — no disk reads — so the
            # projection never lags behind the live view.
            self.signals.progress.emit("Building per-channel max-Z projections…")
            print(f"[Scope] Projection using channels: {channel_subset}")

            n_sub     = len(channel_subset)
            # Max-project the RAW values, and accumulate each channel's intensity
            # histogram as we go. Normalising each plane before the maximum (which
            # is what this used to do) stretches every plane to full scale first,
            # so slices holding nothing but noise contribute at the same
            # brightness as slices holding signal — the projection ends up
            # dominated by amplified background.
            proj_raw = [np.full((H, W), -np.inf, dtype=np.float32) for _ in range(n_sub)]
            est = [_RangeEstimator() for _ in range(n_sub)]

            for z in range(n_z):
                if self._abort:
                    czi.close()
                    return
                for i, raw_c in enumerate(channel_subset):
                    sl = _select_plane(img, dims, raw_c, z)
                    est[i].add(sl)
                    np.maximum(proj_raw[i], sl.astype(np.float32), out=proj_raw[i])
                self.signals.pct.emit(15 + int(80 * (z + 1) / max(1, n_z)))

            # One transfer function per channel, derived from the whole volume,
            # used for both the projection and every Z slice.
            display_ranges = [e.range() for e in est]
            proj_norm = [_norm_with_range(proj_raw[i], display_ranges[i])
                         for i in range(n_sub)]
            del proj_raw
            for i, raw_c in enumerate(channel_subset):
                lo, hi = display_ranges[i]
                print(f"[Scope] channel {raw_c} display range: {lo:.0f} – {hi:.0f}")

            # Default composite (all channels, max-combined) for the first paint.
            composite_proj = np.zeros((H, W, 3), dtype=np.float32)
            for i, raw_c in enumerate(channel_subset):
                info   = channel_info[raw_c]
                colour = _colour_for_channel(
                    info.get("name", ""), info.get("color_hex", ""), i
                )
                g = proj_norm[i]
                composite_proj[:, :, 0] = np.maximum(composite_proj[:, :, 0], g * colour[0])
                composite_proj[:, :, 1] = np.maximum(composite_proj[:, :, 1], g * colour[1])
                composite_proj[:, :, 2] = np.maximum(composite_proj[:, :, 2], g * colour[2])
            proj_u8 = (np.clip(composite_proj, 0, 1) * 255).astype(np.uint8)
            czi.close()

            self.signals.pct.emit(100)
            self.signals.finished.emit({
                "path":           self.path,
                "n_z":            n_z,
                # The TRUE total channel count, so the rest of the app can
                # address every channel:
                "n_c":            n_c_pixel,
                "um_per_px":      um_per_px,
                "z_um":           z_um,
                "projection":     proj_u8,
                "proj_norm":      proj_norm,       # list[(H,W) float32], one per subset channel
                "display_ranges": display_ranges,  # list[(lo, hi)] raw intensity, per subset channel
                "channel_info":   channel_info,   # length == n_c_pixel
                "channel_subset": channel_subset, # e.g. [20, 21, 22, 23]
                "scene_key":      scene_key,
                "dims":           dims,
                "ignored_dims":   ignored,
                "n_scenes":       len(scene_keys),
                # Kept for backwards-compatibility with the rest of the app:
                "axes":           "".join(dims),
                "shape":          tuple(img.shape),
                "raw_shape":      tuple(img.shape),
                "ndim":           ndim,
                "reader_kind":    "czi",
            })

        except ImportError:
            self.signals.error.emit(
                "czifile not installed.\n\nRun:\n  pip install -U \"czifile[all]\" scikit-image"
            )
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# SliceReader — keeps the CZI handle open for on-demand reads
# ---------------------------------------------------------------------------

class SliceReader:
    """
    On-demand Z-slice reader for the 2026 czifile API.

    Reads are done by selecting absolute coordinates on the scene image:
        img(C=c, Z=z).asarray()
    so only the requested plane is pulled from disk (RAM ≈ one slice).

    get_slice(z, channel)                          → uint8 (H,W)     grayscale
    get_coloured_slice(z, subset_idx, info, subset)→ uint8 (H,W,3)   false-colour
    get_composite_rgb(z, n_c, info, subset)        → uint8 (H,W,3)   composite
    """

    def __init__(self, path: str, scene_key=None, dims=None):
        import czifile
        self._czi = czifile.CziFile(path)

        # A CZI file handle has an internal read position + decode cache and is
        # NOT thread-safe.  The overview worker thread and the main GUI thread
        # both read through this one reader, so every actual read of the handle
        # must be serialised or the decompressor gets handed corrupted bytes
        # (ZSTD "Unknown frame descriptor" / "Src size is incorrect").
        self._lock = threading.Lock()

        try:
            keys = list(self._czi.scenes.keys())
        except Exception:
            keys = []

        if scene_key is not None and scene_key in keys:
            self._img = self._czi.scenes[scene_key]
        elif keys:
            self._img = self._czi.scenes[keys[0]]
        else:
            self._img = self._czi.scenes()

        self._dims = tuple(dims) if dims else tuple(self._img.dims)
        print(f"[SliceReader] dims={self._dims}  sizes={dict(self._img.sizes)}")

    def _raw_slice(self, c_idx: int, z_idx: int) -> np.ndarray:
        """Return squeezed 2-D (Y, X) array for one absolute channel + Z.

        Guarded by the instance lock so concurrent reads from the overview
        worker and the GUI thread can never corrupt the shared file handle.
        """
        with self._lock:
            return _select_plane(self._img, self._dims, c_idx, z_idx)

    def get_slice(self, z: int, channel: int) -> np.ndarray:
        """Grayscale processed uint8 (H, W).  `channel` is an absolute index."""
        return _process_slice(self._raw_slice(channel, z))

    def get_coloured_slice(self, z: int, subset_idx: int,
                           channel_info: list,
                           channel_subset: list | None = None) -> np.ndarray:
        """
        Single-channel false-colour RGB (H, W, 3) uint8.
        subset_idx is an index into channel_subset (0 = first fluorescence channel).
        channel_subset maps subset_idx → absolute channel index (e.g. 20..23).
        """
        raw_ch = channel_subset[subset_idx] if channel_subset is not None else subset_idx
        raw    = self._raw_slice(raw_ch, z)
        f      = _norm_to_float(raw)
        info   = channel_info[raw_ch] if raw_ch < len(channel_info) else {}
        colour = _colour_for_channel(
            info.get("name", ""),
            info.get("color_hex", ""),
            raw_ch  # fallback palette index
        )
        rgb = np.stack([f * colour[0], f * colour[1], f * colour[2]], axis=-1)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def get_composite_rgb(self, z: int, n_c: int,
                          channel_info: list,
                          channel_subset: list | None = None) -> np.ndarray:
        """
        Additive max-brightness composite → RGB uint8 (H, W, 3).
        channel_subset: absolute channel indices to blend; None = all n_c channels.
        """
        channels_to_use = channel_subset if channel_subset is not None else list(range(n_c))

        s0   = self._raw_slice(channels_to_use[0], z)
        H, W = s0.shape[0], s0.shape[1]
        comp = np.zeros((H, W, 3), dtype=np.float32)

        for i, raw_c in enumerate(channels_to_use):
            raw    = self._raw_slice(raw_c, z).astype(np.float32)
            f      = _norm_to_float(raw)
            info   = channel_info[raw_c] if raw_c < len(channel_info) else {}
            colour = _colour_for_channel(
                info.get("name", ""),
                info.get("color_hex", ""),
                i   # cycles over the subset for the fallback palette
            )
            comp[:, :, 0] += f * colour[0]
            comp[:, :, 1] += f * colour[1]
            comp[:, :, 2] += f * colour[2]

        peak = comp.max()
        if peak > 0:
            comp /= peak
        return (np.clip(comp, 0, 1) * 255).astype(np.uint8)

    def close(self):
        self._czi.close()


# ---------------------------------------------------------------------------
# OME-TIFF / TIFF support (read-only viewer path + detection input)
# ---------------------------------------------------------------------------

def _tiff_plane(arr: np.ndarray, axes: str, c_idx: int, z_idx: int) -> np.ndarray:
    """Pull one (Y, X) plane from an in-memory TIFF array given its axes string."""
    pos = {a: i for i, a in enumerate(axes)}
    idx: list = [slice(None)] * arr.ndim
    if "Z" in pos:
        idx[pos["Z"]] = int(z_idx)
    if "C" in pos:
        idx[pos["C"]] = int(c_idx)
    # Any other non-spatial leading axes (T, S, mosaic …) → take the first index.
    for a, i in pos.items():
        if a not in ("Y", "X") and isinstance(idx[i], slice):
            idx[i] = 0
    return np.squeeze(np.asarray(arr[tuple(idx)]))


class TiffSliceReader:
    """In-memory plane reader for (OME-)TIFF, mirroring SliceReader's interface.

    A NumPy array's reads are thread-safe, so unlike the CZI handle this needs
    no lock — the Z-slice worker and GUI thread can read concurrently.
    """

    def __init__(self, path: str, axes: str = "", shape=None):
        import tifffile
        with tifffile.TiffFile(path) as tf:
            series = tf.series[0]
            self._arr = np.asarray(series.asarray())
            self._axes = str(axes) if axes else str(series.axes)
        print(f"[TiffSliceReader] axes={self._axes} shape={self._arr.shape}")

    def _raw_slice(self, c_idx: int, z_idx: int) -> np.ndarray:
        return _tiff_plane(self._arr, self._axes, c_idx, z_idx) #type: ignore

    def get_slice(self, z: int, channel: int) -> np.ndarray:
        return _process_slice(self._raw_slice(channel, z))

    def close(self):
        self._arr = None


class TiffLoader(QThread):
    """Background loader for (OME-)TIFF that emits the same payload as CziLoader.

    Channel subset = ALL channels (TIFFs have no fluorescence convention), and
    pixel size / Z spacing are read from OME-XML when present (else 1.0 / unset).
    """

    def __init__(self, path: str, channel: int = 0):
        super().__init__()
        self.path    = path
        self.channel = channel
        self.signals = CziSignals()
        self._abort  = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            import tifffile
            from xml.etree import ElementTree as ET

            self.signals.progress.emit("Opening TIFF…")
            self.signals.pct.emit(5)

            with tifffile.TiffFile(self.path) as tf:
                series = tf.series[0]
                axes   = str(series.axes)
                shape  = tuple(int(s) for s in series.shape)
                ome    = tf.ome_metadata
                arr    = np.asarray(series.asarray())

            def axsize(a):
                return shape[axes.index(a)] if a in axes else 1

            n_z = axsize("Z"); n_c = axsize("C")
            H   = axsize("Y"); W   = axsize("X")

            um_per_px, z_um = 1.0, 0.0
            ch_names: list = []
            ch_colours: list = []
            if ome:
                try:
                    root = ET.fromstring(ome)
                    def local(tag):  # strip namespace
                        return tag.rsplit("}", 1)[-1]
                    for el in root.iter():
                        ln = local(el.tag)
                        if ln == "Pixels":
                            sx = el.attrib.get("PhysicalSizeX")
                            sz = el.attrib.get("PhysicalSizeZ")
                            if sx:
                                um_per_px = float(sx)
                            if sz:
                                z_um = float(sz)
                        elif ln == "Channel":
                            ch_names.append(el.attrib.get("Name", "") or "")
                            ch_colours.append(el.attrib.get("Color", "") or "")
                except Exception as e:
                    print(f"[Scope] OME metadata parse warning: {e}")

            channel_info: list = []
            for c in range(n_c):
                name = ch_names[c] if c < len(ch_names) else f"Ch{c}"
                channel_info.append({"name": name or f"Ch{c}", "color_hex": ""})

            channel_subset = list(range(n_c))

            print("\n" + "=" * 64)
            print(f"[Scope] TIFF : {os.path.basename(self.path)}")
            print(f"[Scope] axes = {axes}   shape = {shape}")
            z_tag = f"{z_um:.4f} µm/slice" if z_um > 0 else "(Z spacing not in metadata)"
            print(f"[Scope] Z = {n_z}   C = {n_c}   {um_per_px:.4f} µm/px   {z_tag}")

            self.signals.progress.emit("Building per-channel max-Z projections…")
            self.signals.pct.emit(15)
            # Project raw, normalise once — see the note in CziLoader.run.
            proj_raw = [np.full((H, W), -np.inf, dtype=np.float32)
                        for _ in channel_subset]
            est = [_RangeEstimator() for _ in channel_subset]
            for z in range(n_z):
                if self._abort:
                    return
                for i, c in enumerate(channel_subset):
                    pl = _tiff_plane(arr, axes, c, z)
                    est[i].add(pl)
                    np.maximum(proj_raw[i], pl.astype(np.float32), out=proj_raw[i])
                self.signals.pct.emit(15 + int(80 * (z + 1) / max(1, n_z)))
            display_ranges = [e.range() for e in est]
            proj_norm = [_norm_with_range(proj_raw[i], display_ranges[i])
                         for i in range(len(channel_subset))]
            del proj_raw

            composite = np.zeros((H, W, 3), dtype=np.float32)
            for i, c in enumerate(channel_subset):
                info   = channel_info[c]
                colour = _colour_for_channel(info.get("name", ""), info.get("color_hex", ""), i)
                g = proj_norm[i]
                composite[:, :, 0] = np.maximum(composite[:, :, 0], g * colour[0])
                composite[:, :, 1] = np.maximum(composite[:, :, 1], g * colour[1])
                composite[:, :, 2] = np.maximum(composite[:, :, 2], g * colour[2])
            proj_u8 = (np.clip(composite, 0, 1) * 255).astype(np.uint8)

            self.signals.pct.emit(100)
            self.signals.finished.emit({
                "path":           self.path,
                "n_z":            n_z,
                "n_c":            n_c,
                "um_per_px":      um_per_px,
                "z_um":           z_um,
                "projection":     proj_u8,
                "proj_norm":      proj_norm,
                "display_ranges": display_ranges,
                "channel_info":   channel_info,
                "channel_subset": channel_subset,
                "scene_key":      None,
                "dims":           tuple(axes),
                "axes":           axes,
                "shape":          shape,
                "raw_shape":      shape,
                "ndim":           int(arr.ndim),
                "reader_kind":    "tiff",
            })

        except ImportError:
            self.signals.error.emit(
                "tifffile not installed.\n\nRun:\n  pip install tifffile"
            )
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Colour helper
# ---------------------------------------------------------------------------

def _hex_to_rgb_float(hex_str: str) -> list:
    try:
        h = hex_str.lstrip("#")
        if len(h) == 8:
            r = int(h[2:4], 16) / 255.0
            g = int(h[4:6], 16) / 255.0
            b = int(h[6:8], 16) / 255.0
        elif len(h) == 6:
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
        else:
            return None  # type: ignore[return-value]
        if r == 0.0 and g == 0.0 and b == 0.0:
            return None  # type: ignore[return-value]
        return [r, g, b]
    except Exception:
        return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Shared pan helper
# ---------------------------------------------------------------------------

class _PanMixin:
    """Drag-to-pan that tracks the cursor in PIXELS, not data coordinates.

    Panning by data coordinates is self-defeating: `event.xdata` is where the
    cursor is *given the axes limits right now*, and panning is precisely the act
    of moving those limits. Each motion event therefore measures its delta
    against a frame the previous event just shifted, so the correction never
    settles and the image visibly shivers under the cursor.

    Pixels are an absolute frame. The scale (data units per pixel) is taken once,
    at button-press, from the starting limits and the axes box, and every delta
    for the rest of the drag is measured against the pixel position where the
    press happened. The image then tracks the cursor exactly.
    """

    def _pan_begin(self, event):
        self._pan_px    = (event.x, event.y)
        self._xlim_start = self.ax.get_xlim()
        self._ylim_start = self.ax.get_ylim()
        bbox = self.ax.get_window_extent()
        w = max(1.0, float(bbox.width))
        h = max(1.0, float(bbox.height))
        # Signed, so an inverted y-axis (which is what imshow uses) is handled
        # without a special case.
        self._pan_scale = (
            (self._xlim_start[1] - self._xlim_start[0]) / w,
            (self._ylim_start[1] - self._ylim_start[0]) / h,
        )

    def _pan_active(self) -> bool:
        return getattr(self, "_pan_px", None) is not None

    def _pan_to(self, event) -> bool:
        """Apply the drag. Works even when the cursor leaves the axes, so a pan
        no longer stalls the moment you drag past the edge of the image."""
        if not self._pan_active() or self._xlim_start is None or self._ylim_start is None:
            return False
        if event.x is None or event.y is None:
            return False
        dx = (event.x - self._pan_px[0]) * self._pan_scale[0]
        dy = (event.y - self._pan_px[1]) * self._pan_scale[1]
        self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
        self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
        self.fig.canvas.draw_idle()
        return True

    def _pan_end(self):
        self._pan_px = None


# ---------------------------------------------------------------------------
# ImageCanvas  (top-left)
# ---------------------------------------------------------------------------

class ImageCanvas(_PanMixin, FigureCanvas):
    cell_clicked    = pyqtSignal(int)
    marker_added    = pyqtSignal(float, float)
    markers_deleted = pyqtSignal(list)

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1a1a1a")
        self.ax  = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self.ax.set_facecolor("#1a1a1a")
        self.ax.axis("off")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._markers: list   = []
        self._mode            = MODE_NORMAL
        self._press_xy        = None
        self._rect_patch      = None
        self._lasso_xy: list  = []
        self._lasso_line      = None
        self._pan_px          = None
        self._xlim_start      = None
        self._ylim_start      = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def set_mode(self, mode: str):
        self._mode = mode
        cur = {
            MODE_NORMAL: Qt.CursorShape.ArrowCursor,
            MODE_ADD:    Qt.CursorShape.CrossCursor,
            MODE_DELETE: Qt.CursorShape.ForbiddenCursor,
        }
        self.setCursor(cur.get(mode, Qt.CursorShape.ArrowCursor))

    def show_image(self, image_array, preserve_view: bool = False):
        if preserve_view:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
        self.ax.cla()
        self.ax.axis("off")
        if image_array is not None:
            if image_array.ndim == 3:
                self.ax.imshow(image_array, interpolation="nearest")
            else:
                self.ax.imshow(image_array, cmap="gray", interpolation="nearest",
                               vmin=0, vmax=255)
        self._draw_markers()
        if preserve_view:
            self.ax.set_xlim(saved_xlim)  # type: ignore
            self.ax.set_ylim(saved_ylim)  # type: ignore
        self.fig.canvas.draw_idle()

    def zoom(self, factor: float):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xc = (xlim[0] + xlim[1]) / 2
        yc = (ylim[0] + ylim[1]) / 2
        self.ax.set_xlim(xc - (xlim[1]-xlim[0]) / 2 / factor,
                         xc + (xlim[1]-xlim[0]) / 2 / factor)
        self.ax.set_ylim(yc - (ylim[1]-ylim[0]) / 2 / factor,
                         yc + (ylim[1]-ylim[0]) / 2 / factor)
        self.fig.canvas.draw_idle()

    def set_markers(self, markers: list):
        self._markers = markers
        self._draw_markers()
        self.fig.canvas.draw_idle()

    def highlight_marker(self, idx: int):
        self._draw_markers(highlight=idx)
        self.fig.canvas.draw_idle()

    def _draw_markers(self, highlight: int = -1):
        for coll in list(self.ax.collections):
            coll.remove()
        for i, m in enumerate(self._markers):
            color = "#00d4ff" if m["accepted"] else "#ff4444"
            alpha = 0.95 if m["accepted"] else 0.6
            size  = 120 if i == highlight else 60
            edge  = "#ffff00" if i == highlight else "#ffffff"
            lw    = 2.0 if i == highlight else 1.2
            self.ax.scatter(
                [m["x"]], [m["y"]],
                c=color, s=size, marker="o",
                linewidths=lw, edgecolors=edge,
                zorder=5, alpha=alpha
            )

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._press_xy = (event.xdata, event.ydata)

        if self._mode == MODE_NORMAL:
            click_radius_sq = 10 ** 2
            for i, m in enumerate(self._markers):
                if (event.xdata - m["x"])**2 + (event.ydata - m["y"])**2 < click_radius_sq:
                    self.cell_clicked.emit(i)
                    self._press_xy = None
                    return
            self._pan_begin(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

        elif self._mode == MODE_ADD:
            self._rect_patch = Rectangle(
                (event.xdata, event.ydata), 0, 0,
                linewidth=1.5, edgecolor="#00ff88", facecolor="none",
                linestyle="--", zorder=10
            )
            self.ax.add_patch(self._rect_patch)

        elif self._mode == MODE_DELETE:
            self._lasso_xy   = [(event.xdata, event.ydata)]
            self._lasso_line, = self.ax.plot(
                [event.xdata], [event.ydata],
                color="#ff4444", linewidth=1.5, zorder=10
            )

    def _on_motion(self, event):
        # Pan first, and without the in-axes guard: a drag should keep working
        # when the cursor runs off the edge of the image.
        if self._mode == MODE_NORMAL and self._pan_active():
            self._pan_to(event)
            return
        if event.inaxes != self.ax or event.xdata is None or self._press_xy is None:
            return
        if self._mode == MODE_ADD and self._rect_patch is not None:
            x0, y0 = self._press_xy
            self._rect_patch.set_width(event.xdata - x0)
            self._rect_patch.set_height(event.ydata - y0)
            self.fig.canvas.draw_idle()
        elif self._mode == MODE_DELETE and self._lasso_line is not None:
            self._lasso_xy.append((event.xdata, event.ydata))
            self._lasso_line.set_data(
                [p[0] for p in self._lasso_xy],
                [p[1] for p in self._lasso_xy]
            )
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._press_xy is None:
            return
        if self._mode == MODE_NORMAL:
            self._pan_end()
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif self._mode == MODE_ADD and self._rect_patch is not None:
            if event.inaxes == self.ax and event.xdata is not None:
                x0, y0 = self._press_xy
                self.marker_added.emit((x0 + event.xdata) / 2, (y0 + event.ydata) / 2)
            self._rect_patch.remove()
            self._rect_patch = None
            self.fig.canvas.draw_idle()
        elif self._mode == MODE_DELETE and self._lasso_line is not None:
            if len(self._lasso_xy) > 2:
                path = MplPath(self._lasso_xy + [self._lasso_xy[0]])
                to_delete = [
                    i for i, m in enumerate(self._markers)
                    if path.contains_point((m["x"], m["y"]))
                ]
                if to_delete:
                    self.markers_deleted.emit(to_delete)
            self._lasso_line.remove()
            self._lasso_line = None
            self._lasso_xy   = []
            self.fig.canvas.draw_idle()
        self._press_xy = None


# ---------------------------------------------------------------------------
# ZSliceCanvas  (top-right)
# ---------------------------------------------------------------------------

class ZSliceCanvas(_PanMixin, FigureCanvas):
    # Emits the traced polyline in image coordinates, [(x, y), ...]. The window
    # pairs it with the Z it was drawn on and accumulates a 3-D length; the
    # canvas itself has no idea which slice it is showing.
    trace_done = pyqtSignal(object)

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1a1a1a")
        self.ax  = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self.ax.set_facecolor("#1a1a1a")
        self.ax.axis("off")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._measure_mode  = False
        self._freehand_xy: list = []
        self._freehand_line = None
        self._measure_text  = None
        self._pan_px        = None
        self._xlim_start    = None
        self._ylim_start    = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def show_slice(self, slice_array, cell_mask=None, cilia_mask=None,
                   preserve_view: bool = False):
        if preserve_view:
            saved_xlim = self.ax.get_xlim()
            saved_ylim = self.ax.get_ylim()
        self.ax.cla()
        self.ax.axis("off")
        if slice_array is not None:
            if slice_array.ndim == 3:
                self.ax.imshow(slice_array, interpolation="nearest")
            else:
                self.ax.imshow(slice_array, cmap="gray", interpolation="nearest",
                               vmin=0, vmax=255)
        if cell_mask is not None:
            ov = np.zeros((*cell_mask.shape, 4))
            ov[cell_mask > 0] = [0.1, 0.4, 0.9, 0.35]
            self.ax.imshow(ov, interpolation="nearest")
        if cilia_mask is not None:
            ov = np.zeros((*cilia_mask.shape, 4))
            ov[cilia_mask > 0] = [0.9, 0.1, 0.6, 0.6]
            self.ax.imshow(ov, interpolation="nearest")
        if preserve_view:
            self.ax.set_xlim(saved_xlim)  # type: ignore
            self.ax.set_ylim(saved_ylim)  # type: ignore
        self.fig.canvas.draw_idle()

    def set_measure_mode(self, enabled: bool):
        self._measure_mode = enabled
        self._freehand_xy  = []
        for attr in ("_freehand_line", "_measure_text"):
            obj = getattr(self, attr)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self.fig.canvas.draw_idle()

    def zoom(self, factor: float):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xc = (xlim[0] + xlim[1]) / 2
        yc = (ylim[0] + ylim[1]) / 2
        self.ax.set_xlim(xc - (xlim[1]-xlim[0]) / 2 / factor,
                         xc + (xlim[1]-xlim[0]) / 2 / factor)
        self.ax.set_ylim(yc - (ylim[1]-ylim[0]) / 2 / factor,
                         yc + (ylim[1]-ylim[0]) / 2 / factor)
        self.fig.canvas.draw_idle()

    def focus_window(self, xc: float, yc: float, half: float):
        """Center the view on (xc, yc) with a half-width of `half` pixels.

        ylim is inverted to match matplotlib's image (origin-upper) convention.
        """
        self.ax.set_xlim(xc - half, xc + half)
        self.ax.set_ylim(yc + half, yc - half)
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._measure_mode:
            for attr in ("_freehand_line", "_measure_text"):
                obj = getattr(self, attr)
                if obj is not None:
                    try:
                        obj.remove()
                    except Exception:
                        pass
                    setattr(self, attr, None)
            self._freehand_xy = [(event.xdata, event.ydata)]
            self._freehand_line, = self.ax.plot(
                [event.xdata], [event.ydata],
                color="#ff0099", linewidth=2, zorder=10
            )
            self.fig.canvas.draw_idle()
        else:
            self._pan_begin(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_motion(self, event):
        # Pan first, and without the in-axes guard — see _PanMixin.
        if not self._measure_mode and self._pan_active():
            self._pan_to(event)
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._measure_mode and self._freehand_line is not None:
            self._freehand_xy.append((event.xdata, event.ydata))
            self._freehand_line.set_data(
                [p[0] for p in self._freehand_xy],
                [p[1] for p in self._freehand_xy]
            )
            self.fig.canvas.draw_idle()

    def show_measure_label(self, text: str):
        """Put the running measurement next to the last traced segment."""
        if not self._freehand_xy:
            return
        pts = np.array(self._freehand_xy)
        mid = pts[len(pts) // 2]
        if self._measure_text is not None:
            try:
                self._measure_text.remove()
            except Exception:
                pass
        self._measure_text = self.ax.text(
            mid[0], mid[1] - 8, text,
            color="#ff0099", fontsize=9,
            ha="center", zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a1a",
                      ec="#ff0099", alpha=0.7)
        )
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._measure_mode and len(self._freehand_xy) > 1:
            self.trace_done.emit(list(self._freehand_xy))
        elif not self._measure_mode:
            self._pan_end()
            self.setCursor(Qt.CursorShape.ArrowCursor)


# ---------------------------------------------------------------------------
# OverviewPanel  (top-left)
# ---------------------------------------------------------------------------

class OverviewPanel(QWidget):
    file_loaded = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        self.title_lbl = QLabel("Z-Projection (max intensity)")
        self.title_lbl.setFont(QFont("Courier New", 9))
        self.title_lbl.setStyleSheet("color: #00aaff;")
        header.addWidget(self.title_lbl)
        header.addStretch()

        btn_style = """
            QPushButton {
                background: #2a2a2a; color: #e0e0e0;
                border: 1px solid #444; border-radius: 4px;
            }
            QPushButton:hover   { background: #3a3a3a; border-color: #00aaff; }
            QPushButton:checked { background: #00aaff; color: #000; border-color: #00aaff; }
        """
        self.btn_plus    = QPushButton("+")
        self.btn_minus   = QPushButton("−")
        self.btn_zoomin  = QPushButton("🔍+")
        self.btn_zoomout = QPushButton("🔍−")
        for btn in (self.btn_plus, self.btn_minus):
            btn.setFixedSize(28, 28)
            btn.setFont(QFont("Courier New", 12, QFont.Weight.Bold))
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
        for btn in (self.btn_zoomin, self.btn_zoomout):
            btn.setFixedSize(40, 28)
            btn.setFont(QFont("Courier New", 9))
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
        self.btn_plus.clicked.connect(self._on_plus)
        self.btn_minus.clicked.connect(self._on_minus)
        self.btn_zoomin.clicked.connect(lambda: self.canvas.zoom(1.3))
        self.btn_zoomout.clicked.connect(lambda: self.canvas.zoom(1 / 1.3))
        header.addWidget(self.btn_zoomout)
        header.addWidget(self.btn_zoomin)
        header.addWidget(self.btn_plus)
        header.addWidget(self.btn_minus)
        layout.addLayout(header)

        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)

        self.status_lbl = QLabel("")
        self.status_lbl.setFont(QFont("Courier New", 8))
        self.status_lbl.setStyleSheet("color: #666;")
        layout.addWidget(self.status_lbl)

        self._open_cid = None
        self._loader   = None
        self._progress = None
        self._show_open_prompt()

    def _show_open_prompt(self):
        ax = self.canvas.ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        ax.axis("off")
        ax.text(0.5, 0.5, "Open .czi file",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color="#00aaff", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.6",
                          fc="#1e1e1e", ec="#00aaff", linewidth=1.5))
        ax.text(0.5, 0.38, "click to browse",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#555", fontfamily="monospace")
        self.canvas.fig.canvas.draw_idle()
        self._open_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_open_click
        )

    def _on_canvas_open_click(self, event):
        if self._open_cid is not None:
            self.canvas.mpl_disconnect(self._open_cid)
            self._open_cid = None
            self.canvas._pan_px     = None
            self.canvas._xlim_start = None
            self.canvas._ylim_start = None
            self.canvas._press_xy   = None
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self.open_file()

    def open_file(self, path: str = ""):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open image", "",
                "Microscopy images (*.czi *.tif *.tiff *.ome.tif);;"
                "CZI files (*.czi);;TIFF / OME-TIFF (*.tif *.tiff *.ome.tif);;"
                "All files (*)"
            )
        if not path:
            self._show_open_prompt()
            return

        if self._loader is not None:
            self._loader.abort()
            self._loader.wait()

        lower = path.lower()
        is_tiff = lower.endswith((".tif", ".tiff"))

        self._progress = QProgressDialog(
            "Opening TIFF…" if is_tiff else "Opening CZI file…",
            "Cancel", 0, 100, self)
        self._progress.setWindowTitle("Loading")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(300)
        self._progress.setValue(0)

        self._loader = TiffLoader(path) if is_tiff else CziLoader(path, channel=0)
        progress     = self._progress
        self._loader.signals.progress.connect(
            lambda msg: progress.setLabelText(msg) if progress is not None else None
        )
        self._loader.signals.pct.connect(self._progress.setValue)
        self._loader.signals.finished.connect(self._on_load_done)
        self._loader.signals.error.connect(self._on_load_error)
        self._progress.canceled.connect(self._loader.abort)
        self._loader.start()

    @pyqtSlot(object)
    def _on_load_done(self, czi_data: dict):
        if self._progress:
            self._progress.close()
        fname = os.path.basename(czi_data["path"])
        proj  = czi_data["projection"]

        self.canvas.show_image(proj)
        self.canvas._pan_px     = None
        self.canvas._xlim_start = None
        self.canvas._ylim_start = None
        self.canvas._press_xy   = None
        self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

        self.title_lbl.setText(
            f"Z-Projection (max)  —  {fname}  "
            f"|  Z={czi_data['n_z']}  C={czi_data['n_c']}  "
            f"|  {czi_data['um_per_px']:.4f} µm/px"
        )
        self.status_lbl.setText("File loaded — run detection to populate cells")

        # Loud, once, in the GUI: a partially-read image must not look like a
        # whole one, because every count and length downstream would be wrong.
        ignored = czi_data.get("ignored_dims") or {}
        n_scenes = int(czi_data.get("n_scenes") or 0)
        notes = []
        if ignored:
            notes.append("Unhandled dimensions: "
                         + ", ".join(f"{d}={n}" for d, n in ignored.items())
                         + ".\nOnly index 0 of each is read — this view and any "
                           "detection cover one tile/timepoint, not the whole image."
                           "\nFor a mosaic, stitch it in ZEN and re-open.")
        if n_scenes > 1:
            notes.append(f"This file has {n_scenes} scenes; scene "
                         f"{czi_data.get('scene_key')!r} is loaded. "
                         "Open the others separately to analyse them.")
        if notes:
            QMessageBox.warning(self, "Partial image loaded", "\n\n".join(notes))

        self.btn_plus.setEnabled(True)
        self.btn_minus.setEnabled(True)
        self.btn_zoomin.setEnabled(True)
        self.btn_zoomout.setEnabled(True)
        self.file_loaded.emit(czi_data)

    @pyqtSlot(str)
    def _on_load_error(self, msg: str):
        if self._progress:
            self._progress.close()
        QMessageBox.critical(self, "CZI Load Error", msg)
        self._show_open_prompt()

    def _on_plus(self, checked: bool):
        if checked:
            self.btn_minus.setChecked(False)
            self.canvas.set_mode(MODE_ADD)
            self.status_lbl.setText("Add mode — drag a rectangle around a cell")
        else:
            self.canvas.set_mode(MODE_NORMAL)
            self.status_lbl.setText("Normal mode")

    def _on_minus(self, checked: bool):
        if checked:
            self.btn_plus.setChecked(False)
            self.canvas.set_mode(MODE_DELETE)
            self.status_lbl.setText("Delete mode — freehand draw to remove cells")
        else:
            self.canvas.set_mode(MODE_NORMAL)
            self.status_lbl.setText("Normal mode")

    def reset_mode(self):
        self.btn_plus.setChecked(False)
        self.btn_minus.setChecked(False)
        self.canvas.set_mode(MODE_NORMAL)
        self.status_lbl.setText("")


# ---------------------------------------------------------------------------
# ZViewerPanel  (top-right)
# ---------------------------------------------------------------------------

class ZViewerPanel(QWidget):
    measure_toggled = pyqtSignal(bool)
    z_changed       = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        tool_col = QVBoxLayout()
        tool_col.setSpacing(4)
        tool_col.addStretch()

        self.btn_ruler   = QToolButton()
        self.btn_zoomin  = QToolButton()
        self.btn_zoomout = QToolButton()
        self.btn_ruler.setText("📏")
        self.btn_ruler.setCheckable(True)
        self.btn_ruler.setToolTip("Freehand measure — drag along cilia to measure length")
        self.btn_zoomin.setText("🔍+")
        self.btn_zoomin.setToolTip("Zoom in")
        self.btn_zoomout.setText("🔍−")
        self.btn_zoomout.setToolTip("Zoom out")

        btn_style = """
            QToolButton {
                background: #2a2a2a; color: #e0e0e0;
                border: 1px solid #444; border-radius: 4px;
                padding: 4px; font-size: 14px;
                min-width: 32px; min-height: 32px;
            }
            QToolButton:hover   { background: #3a3a3a; border-color: #00aaff; }
            QToolButton:checked { background: #ff0099; color: #fff; border-color: #ff0099; }
            QToolButton:pressed { background: #444; }
        """
        for btn in (self.btn_ruler, self.btn_zoomin, self.btn_zoomout):
            btn.setStyleSheet(btn_style)
            tool_col.addWidget(btn)

        tool_col.addStretch()
        layout.addLayout(tool_col)

        self.canvas = ZSliceCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        slider_col = QVBoxLayout()
        slider_col.setSpacing(2)
        z_top = QLabel("Z")
        z_top.setFont(QFont("Courier New", 8))
        z_top.setStyleSheet("color: #666;")
        z_top.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.z_slider = QSlider(Qt.Orientation.Vertical)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.setValue(0)
        self.z_slider.setTickPosition(QSlider.TickPosition.TicksLeft)
        self.z_slider.setTickInterval(5)
        self.z_slider.setStyleSheet("""
            QSlider::groove:vertical {
                background: #2a2a2a; width: 6px;
                border: 1px solid #444; border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #00aaff; height: 14px; width: 14px;
                margin: 0 -4px; border-radius: 7px;
            }
            QSlider::handle:vertical:hover { background: #33bbff; }
        """)
        self.z_val = QLabel("0")
        self.z_val.setFont(QFont("Courier New", 8))
        self.z_val.setStyleSheet("color: #00aaff;")
        self.z_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        slider_col.addWidget(z_top)
        slider_col.addWidget(self.z_slider, stretch=1)
        slider_col.addWidget(self.z_val)
        layout.addLayout(slider_col)

        self.btn_ruler.toggled.connect(self._on_ruler_toggle)
        self.btn_zoomin.clicked.connect(lambda: self.canvas.zoom(1.3))
        self.btn_zoomout.clicked.connect(lambda: self.canvas.zoom(1 / 1.3))

        self._load_placeholder()

    def _on_ruler_toggle(self, checked: bool):
        self.canvas.set_measure_mode(checked)
        self.measure_toggled.emit(checked)

    def _on_z_changed(self, value: int):
        self.z_val.setText(str(value))
        self.z_changed.emit(value)

    def configure(self, n_z: int):
        self.z_slider.setMaximum(max(0, n_z - 1))
        self.z_slider.setValue(n_z // 2)
        self.z_slider.setTickInterval(max(1, n_z // 10))

    def _load_placeholder(self):
        self.canvas.ax.cla()
        self.canvas.ax.set_facecolor("#1a1a1a")
        self.canvas.ax.axis("off")
        self.canvas.ax.text(
            0.5, 0.5, "Select a cell\nto view Z-slice",
            transform=self.canvas.ax.transAxes,
            ha="center", va="center",
            fontsize=10, color="#444", fontfamily="monospace"
        )
        self.canvas.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# DataTablePanel  (bottom)
# ---------------------------------------------------------------------------

class DataTablePanel(QWidget):
    """The measurements table.

    `img_name` and `num` are deliberately NOT live columns. They are pure
    functions of the export — the image the rows came from, and the row's
    position — so holding them in the table meant every add or delete had to
    rewrite the numbering of every row after it, and a stale number was one
    missed update away. They are generated in `_export` instead: `num` is just
    the row index at the moment you export, and `img_name` rides along invisibly
    on each row (so a CSV imported from a different image keeps its own name).
    """

    VIEW_COLUMNS   = ["x", "y", "z", "length"]
    EXPORT_COLUMNS = ["img_name", "num", "x", "y", "z", "length"]
    COLUMNS = EXPORT_COLUMNS          # back-compat for the import validator
    LENGTH_COL = 3                    # index of "length" among VIEW_COLUMNS

    # Per-row image name lives here rather than in a visible column.
    IMG_ROLE = Qt.ItemDataRole.UserRole + 1

    import_requested = pyqtSignal()
    export_requested = pyqtSignal()   # window supplies path defaults

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.default_img_name = ""        # set by the window on file load
        self.table = QTableWidget(0, len(self.VIEW_COLUMNS))
        self.table.setHorizontalHeaderLabels(self.VIEW_COLUMNS)
        _hdr = self.table.horizontalHeader()
        assert _hdr is not None
        _hdr.setStretchLastSection(True)
        _hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked |
            QTableWidget.EditTrigger.SelectedClicked
        )
        self.table.setAlternatingRowColors(True)
        # Selection has to be written to beat the alternate-row rule. Both
        # `::item:selected` and `::item:alternate` are single pseudo-states, so
        # they carry equal specificity and whichever appears LAST wins — which is
        # why every even row used to lose its highlight entirely. The compound
        # `:alternate:selected` selector is more specific than either, so it
        # settles the question regardless of order, and the `!active` variants
        # keep the row visibly selected while focus sits on the image canvas.
        self.table.setStyleSheet("""
            QTableWidget {
                background: #1a1a1a; color: #d0d0d0;
                gridline-color: #333; border: none;
                font-family: 'Courier New'; font-size: 10px;
                selection-background-color: #00558a;
                selection-color: #ffffff;
            }
            QTableWidget::item            { background: #1a1a1a; }
            QTableWidget::item:alternate  { background: #1f1f1f; }
            QTableWidget::item:selected,
            QTableWidget::item:alternate:selected,
            QTableWidget::item:selected:!active,
            QTableWidget::item:alternate:selected:!active {
                background: #00558a; color: #ffffff;
            }
            QHeaderView::section {
                background: #252525; color: #888;
                border: none; border-bottom: 1px solid #444;
                padding: 4px 6px;
                font-family: 'Courier New'; font-size: 10px;
            }
        """)
        for i, w in enumerate([90, 90, 60, 110]):   # x, y, z, length
            self.table.setColumnWidth(i, w)
        layout.addWidget(self.table, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_import = QPushButton("Import")
        self.btn_import.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        self.btn_import.setFixedSize(90, 32)
        self.btn_import.setStyleSheet("""
            QPushButton {
                background: #1a1a1a; color: #aaaaaa;
                border: 2px solid #555; border-radius: 4px;
            }
            QPushButton:hover  { background: #555; color: #fff; }
            QPushButton:pressed { background: #444; }
        """)
        self.btn_import.clicked.connect(self.import_requested.emit)
        btn_row.addWidget(self.btn_import)
        self.btn_export = QPushButton("Export")
        self.btn_export.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
        self.btn_export.setFixedSize(90, 32)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background: #1a1a1a; color: #00aaff;
                border: 2px solid #00aaff; border-radius: 4px;
            }
            QPushButton:hover  { background: #00aaff; color: #000; }
            QPushButton:pressed { background: #0088cc; }
        """)
        self.btn_export.clicked.connect(self._export)
        btn_row.addWidget(self.btn_export)
        layout.addLayout(btn_row)

    def add_row(self, x, y, z, length, img_name: str = ""):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for col, val in enumerate([x, y, z, length]):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, col, item)
        first = self.table.item(r, 0)
        if first is not None:
            first.setData(self.IMG_ROLE, img_name or self.default_img_name)

    def row_img_name(self, row: int) -> str:
        item = self.table.item(row, 0)
        if item is not None:
            v = item.data(self.IMG_ROLE)
            if v:
                return str(v)
        return self.default_img_name or "unknown"

    def remove_rows(self, indices: list):
        for i in sorted(indices, reverse=True):
            self.table.removeRow(i)

    def export_to(self, path: str) -> int:
        """Write the table as CSV, generating img_name and num at this moment.

        Returns the number of rows written.
        """
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.EXPORT_COLUMNS)
            n = 0
            for row in range(self.table.rowCount()):
                cells = [self.table.item(row, c).text()
                         if self.table.item(row, c) is not None else ""
                         for c in range(self.table.columnCount())]
                # num is the row's position at export time — nothing to keep in
                # sync while cells are being added and removed.
                writer.writerow([self.row_img_name(row), row + 1] + cells)
                n += 1
        return n

    def _export(self):
        self.export_requested.emit()


# ---------------------------------------------------------------------------
# Per-channel intensity adjustment (brightness / contrast / min-luminescence)
# ---------------------------------------------------------------------------

def _apply_levels(f: np.ndarray, min_lum: float,
                  contrast: float, brightness: float) -> np.ndarray:
    """
    Adjust a normalised intensity map (values in [0, 1]).

      min_lum    : black point / floor.  Intensities below this are clipped to
                   0 and the surviving range is rescaled back up to [0, 1].
      contrast   : gain about mid-grey (0.5).  1.0 = unchanged.
      brightness : overall gain.  1.0 = unchanged.

    Applied in that order, then clipped back to [0, 1].  This is a pure NumPy
    operation on an (H, W) array, so it is cheap enough to run on every slider
    move with no disk access.
    """
    out = f
    if min_lum > 0.0:
        denom = max(1e-6, 1.0 - min_lum)
        out = (out - min_lum) / denom
        out = np.clip(out, 0.0, 1.0)
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
    if brightness != 1.0:
        out = out * brightness
    return np.clip(out, 0.0, 1.0)


def _rgb_hex(colour: list) -> str:
    r, g, b = (int(round(max(0.0, min(1.0, c)) * 255)) for c in colour[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Projection cache worker
# ---------------------------------------------------------------------------

class _ProjCacheWorker(QThread):
    """
    Reads every Z plane of each requested channel ONCE and emits that channel's
    max-Z projection as a normalised float32 (H, W) map.  After this runs, the
    overview is rebuilt purely from these cached maps, so toggling channels or
    moving processing sliders never touches the disk again.

    Shares the (locked) SliceReader with the GUI thread; the reader's lock
    serialises access so the two threads can't corrupt the file handle.
    """

    channel_ready = pyqtSignal(int, object)   # (raw channel idx, float32 HxW)
    all_done      = pyqtSignal()

    def __init__(self, reader, subset: list, n_z: int, ranges: dict | None = None):
        super().__init__()
        self._reader = reader
        self._subset = list(subset)
        self._n_z    = max(1, int(n_z))
        self._ranges = dict(ranges or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for raw in self._subset:
            if self._cancelled:
                return
            # Max-project raw, normalise once at the end (see CziLoader.run).
            acc = None
            est = _RangeEstimator()
            have_range = raw in self._ranges
            for z in range(self._n_z):
                if self._cancelled:
                    return
                try:
                    pl = np.asarray(self._reader._raw_slice(raw, z))
                except Exception as e:
                    print(f"[Scope] proj-cache read error (c={raw}, z={z}): {e}")
                    continue
                if not have_range:
                    est.add(pl)
                pf = pl.astype(np.float32)
                acc = pf if acc is None else np.maximum(acc, pf)
            if acc is not None and not self._cancelled:
                rng = self._ranges.get(raw) or est.range()
                self.channel_ready.emit(raw, _norm_with_range(acc, rng))
        if not self._cancelled:
            self.all_done.emit()


# ---------------------------------------------------------------------------
# Z-slice reader worker  (keeps Z-scrubbing smooth)
# ---------------------------------------------------------------------------

class _ZSliceWorker(QThread):
    """
    Reads Z-slice planes OFF the GUI thread so the slider never blocks.

    Two things keep scrubbing smooth:
      • Coalescing — only the most-recently-requested Z is served.  If the
        user is dragging, intermediate Z values are skipped; we always catch
        up to wherever the slider has landed.
      • An LRU cache of normalised planes keyed by (channel, z), bounded by a
        memory budget, so revisiting a Z (scrubbing back and forth) is instant.

    Emits `ready(z, {raw_channel: float32 (H,W)})` for the visible channels of
    the served Z.  The GUI thread then composites (cheap) and displays.
    """

    ready = pyqtSignal(int, object)   # (z, {raw: float32 map})

    def __init__(self, reader, mem_budget_mb: int = 256, ranges: dict | None = None):
        super().__init__()
        self._reader = reader
        self._ranges = dict(ranges or {})
        self._cond   = threading.Condition()
        self._req    = None            # (z, tuple(raw channels))
        self._stop   = False
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._mem    = 0
        self._budget = max(32, int(mem_budget_mb)) * 1024 * 1024

    def request(self, z: int, raws):
        """Ask for a Z (replaces any pending request — newest wins)."""
        with self._cond:
            self._req = (int(z), tuple(raws))
            self._cond.notify()

    def stop(self):
        with self._cond:
            self._stop = True
            self._cond.notify()

    def _superseded(self) -> bool:
        with self._cond:
            return self._stop or self._req is not None

    def _get(self, raw: int, z: int) -> np.ndarray:
        key = (raw, z)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        # Same per-channel transfer function as the projection, so brightness
        # does not jump around as the Z slider moves.
        plane = _norm_with_range(self._reader._raw_slice(raw, z),
                                 self._ranges.get(raw))
        self._cache[key] = plane
        self._mem += plane.nbytes
        # Evict least-recently-used until back under budget.
        while self._mem > self._budget and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._mem -= old.nbytes
        return plane

    def run(self):
        while True:
            with self._cond:
                while self._req is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                req = self._req
                self._req = None
            if req is None:
                continue
            z, raws = req

            maps = {}
            for raw in raws:
                if self._superseded():       # a newer Z arrived — abandon this one
                    break
                try:
                    maps[raw] = self._get(raw, z)
                except Exception as e:
                    print(f"[Scope] Z read error (c={raw}, z={z}): {e}")
            else:
                # only emit if we finished the whole set without being superseded
                if maps:
                    self.ready.emit(z, maps)


# ---------------------------------------------------------------------------
# Process dialog — per-channel brightness / contrast / min-luminescence
# ---------------------------------------------------------------------------

class ProcessDialog(QDialog):
    """
    Non-modal panel with one row of sliders per channel.  Edits are written
    straight into the shared channel dicts and `levels_changed` is emitted so
    the main window can re-composite from cache live.
    """

    levels_changed     = pyqtSignal()
    visibility_changed = pyqtSignal(int, bool)   # (channel index, visible)

    # slider integer range → float value
    _BC_MAX = 300   # brightness / contrast slider: 0..300 → 0.00..3.00
    _ML_MAX = 99    # min-lum slider:               0..99  → 0.00..0.99

    def __init__(self, channels: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Process — brightness / contrast / floor")
        self.setModal(False)
        self.setMinimumWidth(460)
        self._channels = channels
        self.vis_checks: list = []
        self._rows: list = []   # per-channel (b_slider, c_slider, m_slider, labels)

        self.setStyleSheet("""
            QDialog { background: #161616; }
            QLabel  { color: #d0d0d0; font-family: 'Courier New'; font-size: 11px; }
            QCheckBox { color: #d0d0d0; font-family: 'Courier New'; font-size: 12px; }
            QSlider::groove:horizontal {
                background: #2a2a2a; height: 5px;
                border: 1px solid #444; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00aaff; width: 12px; margin: -5px 0; border-radius: 6px;
            }
            QSlider::handle:horizontal:hover { background: #33bbff; }
            QPushButton {
                background: #2a2a2a; color: #e0e0e0;
                border: 1px solid #444; border-radius: 4px; padding: 4px 10px;
                font-family: 'Courier New';
            }
            QPushButton:hover { background: #3a3a3a; border-color: #00aaff; }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner  = QWidget()
        grid   = QGridLayout(inner)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        # Header row
        for col, text in enumerate(["channel", "brightness", "contrast", "min lum"]):
            h = QLabel(text)
            h.setStyleSheet("color: #00aaff;")
            grid.addWidget(h, 0, col * 2, 1, 2 if col else 1)

        row = 1
        for i, ch in enumerate(channels):
            # Visibility + coloured name
            chk = QCheckBox(ch["name"] or f"Ch {ch['raw']}")
            chk.setChecked(bool(ch.get("visible", True)))
            swatch = _rgb_hex(ch["colour"])
            chk.setStyleSheet(f"QCheckBox {{ color: {swatch}; }}")
            chk.toggled.connect(lambda v, idx=i: self.visibility_changed.emit(idx, v))
            self.vis_checks.append(chk)
            grid.addWidget(chk, row, 0, 1, 2)

            b_s, b_l = self._make_slider(self._BC_MAX, int(round(ch["brightness"] * 100)))
            c_s, c_l = self._make_slider(self._BC_MAX, int(round(ch["contrast"]   * 100)))
            m_s, m_l = self._make_slider(self._ML_MAX, int(round(ch["min_lum"]    * 100)))

            b_s.valueChanged.connect(lambda val, idx=i, lbl=b_l: self._on_b(idx, val, lbl))
            c_s.valueChanged.connect(lambda val, idx=i, lbl=c_l: self._on_c(idx, val, lbl))
            m_s.valueChanged.connect(lambda val, idx=i, lbl=m_l: self._on_m(idx, val, lbl))

            grid.addWidget(self._slider_cell(b_s, b_l), row, 2, 1, 2)
            grid.addWidget(self._slider_cell(c_s, c_l), row, 4, 1, 2)
            grid.addWidget(self._slider_cell(m_s, m_l), row, 6, 1, 2)

            self._rows.append((b_s, c_s, m_s, b_l, c_l, m_l))
            row += 1

        scroll.setWidget(inner)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, stretch=1)
        self._inner = inner

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        reset = QPushButton("Reset all")
        reset.clicked.connect(self._reset_all)
        close = QPushButton("Close")
        close.clicked.connect(self.hide)
        btn_row.addWidget(reset)
        btn_row.addWidget(close)
        outer.addLayout(btn_row)

        _fit_dialog_to_screen(self, parent)

    def sizeHint(self):
        inner = self._inner.sizeHint()
        return QSize(max(480, inner.width() + 44), inner.height() + 76)

    # -- widget factories ------------------------------------------------
    def _make_slider(self, maximum: int, value: int):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setMinimum(0)
        s.setMaximum(maximum)
        s.setValue(max(0, min(maximum, value)))
        s.setFixedWidth(120)
        lbl = QLabel(f"{value / 100:.2f}")
        lbl.setFixedWidth(38)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return s, lbl

    def _slider_cell(self, slider, label) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(slider)
        h.addWidget(label)
        return w

    # -- slider handlers -------------------------------------------------
    def _on_b(self, idx, val, lbl):
        self._channels[idx]["brightness"] = val / 100.0
        lbl.setText(f"{val / 100:.2f}")
        self.levels_changed.emit()

    def _on_c(self, idx, val, lbl):
        self._channels[idx]["contrast"] = val / 100.0
        lbl.setText(f"{val / 100:.2f}")
        self.levels_changed.emit()

    def _on_m(self, idx, val, lbl):
        self._channels[idx]["min_lum"] = val / 100.0
        lbl.setText(f"{val / 100:.2f}")
        self.levels_changed.emit()

    def _reset_all(self):
        for i, ch in enumerate(self._channels):
            ch["brightness"], ch["contrast"], ch["min_lum"] = 1.0, 1.0, 0.0
            b_s, c_s, m_s, b_l, c_l, m_l = self._rows[i]
            for s, l, v in ((b_s, b_l, 100), (c_s, c_l, 100), (m_s, m_l, 0)):
                s.blockSignals(True)
                s.setValue(v)
                s.blockSignals(False)
                l.setText(f"{v / 100:.2f}")
        self.levels_changed.emit()

    def set_visible_state(self, idx: int, visible: bool):
        """Sync a checkbox from the outside (e.g. the Channel menu) without echoing."""
        if 0 <= idx < len(self.vis_checks):
            chk = self.vis_checks[idx]
            chk.blockSignals(True)
            chk.setChecked(visible)
            chk.blockSignals(False)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ScopeWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scope")
        self.setMinimumSize(1280, 800)

        self._markers: list     = []
        self._current_file: str = ""
        self._czi_data: dict    = {}
        self._slice_reader: SliceReader | None = None

        # Per-channel view/processing state (source of truth for both panels):
        #   {raw, name, colour, visible, brightness, contrast, min_lum}
        self._channels: list = []
        self._frame_shape = None          # (H, W) of a single plane, once known

        # Caches that turn channel toggles / slider moves into pure CPU work:
        self._proj_cache: dict = {}       # raw idx -> max-Z float map (H, W)
        self._zslice_z = None             # which Z the slice cache holds
        self._zslice_floats: dict = {}    # raw idx -> float plane for current Z
        self._zslice_preserve = True      # carry preserve_view into async ready

        self._proj_worker = None
        self._zworker = None              # background Z-slice reader
        self._detect_worker = None        # cell-detection worker
        self._detect_progress = None
        self._proc_dialog = None
        self._vis_actions: list = []      # checkable Channel-menu actions
        self._overlap_mode = False        # show colocalisation instead of additive
        self._display_ranges: dict = {}   # raw channel idx -> (lo, hi) intensity
        # Guards the two-way tie between the table selection and the canvas
        # marker, so selecting one does not bounce back and reselect the other.
        self._syncing_selection = False
        # 3-D ruler: list of (points, z) traces accumulated since the tool was
        # last enabled, so a cilium can be followed across slices.
        self._measure_track: list = []

        self.overview_panel = OverviewPanel()
        self.zviewer_panel  = ZViewerPanel()

        self._apply_theme()
        self._build_layout()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #141414; color: #d0d0d0;
                font-family: 'Courier New';
            }
            QSplitter::handle { background: #333; }
            QSplitter::handle:hover { background: #00aaff; }
            QStatusBar {
                background: #1a1a1a; color: #555;
                font-family: 'Courier New'; font-size: 9px;
                border-top: 1px solid #2a2a2a;
            }
        """)

    def _build_layout(self):
        menubar = self.menuBar()
        if menubar is None:
            print("menubar failed to load properly")
            return
        file_menu = menubar.addMenu("File")
        open_act  = QAction("Open .czi…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.overview_panel.open_file)
        if file_menu is None:
            print("file_menu not loading properly")
            return
        file_menu.addAction(open_act)

        # Channel menu = per-channel visibility toggles (non-exclusive)
        self._channel_menu = menubar.addMenu("Channel")
        self._rebuild_channel_menu([], [])

        # Image menu = the Process panel
        image_menu = menubar.addMenu("Image")
        self._process_act = QAction("Process… (brightness / contrast / floor)", self)
        self._process_act.setShortcut("Ctrl+P")
        self._process_act.setEnabled(False)
        self._process_act.triggered.connect(self._open_process_dialog)
        if image_menu is not None:
            image_menu.addAction(self._process_act)

        # Analyze menu = automated cell detection
        analyze_menu = menubar.addMenu("Analyze")
        self._detect_act = QAction("Detect ciliated GAD1+ cells…", self)
        self._detect_act.setShortcut("Ctrl+D")
        self._detect_act.setEnabled(False)
        self._detect_act.triggered.connect(self._open_detect)
        if analyze_menu is not None:
            analyze_menu.addAction(self._detect_act)

        v_split = QSplitter(Qt.Orientation.Vertical)
        h_top   = QSplitter(Qt.Orientation.Horizontal)
        h_top.addWidget(self._framed(self.overview_panel))
        h_top.addWidget(self._framed(self.zviewer_panel))
        h_top.setSizes([550, 550])

        self.table_panel = DataTablePanel()
        v_split.addWidget(h_top)
        v_split.addWidget(self._framed(self.table_panel))
        v_split.setSizes([520, 280])
        self.setCentralWidget(v_split)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Scope — open a .czi file to begin")

        self.overview_panel.canvas.cell_clicked.connect(self._on_cell_clicked)
        self.overview_panel.canvas.marker_added.connect(self._on_marker_added)
        self.overview_panel.canvas.markers_deleted.connect(self._on_markers_deleted)
        self.overview_panel.file_loaded.connect(self._on_file_loaded)
        self.table_panel.import_requested.connect(self._on_import_csv)
        self.table_panel.export_requested.connect(self._on_export_csv)
        _sel = self.table_panel.table.selectionModel()
        if _sel is not None:
            _sel.selectionChanged.connect(self._on_table_selection_changed)

        self.zviewer_panel.canvas.trace_done.connect(self._on_trace_done)
        self.zviewer_panel.measure_toggled.connect(self._on_measure_toggled)
        self.zviewer_panel.z_changed.connect(self._on_z_changed)

    # ------------------------------------------------------------------
    # Channel visibility (Channel menu)
    # ------------------------------------------------------------------

    def _rebuild_channel_menu(self, channels: list, raw_indices: list | None = None):
        """Rebuild the Channel menu as a set of per-channel visibility toggles."""
        if self._channel_menu is None:
            return
        self._channel_menu.clear()
        self._vis_actions = []

        show_all = QAction("Show all", self)
        show_all.triggered.connect(lambda: self._set_all_visible(True))
        hide_all = QAction("Hide all", self)
        hide_all.triggered.connect(lambda: self._set_all_visible(False))
        self._channel_menu.addAction(show_all)
        self._channel_menu.addAction(hide_all)
        self._channel_menu.addSeparator()

        overlap_act = QAction("Overlap (colocalisation)", self)
        overlap_act.setCheckable(True)
        overlap_act.setChecked(self._overlap_mode)
        overlap_act.setToolTip(
            "Show where the visible channels overlap (pixelwise minimum) "
            "instead of the additive composite"
        )
        overlap_act.toggled.connect(self._set_overlap_mode)
        self._channel_menu.addAction(overlap_act)
        self._channel_menu.addSeparator()

        for i, ch in enumerate(channels):
            name    = ch.get("name", "") or ""
            raw_idx = ch.get("raw", raw_indices[i] if raw_indices else i)
            label   = f"Ch {raw_idx}: {name}" if name else f"Ch {raw_idx}"
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(bool(ch.get("visible", True)))
            act.toggled.connect(lambda v, idx=i: self._set_channel_visible(idx, v, "menu"))
            self._channel_menu.addAction(act)
            self._vis_actions.append(act)

    def _set_all_visible(self, visible: bool):
        for i in range(len(self._channels)):
            self._set_channel_visible(i, visible, "all", render=False)
        self._render_overview(preserve_view=True)
        self._recomposite_z(preserve_view=True)
        self._save_channel_prefs()

    def _set_channel_visible(self, idx: int, visible: bool,
                             source: str = "", render: bool = True):
        if not (0 <= idx < len(self._channels)):
            return
        self._channels[idx]["visible"] = bool(visible)

        # Keep the menu action and the dialog checkbox in sync without echoing.
        if source != "menu" and idx < len(self._vis_actions):
            act = self._vis_actions[idx]
            act.blockSignals(True)
            act.setChecked(visible)
            act.blockSignals(False)
        if source != "dialog" and self._proc_dialog is not None:
            self._proc_dialog.set_visible_state(idx, visible)

        if render:
            self._render_overview(preserve_view=True)
            self._recomposite_z(preserve_view=True)
            self._save_channel_prefs()

    def _set_overlap_mode(self, on: bool):
        self._overlap_mode = bool(on)
        # Both panels share the compositor, so a recomposite from cache is all
        # that's needed — no disk reads.
        self._render_overview(preserve_view=True)
        self._recomposite_z(preserve_view=True)
        self._save_channel_prefs()

    # ------------------------------------------------------------------
    # Process dialog (brightness / contrast / min-luminescence)
    # ------------------------------------------------------------------

    def _open_process_dialog(self):
        if not self._channels:
            return
        if self._proc_dialog is None:
            self._proc_dialog = ProcessDialog(self._channels, self)
            self._proc_dialog.levels_changed.connect(self._on_processing_changed)
            self._proc_dialog.visibility_changed.connect(
                lambda i, v: self._set_channel_visible(i, v, "dialog"))
        self._proc_dialog.show()
        self._proc_dialog.raise_()
        self._proc_dialog.activateWindow()

    def _on_processing_changed(self):
        # Levels live in the cached float maps' colourisation step, so this is
        # pure CPU — recomposite both panels from cache, no disk access.
        self._render_overview(preserve_view=True)
        self._recomposite_z(preserve_view=True)
        self._save_channel_prefs()

    # ------------------------------------------------------------------
    # Remembering the processing settings
    # ------------------------------------------------------------------
    #
    # Levels are a property of how a stain was acquired, not of one file, so a
    # user working through a batch sets them once and wants them to stick — both
    # across files in a session and across restarts. They are keyed by channel
    # NAME (e.g. "AF647"), because the absolute channel index differs between
    # acquisitions while the fluorophore does not; the subset position is the
    # fallback when a name is missing.

    _PREF_KEYS = ("brightness", "contrast", "min_lum", "visible")

    def _settings(self) -> QSettings:
        return QSettings("Scope", "Scope")

    def _save_channel_prefs(self):
        if not self._channels:
            return
        try:
            import json
            by_name = {
                (ch.get("name") or f"__idx{i}"): {k: ch[k] for k in self._PREF_KEYS}
                for i, ch in enumerate(self._channels)
            }
            by_index = [{k: ch[k] for k in self._PREF_KEYS} for ch in self._channels]
            s = self._settings()
            s.setValue("channel_prefs_by_name", json.dumps(by_name))
            s.setValue("channel_prefs_by_index", json.dumps(by_index))
            s.setValue("overlap_mode", bool(self._overlap_mode))
        except Exception as e:
            print(f"[Scope] could not save processing settings: {e}")

    def _apply_saved_channel_prefs(self) -> bool:
        """Push the remembered levels onto the freshly-built channel list.

        Returns True if anything was restored.
        """
        if not self._channels:
            return False
        try:
            import json
            s = self._settings()
            by_name = json.loads(s.value("channel_prefs_by_name") or "{}")
            by_index = json.loads(s.value("channel_prefs_by_index") or "[]")
            self._overlap_mode = str(s.value("overlap_mode", "false")).lower() in (
                "true", "1", "yes")
        except Exception as e:
            print(f"[Scope] could not read saved processing settings: {e}")
            return False

        applied = 0
        for i, ch in enumerate(self._channels):
            saved = by_name.get(ch.get("name") or "")
            if saved is None and i < len(by_index):
                saved = by_index[i]
            if not isinstance(saved, dict):
                continue
            for k in self._PREF_KEYS:
                if k in saved:
                    ch[k] = (bool(saved[k]) if k == "visible" else float(saved[k]))
            applied += 1

        # Never restore into a completely blank view — a user who hid everything
        # before quitting would otherwise reopen to a black window with no clue
        # why. Levels are kept; visibility is reset.
        if applied and not any(ch["visible"] for ch in self._channels):
            for ch in self._channels:
                ch["visible"] = True
            print("[Scope] saved settings hid every channel — showing all instead")
        return applied > 0

    # ------------------------------------------------------------------
    # Cell detection (Analyze menu)
    # ------------------------------------------------------------------

    def _open_detect(self):
        if DetectionWorker is None or ChannelMapDialog is None:
            QMessageBox.warning(self, "Detection unavailable",
                                "The analysis module could not be imported.")
            return
        if not self._channels or self._slice_reader is None:
            return

        dlg = ChannelMapDialog(
            self._channels,
            self._czi_data.get("um_per_px", 1.0),
            self._czi_data.get("z_um", 0.0),
            self,
            source_path=self._czi_data.get("path", ""),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if self._detect_worker is not None:
            self._detect_worker.abort()
            self._detect_worker.wait(1000)
            self._detect_worker = None

        self._detect_progress = QProgressDialog("Detecting cells…", "Cancel", 0, 100, self)
        self._detect_progress.setWindowTitle("Analyze")
        self._detect_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._detect_progress.setMinimumDuration(0)
        # Keep the bar visible across every stage: by default a QProgressDialog
        # hides itself the moment a stage hits 100%, which makes slower stages
        # (e.g. the cilia-gating distance transform) look like nothing is there.
        self._detect_progress.setAutoReset(False)
        self._detect_progress.setAutoClose(False)
        self._detect_progress.setValue(0)

        n_z = self._czi_data.get("n_z", 1)
        um  = self._czi_data.get("um_per_px", 1.0)
        worker = DetectionWorker(self._slice_reader, dlg.mapping, n_z, um, dlg.params,
                                 save_path=getattr(dlg, "save_path", None),
                                 channel_meta=getattr(dlg, "channel_meta", None))
        worker.progress.connect(self._on_detect_progress)
        worker.finished.connect(self._on_detect_done)
        worker.error.connect(self._on_detect_error)
        worker.finished.connect(worker.deleteLater)
        self._detect_progress.canceled.connect(worker.abort)
        self._detect_worker = worker
        worker.start()

    @pyqtSlot(int, str)
    def _on_detect_progress(self, pct: int, msg: str):
        if self._detect_progress is not None:
            self._detect_progress.setValue(pct)
            self._detect_progress.setLabelText(msg)

    @pyqtSlot(object)
    def _on_detect_done(self, results: list):
        if self._detect_progress is not None:
            self._detect_progress.close()
            self._detect_progress = None

        # Replace any existing markers / table rows with the detected cells.
        # Clearing and refilling the table churns the selection; the guard stops
        # each intermediate state from yanking the canvas around.
        self._syncing_selection = True
        self.table_panel.table.setRowCount(0)
        self._markers = []
        for cell in results:
            row = cell.as_row(self._current_file or "unknown")
            # No "num" on the marker: a cell's number is its position in the
            # table, resolved at export, so nothing needs renumbering here.
            self._markers.append(
                {"x": cell.x, "y": cell.y, "accepted": True,
                 "z": int(cell.z), "length": row["length"]})
            self.table_panel.add_row(row["x"], row["y"], row["z"], row["length"],
                                     img_name=row["img_name"])
        self._syncing_selection = False
        self.overview_panel.canvas.set_markers(self._markers)
        self._update_cell_count()

        n     = len(results)
        n_cil = sum(1 for c in results if c.has_cilium)
        self.status.showMessage(
            f"Detection complete — {n} cell(s), {n_cil} with a measurable cilium")
        if n == 0:
            QMessageBox.information(
                self, "Detection complete",
                "Detection finished but found 0 cells.\n\n"
                "The most common cause is the soma brightness test rejecting every "
                "nucleus. Check the terminal output for the per-stage counts, then "
                "try lowering the soma differential %, widening the comparison shell, "
                "or confirming the GAD1 (Soma) channel is mapped correctly.")

    @pyqtSlot(str)
    def _on_detect_error(self, msg: str):
        if self._detect_progress is not None:
            self._detect_progress.close()
            self._detect_progress = None
        QMessageBox.critical(self, "Detection error", msg)

    # ------------------------------------------------------------------
    # Compositing (cheap, in-memory; shared by both panels)
    # ------------------------------------------------------------------

    def _composite(self, float_maps: dict) -> np.ndarray | None:
        """
        Combine the visible channels into an RGB uint8 image, applying each
        channel's brightness/contrast/floor first.  `float_maps` maps a raw
        channel index to its (H, W) normalised intensity map.

        Normal mode  : additive max-combine of each channel in its colour.
        Overlap mode : pixelwise minimum across the visible channels, shown in
                       white — i.e. only where ALL visible channels have signal
                       (colocalisation).  Needs at least two visible channels;
                       with fewer it falls back to the normal composite.
        """
        visible = [ch for ch in self._channels
                   if ch["visible"] and ch["raw"] in float_maps]
        if not visible:
            if self._frame_shape is not None:
                H, W = self._frame_shape
                return np.zeros((H, W, 3), dtype=np.uint8)
            return None

        H, W = float_maps[visible[0]["raw"]].shape[:2]

        if self._overlap_mode and len(visible) >= 2:
            adjusted = [
                _apply_levels(float_maps[ch["raw"]],
                              ch["min_lum"], ch["contrast"], ch["brightness"])
                for ch in visible
            ]
            overlap = adjusted[0]
            for f in adjusted[1:]:
                overlap = np.minimum(overlap, f)
            g = np.clip(overlap, 0.0, 1.0)
            comp = np.stack([g, g, g], axis=-1)   # white = both present
            return (comp * 255).astype(np.uint8)

        comp = np.zeros((H, W, 3), dtype=np.float32)
        for ch in visible:
            f   = _apply_levels(float_maps[ch["raw"]],
                                ch["min_lum"], ch["contrast"], ch["brightness"])
            col = ch["colour"]
            comp[:, :, 0] = np.maximum(comp[:, :, 0], f * col[0])
            comp[:, :, 1] = np.maximum(comp[:, :, 1], f * col[1])
            comp[:, :, 2] = np.maximum(comp[:, :, 2], f * col[2])
        return (np.clip(comp, 0, 1) * 255).astype(np.uint8)

    def _render_overview(self, preserve_view: bool = True):
        if not self._proj_cache:
            return  # cache not built yet — keep the static loader projection
        img = self._composite(self._proj_cache)
        if img is not None:
            self.overview_panel.canvas.show_image(img, preserve_view=preserve_view)

    # ------------------------------------------------------------------
    # Projection cache
    # ------------------------------------------------------------------

    def _start_proj_cache(self):
        if self._slice_reader is None or not self._channels:
            return
        if self._proj_worker is not None:
            try:
                self._proj_worker.cancel()
                self._proj_worker.wait(500)
            except RuntimeError:
                pass
            self._proj_worker = None

        self._proj_cache = {}
        subset = [ch["raw"] for ch in self._channels]
        worker = _ProjCacheWorker(self._slice_reader, subset,
                                  self._czi_data.get("n_z", 1),
                                  ranges=getattr(self, "_display_ranges", {}))
        worker.channel_ready.connect(self._on_proj_channel_ready)
        worker.all_done.connect(
            lambda: self.status.showMessage(
                "Projection cache ready — channel toggles and processing are now instant"
            )
        )
        worker.finished.connect(worker.deleteLater)
        self._proj_worker = worker
        worker.start()

    @pyqtSlot(int, object)
    def _on_proj_channel_ready(self, raw: int, fmap: np.ndarray):
        self._proj_cache[raw] = fmap
        if self._frame_shape is None:
            self._frame_shape = fmap.shape[:2]
        self._render_overview(preserve_view=True)

    def _framed(self, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { border: 1px solid #2a2a2a; border-radius: 2px; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return frame

    # ------------------------------------------------------------------
    # CZI file loaded
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_file_loaded(self, czi_data: dict):
        self._czi_data     = czi_data
        self._current_file = os.path.basename(czi_data["path"])
        self.table_panel.default_img_name = self._current_file

        # Stop background workers that hold the OLD reader BEFORE we close it,
        # otherwise a worker could read a closed file handle.
        if self._zworker is not None:
            try:
                self._zworker.stop()
                self._zworker.wait(800)
            except RuntimeError:
                pass
            self._zworker = None
        if self._proj_worker is not None:
            try:
                self._proj_worker.cancel()
                self._proj_worker.wait(800)
            except RuntimeError:
                pass
            self._proj_worker = None

        if self._slice_reader is not None:
            try:
                self._slice_reader.close()
            except Exception:
                pass

        if czi_data.get("reader_kind") == "tiff":
            self._slice_reader = TiffSliceReader(
                czi_data["path"],
                axes=czi_data.get("axes", ""),
                shape=czi_data.get("shape"),
            )
        else:
            self._slice_reader = SliceReader(
                czi_data["path"],
                scene_key=czi_data.get("scene_key"),
                dims=czi_data.get("dims"),
            )

        # Build the per-channel view/processing state from the fluorescence
        # subset (the last 4 of the true channel count, e.g. [20, 21, 22, 23]).
        subset    = czi_data.get("channel_subset", [])
        full_info = czi_data.get("channel_info", [])
        self._channels = []
        for i, raw in enumerate(subset):
            info = full_info[raw] if raw < len(full_info) else {"name": f"Ch{raw}", "color_hex": ""}
            self._channels.append({
                "raw":        raw,
                "name":       info.get("name", "") or f"Ch {raw}",
                "colour":     _colour_for_channel(info.get("name", ""),
                                                  info.get("color_hex", ""), i),
                "visible":    True,
                "brightness": 1.0,
                "contrast":   1.0,
                "min_lum":    0.0,
            })
        # Reapply whatever levels were last in use, so a batch of files opens
        # looking the same rather than resetting to flat defaults each time.
        if self._apply_saved_channel_prefs():
            shown = ", ".join(
                f"{c['name']}(b{c['brightness']:.2f} c{c['contrast']:.2f} "
                f"m{c['min_lum']:.2f}{'' if c['visible'] else ' hidden'})"
                for c in self._channels)
            print(f"[Scope] restored processing settings: {shown}")

        print(f"[Scope] _on_file_loaded: channels = "
              f"{[(c['raw'], c['name']) for c in self._channels]}")

        # A new file → tear down the old Process dialog (channels changed).
        if self._proc_dialog is not None:
            self._proc_dialog.close()
            self._proc_dialog.deleteLater()
            self._proc_dialog = None

        # Reset caches for the new file.
        self._proj_cache    = {}
        self._zslice_z      = None
        self._zslice_floats = {}
        self._frame_shape   = None

        # Per-channel display ranges measured over the whole volume by the loader,
        # keyed by absolute channel index. Shared by the projection and the Z view.
        subset_ranges = czi_data.get("display_ranges") or []
        self._display_ranges = {
            ch["raw"]: subset_ranges[i]
            for i, ch in enumerate(self._channels)
            if i < len(subset_ranges)
        }

        # Create the background Z-slice reader for this file.
        self._zworker = _ZSliceWorker(self._slice_reader, ranges=self._display_ranges)
        self._zworker.ready.connect(self._on_zslice_ready)
        self._zworker.start()

        self.zviewer_panel.configure(czi_data["n_z"])
        self._rebuild_channel_menu(self._channels)
        self._process_act.setEnabled(True)
        self._detect_act.setEnabled(DetectionWorker is not None)

        # Show the static projection from the loader immediately (gives the user
        # something to look at while the per-channel cache builds in the
        # background), then request the centre Z slice asynchronously.
        proj = czi_data["projection"]
        if proj is not None and getattr(proj, "ndim", 0) >= 2:
            self._frame_shape = proj.shape[:2]
        self.overview_panel.canvas.show_image(proj, preserve_view=False)

        mid_z = czi_data["n_z"] // 2
        self._request_z(mid_z, preserve_view=False)

        # Per-channel projection cache.  The loader already read every Z plane
        # once to build the static projection; it hands us those per-channel
        # max-Z maps in `proj_norm`, so we can seed the cache directly and make
        # channel toggles / processing sliders lag-free immediately — no second
        # disk pass.  Fall back to a background re-read only if it's missing.
        proj_norm = czi_data.get("proj_norm")
        if proj_norm and len(proj_norm) == len(self._channels):
            self._proj_cache = {
                ch["raw"]: proj_norm[i] for i, ch in enumerate(self._channels)
            }
            if self._frame_shape is None:
                self._frame_shape = proj_norm[0].shape[:2]
            self._render_overview(preserve_view=False)
            cache_ready = True
        else:
            self._start_proj_cache()
            cache_ready = False

        self.status.showMessage(
            f"Loaded: {self._current_file}  |  "
            f"Z={czi_data['n_z']}  C_total={czi_data['n_c']}  "
            f"fluorescence={subset}  |  {czi_data['um_per_px']:.4f} µm/px  |  "
            + ("channel toggles & processing are instant" if cache_ready
               else "building projection cache…")
        )

    # ------------------------------------------------------------------
    # Z-slice rendering (right panel) — reads happen off the GUI thread
    # ------------------------------------------------------------------

    def _request_z(self, z: int, preserve_view: bool = True):
        """Ask the background worker for a Z.  Returns immediately; the image
        is swapped in when `_on_zslice_ready` fires.  The current image stays
        on screen meanwhile, so scrubbing never blocks or flickers to black."""
        self._zslice_preserve = preserve_view
        self._update_z_status(z)
        visible = [ch["raw"] for ch in self._channels if ch["visible"]]
        if not visible:
            # Nothing to read — show black right away.
            self._zslice_z      = z
            self._zslice_floats = {}
            self._show_zslice_composite(preserve_view)
            return
        if self._zworker is not None:
            self._zworker.request(z, visible)

    def _recomposite_z(self, preserve_view: bool = True):
        """Re-render the right panel after a toggle / level / overlap change.
        If the needed planes are already cached for the current Z this is pure
        CPU; otherwise it asks the worker for the missing channels."""
        z = self.zviewer_panel.z_slider.value()
        if z != self._zslice_z:
            self._request_z(z, preserve_view)
            return
        visible = [ch["raw"] for ch in self._channels if ch["visible"]]
        missing = [r for r in visible if r not in self._zslice_floats]
        if missing and self._zworker is not None:
            self._zslice_preserve = preserve_view
            self._zworker.request(z, visible)   # newly-visible channels pop in on ready
        self._show_zslice_composite(preserve_view)

    @pyqtSlot(int, object)
    def _on_zslice_ready(self, z: int, maps: dict):
        # Ignore results for a Z the slider has already moved away from.
        if z != self.zviewer_panel.z_slider.value():
            return
        self._zslice_z      = z
        self._zslice_floats = maps
        if self._frame_shape is None and maps:
            self._frame_shape = next(iter(maps.values())).shape[:2]
        self._show_zslice_composite(getattr(self, "_zslice_preserve", True))

    def _show_zslice_composite(self, preserve_view: bool = True):
        img = self._composite(self._zslice_floats)
        if img is not None:
            self.zviewer_panel.canvas.show_slice(img, preserve_view=preserve_view)
            pf = getattr(self, "_pending_focus", None)
            if pf is not None:
                x, y, half, target_z = pf
                # Only snap the window once the intended Z is actually showing.
                if self.zviewer_panel.z_slider.value() == target_z:
                    self.zviewer_panel.canvas.focus_window(x, y, half)
                    self._pending_focus = None

    def _update_z_status(self, z: int):
        um  = self._czi_data.get("um_per_px", 1.0)
        n_z = self._czi_data.get("n_z", 1)
        on  = [c["name"] for c in self._channels if c["visible"]]
        mode = "overlap" if (self._overlap_mode and len(on) >= 2) else "composite"
        self.status.showMessage(
            f"{self._current_file}  |  Z = {z} / {n_z - 1}  "
            f"|  visible: {', '.join(on) if on else '(none)'}  "
            f"|  {mode}  |  {um:.4f} µm/px"
        )

    @pyqtSlot(int)
    def _on_z_changed(self, z: int):
        self._request_z(z)

    # ------------------------------------------------------------------
    # Marker management
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, idx: int):
        """A marker pin was clicked on the overview."""
        self._select_cell(idx, sync_table=True)

    def _on_table_selection_changed(self):
        """A table row was selected — do exactly what clicking its pin does.

        Driven from the selection signal rather than a click signal so keyboard
        navigation through the table works too. `_syncing_selection` breaks the
        loop that would otherwise form: selecting a row moves the canvas, which
        selects the row, which moves the canvas…
        """
        if self._syncing_selection:
            return
        row = self.table_panel.table.currentRow()
        if row < 0:
            return
        self._select_cell(row, sync_table=False)

    def _select_cell(self, idx: int, sync_table: bool = True):
        if idx < 0 or idx >= len(self._markers):
            return
        m = self._markers[idx]
        self.overview_panel.canvas.highlight_marker(idx)
        if sync_table:
            self._syncing_selection = True
            try:
                self.table_panel.table.selectRow(idx)
            finally:
                self._syncing_selection = False

        # Center the right view on a fixed 30 µm × 30 µm window at the cell's Z.
        um   = self._czi_data.get("um_per_px", 1.0) or 1.0
        half = (30.0 / um) / 2.0
        n_z  = max(1, self._czi_data.get("n_z", 1))
        z    = int(m.get("z", self.zviewer_panel.z_slider.value()))
        z    = max(0, min(z, n_z - 1))
        # Apply the focus once the (correct) Z slice has rendered.
        self._pending_focus = (float(m["x"]), float(m["y"]), float(half), z)

        slider = self.zviewer_panel.z_slider
        if slider.value() != z:
            slider.setValue(z)        # triggers the Z load → focus applied on ready
        else:
            self._request_z(z)        # already on this Z → re-render, then focus

        self.status.showMessage(
            f"Cell {idx + 1}  |  x={m['x']:.1f}  y={m['y']:.1f}  z={z}  "
            f"|  30 µm view"
        )

    def _on_marker_added(self, x: float, y: float):
        z = self.zviewer_panel.z_slider.value()
        self._markers.append({"x": x, "y": y, "accepted": True,
                              "z": int(z), "length": "—"})
        self.overview_panel.canvas.set_markers(self._markers)
        self.table_panel.add_row(round(x, 1), round(y, 1), z, "—",
                                 img_name=self._current_file or "unknown")
        self.overview_panel.reset_mode()
        self._update_cell_count()
        self.status.showMessage(
            f"Cell {len(self._markers)} added at ({x:.1f}, {y:.1f})  z={z}")

    def _suggested_output_path(self, suffix: str, ext: str) -> str:
        """`<folder of the open image>/<image name> - <suffix><ext>`.

        Falls back to a bare filename (so the dialog opens wherever it last was)
        when nothing is loaded.
        """
        src = self._czi_data.get("path") or ""
        if not src:
            return f"cilia - {suffix}{ext}"
        stem = os.path.basename(src)
        for e in (".ome.tif", ".ome.tiff", ".czi", ".tif", ".tiff"):
            if stem.lower().endswith(e):
                stem = stem[:-len(e)]
                break
        else:
            stem = os.path.splitext(stem)[0]
        return os.path.join(os.path.dirname(src), f"{stem} - {suffix}{ext}")

    def _on_export_csv(self):
        default = self._suggested_output_path("Measurements", ".csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", default, "CSV files (*.csv)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".csv"
        try:
            n = self.table_panel.export_to(path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return
        print(f"[Scope] Exported {n} row(s) to {path}")
        self.status.showMessage(
            f"Exported {n} cell(s) to {os.path.basename(path)}")

    def _on_import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import measurements (CSV)", "", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        import csv
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                fields = {(c or "").strip() for c in (reader.fieldnames or [])}
                required = {"x", "y", "z", "length"}
                if not required.issubset(fields):
                    QMessageBox.warning(
                        self, "Import failed",
                        "That CSV doesn't match the export format.\n\n"
                        f"Expected columns: "
                        f"{', '.join(DataTablePanel.EXPORT_COLUMNS)}\n"
                        f"(img_name and num are optional)")
                    return
                rows = list(reader)
        except Exception as e:
            QMessageBox.critical(self, "Import error", str(e))
            return

        def _num(val, default):
            try:
                return int(float(val))
            except (TypeError, ValueError):
                return default

        self._syncing_selection = True
        self.table_panel.table.setRowCount(0)
        self._markers = []
        skipped = 0
        for i, row in enumerate(rows, start=1):
            try:
                x = float(row["x"]); y = float(row["y"])
            except (TypeError, ValueError, KeyError):
                skipped += 1
                continue
            z      = _num(row.get("z"), 0)
            length = (row.get("length") or "—").strip() or "—"
            img    = (row.get("img_name") or self._current_file or "imported").strip()
            # The CSV's own "num" is ignored: numbering is regenerated on export
            # from row order. The image name is kept, so a CSV imported from a
            # different image still exports under that image's name.
            self._markers.append({"x": x, "y": y, "accepted": True,
                                  "z": z, "length": length})
            self.table_panel.add_row(round(x, 1), round(y, 1), z, length, img_name=img)

        self._syncing_selection = False
        self.overview_panel.canvas.set_markers(self._markers)
        self._update_cell_count()
        msg = f"Imported {len(self._markers)} cell(s) from {os.path.basename(path)}"
        if skipped:
            msg += f"  ({skipped} row(s) skipped)"
        self.status.showMessage(msg)

    def _on_markers_deleted(self, indices: list):
        self._syncing_selection = True
        for i in sorted(indices, reverse=True):
            self._markers.pop(i)
        # Nothing to renumber — cell numbers are generated at export from row
        # order, so a deletion is just a deletion.
        self.overview_panel.canvas.set_markers(self._markers)
        self.table_panel.remove_rows(indices)
        self._syncing_selection = False
        self.overview_panel.reset_mode()
        self._update_cell_count()
        self.status.showMessage(f"{len(indices)} cell(s) removed")

    # ------------------------------------------------------------------
    # Manual 3-D ruler
    # ------------------------------------------------------------------

    def _on_measure_toggled(self, on: bool):
        """Enabling the ruler starts a fresh measurement."""
        self._measure_track = []
        if on:
            self.status.showMessage(
                "Measure mode — trace the cilium on this slice, then change Z and "
                "trace the next part; lengths add up in 3-D"
            )
        else:
            self.status.showMessage("Measure mode OFF")

    def _measure_total_um(self):
        """Total physical length of the accumulated trace, in microns.

        A cilium rarely lies flat in one plane, so the ruler is a chain of traces,
        each drawn on its own slice. Within a trace the travel is in-plane and
        scales by `um_per_px`; between traces the jump also has an axial component
        of Δz slices, which scales by `z_um`. Measuring only in-plane on a single
        slice — which is all the old ruler did — systematically undercounts every
        cilium that runs through the stack, and the steeper it runs the worse the
        undercount gets.
        """
        um = float(self._czi_data.get("um_per_px", 1.0) or 1.0)
        z_um = float(self._czi_data.get("z_um", 0.0) or 0.0)
        total = 0.0
        prev_end = None            # (x, y, z)
        for pts, z in self._measure_track:
            arr = np.asarray(pts, dtype=np.float64)
            if arr.shape[0] > 1:
                d = np.diff(arr, axis=0)
                total += float(np.sum(np.hypot(d[:, 0], d[:, 1]))) * um
            if prev_end is not None:
                dx = (arr[0, 0] - prev_end[0]) * um
                dy = (arr[0, 1] - prev_end[1]) * um
                dz = (z - prev_end[2]) * z_um
                total += float(np.sqrt(dx * dx + dy * dy + dz * dz))
            prev_end = (float(arr[-1, 0]), float(arr[-1, 1]), int(z))
        return total, z_um

    @pyqtSlot(object)
    def _on_trace_done(self, pts: list):
        if not pts or len(pts) < 2:
            return
        z = self.zviewer_panel.z_slider.value()
        self._measure_track.append((pts, int(z)))
        total_um, z_um = self._measure_total_um()
        n = len(self._measure_track)

        self.zviewer_panel.canvas.show_measure_label(f"{total_um:.2f} µm")

        row_note = ""
        if self.table_panel.table.selectedItems():
            row  = self.table_panel.table.currentRow()
            item = QTableWidgetItem(f"{total_um:.2f} µm")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_panel.table.setItem(row, DataTablePanel.LENGTH_COL, item)
            row_note = f"  →  row {row + 1}"
        else:
            row_note = "  (select a table row to record it)"

        axial = ""
        if n > 1:
            axial = (f"  |  {n} segments across Z"
                     if z_um > 0 else
                     f"  |  {n} segments, Z spacing unknown — axial travel NOT counted")
        elif z_um <= 0:
            axial = "  |  Z spacing unknown"
        self.status.showMessage(f"Measured: {total_um:.2f} µm{axial}{row_note}")

    def _update_cell_count(self):
        n = sum(1 for m in self._markers if m["accepted"])
        self.overview_panel.status_lbl.setText(f"{n} cells")

    def closeEvent(self, a0: object) -> None:
        if self._proj_worker is not None:
            try:
                self._proj_worker.cancel()
                self._proj_worker.wait(500)
            except RuntimeError:
                pass
        if self._zworker is not None:
            try:
                self._zworker.stop()
                self._zworker.wait(800)
            except RuntimeError:
                pass
        if self._detect_worker is not None:
            try:
                self._detect_worker.abort()
                self._detect_worker.wait(1000)
            except RuntimeError:
                pass
        if self._proc_dialog is not None:
            try:
                self._proc_dialog.close()
            except Exception:
                pass
        if self._slice_reader is not None:
            try:
                self._slice_reader.close()
            except Exception:
                pass
        if a0 is not None and hasattr(a0, "accept"):
            a0.accept()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch():
    app = QApplication(sys.argv)
    app.setApplicationName("Scope")
    window = ScopeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()