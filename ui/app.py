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
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QSlider, QLabel,
    QTableWidget, QTableWidgetItem, QFrame,
    QHeaderView, QSizePolicy, QToolButton,
    QFileDialog, QStatusBar, QProgressDialog,
    QComboBox, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QObject
from PyQt6.QtGui import QFont, QAction, QActionGroup

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath


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
    a = arr2d.astype(np.float32)
    lo = float(np.percentile(a, lo_pct))
    hi = float(np.percentile(a, hi_pct))
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = np.zeros_like(a)
    return np.clip(a, 0.0, 1.0)


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
            um_per_px = 1.0
            xml_names: list = []
            xml_colours: list = []
            try:
                meta_xml: str = czi.metadata()
                root = ET.fromstring(meta_xml)

                channels_container = None
                for candidate in root.iter("Channels"):
                    channels_container = candidate
                    break

                if channels_container is not None:
                    for ch_el in channels_container.findall("Channel"):
                        xml_names.append(ch_el.get("Name", "") or "")
                        col_el = ch_el.find("Color")
                        if col_el is not None and col_el.text:
                            xml_colours.append(col_el.text.strip())
                        else:
                            xml_colours.append("")

                for dist in root.iter("Distance"):
                    if dist.get("Id", "") == "X":
                        val_el = dist.find("Value")
                        if val_el is not None and val_el.text is not None:
                            um_per_px = float(val_el.text) * 1e6
                        break

            except Exception as e:
                print(f"[Scope] Metadata parse warning: {e}")

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
            print(f"[Scope] Z = {n_z}   {um_per_px:.4f} µm/px")
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

            # ── Max-Z composite projection (fluorescence subset only) ────
            self.signals.progress.emit("Building max-Z projection…")
            print(f"[Scope] Projection using channels: {channel_subset}")

            composite_proj = np.zeros((H, W, 3), dtype=np.float32)

            for z in range(n_z):
                if self._abort:
                    czi.close()
                    return
                for i, raw_c in enumerate(channel_subset):
                    sl     = _select_plane(img, dims, raw_c, z).astype(np.float32)
                    f      = _norm_to_float(sl)
                    info   = channel_info[raw_c]
                    colour = _colour_for_channel(
                        info.get("name", ""),
                        info.get("color_hex", ""),
                        i   # fallback palette index cycles over the 4-entry subset
                    )
                    composite_proj[:, :, 0] = np.maximum(composite_proj[:, :, 0], f * colour[0])
                    composite_proj[:, :, 1] = np.maximum(composite_proj[:, :, 1], f * colour[1])
                    composite_proj[:, :, 2] = np.maximum(composite_proj[:, :, 2], f * colour[2])
                self.signals.pct.emit(15 + int(80 * (z + 1) / max(1, n_z)))

            peak = composite_proj.max()
            if peak > 0:
                composite_proj /= peak
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
                "projection":     proj_u8,
                "channel_info":   channel_info,   # length == n_c_pixel
                "channel_subset": channel_subset, # e.g. [20, 21, 22, 23]
                "scene_key":      scene_key,
                "dims":           dims,
                # Kept for backwards-compatibility with the rest of the app:
                "axes":           "".join(dims),
                "shape":          tuple(img.shape),
                "raw_shape":      tuple(img.shape),
                "ndim":           ndim,
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
# ImageCanvas  (top-left)
# ---------------------------------------------------------------------------

class ImageCanvas(FigureCanvas):
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
        self._pan_start       = None
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
            self._pan_start  = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
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
        if event.inaxes != self.ax or event.xdata is None or self._press_xy is None:
            return
        if (self._mode == MODE_NORMAL
                and self._pan_start is not None
                and self._xlim_start is not None
                and self._ylim_start is not None):
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]
            self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
            self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
            self.fig.canvas.draw_idle()
        elif self._mode == MODE_ADD and self._rect_patch is not None:
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
            self._pan_start = None
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

class ZSliceCanvas(FigureCanvas):
    measurement_done = pyqtSignal(float)

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
        self._pan_start     = None
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
            self._pan_start  = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._measure_mode and self._freehand_line is not None:
            self._freehand_xy.append((event.xdata, event.ydata))
            self._freehand_line.set_data(
                [p[0] for p in self._freehand_xy],
                [p[1] for p in self._freehand_xy]
            )
            self.fig.canvas.draw_idle()
        elif (not self._measure_mode
              and self._pan_start is not None
              and self._xlim_start is not None
              and self._ylim_start is not None):
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]
            self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
            self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        if self._measure_mode and len(self._freehand_xy) > 1:
            pts = np.array(self._freehand_xy)
            diffs = np.diff(pts, axis=0)
            length_px = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
            mid = pts[len(pts) // 2]
            if self._measure_text is not None:
                try:
                    self._measure_text.remove()
                except Exception:
                    pass
            self._measure_text = self.ax.text(
                mid[0], mid[1] - 8,
                f"{length_px:.1f} px",
                color="#ff0099", fontsize=9,
                ha="center", zorder=11,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a1a",
                          ec="#ff0099", alpha=0.7)
            )
            self.fig.canvas.draw_idle()
            self.measurement_done.emit(length_px)
        elif not self._measure_mode:
            self._pan_start = None
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
            self.canvas._pan_start  = None
            self.canvas._xlim_start = None
            self.canvas._ylim_start = None
            self.canvas._press_xy   = None
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
            self.open_file()

    def open_file(self, path: str = ""):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open .czi file", "", "CZI files (*.czi);;All files (*)"
            )
        if not path:
            self._show_open_prompt()
            return

        if self._loader is not None:
            self._loader.abort()
            self._loader.wait()

        self._progress = QProgressDialog("Opening CZI file…", "Cancel", 0, 100, self)
        self._progress.setWindowTitle("Loading")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(300)
        self._progress.setValue(0)

        self._loader = CziLoader(path, channel=0)
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
        self.canvas._pan_start  = None
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
    COLUMNS = ["img_name", "num", "x", "y", "z", "length"]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
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
        self.table.setStyleSheet("""
            QTableWidget {
                background: #1a1a1a; color: #d0d0d0;
                gridline-color: #333; border: none;
                font-family: 'Courier New'; font-size: 10px;
            }
            QTableWidget::item:selected { background: #003355; color: #fff; }
            QTableWidget::item:alternate { background: #1f1f1f; }
            QHeaderView::section {
                background: #252525; color: #888;
                border: none; border-bottom: 1px solid #444;
                padding: 4px 6px;
                font-family: 'Courier New'; font-size: 10px;
            }
        """)
        for i, w in enumerate([160, 45, 65, 65, 45, 80]):
            self.table.setColumnWidth(i, w)
        layout.addWidget(self.table, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
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

    def add_row(self, img_name, num, x, y, z, length):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for col, val in enumerate([img_name, num, x, y, z, length]):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, col, item)

    def remove_rows(self, indices: list):
        for i in sorted(indices, reverse=True):
            self.table.removeRow(i)

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", "cilia_measurements.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            import csv
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)
                for row in range(self.table.rowCount()):
                    cells = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        cells.append(item.text() if item is not None else "")
                    writer.writerow(cells)
            print(f"[Scope] Exported to {path}")
        except Exception as e:
            print(f"[Scope] Export failed: {e}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ScopeWindow(QMainWindow):

    _overview_ready = pyqtSignal(object, bool)   # (proj_uint8, preserve_view)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scope")
        self.setMinimumSize(1280, 800)

        self._markers: list     = []
        self._current_file: str = ""
        self._czi_data: dict    = {}
        self._slice_reader: SliceReader | None = None
        self._current_channel: int = 0   # 0 = composite RGB, 1..N = subset index 0..N-1
        self._channel_subset: list = []  # absolute channel indices (e.g. [20,21,22,23])
        self._overview_worker = None

        self.overview_panel = OverviewPanel()
        self.zviewer_panel  = ZViewerPanel()

        self._apply_theme()
        self._build_layout()
        self._overview_ready.connect(self._on_overview_ready)

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

        self._channel_menu  = menubar.addMenu("Channel")
        self._channel_group = QActionGroup(self)
        self._channel_group.setExclusive(True)
        self._rebuild_channel_menu([])

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

        self.zviewer_panel.canvas.measurement_done.connect(self._on_measurement_done)
        self.zviewer_panel.measure_toggled.connect(
            lambda on: self.status.showMessage(
                "Measure mode — drag along the cilia to trace its length" if on
                else "Measure mode OFF"
            )
        )
        self.zviewer_panel.z_changed.connect(self._on_z_changed)

    def _rebuild_channel_menu(self, channel_info: list, raw_indices: list | None = None):
        """
        Rebuild the Channel menu from the fluorescence subset.
        channel_info: metadata for the subset channels only.
        raw_indices:  the real absolute indices (e.g. [20,21,22,23]) — used in labels.
        """
        if self._channel_menu is None:
            return
        self._channel_menu.clear()
        for act in self._channel_group.actions():
            self._channel_group.removeAction(act)

        def _make_action(label: str, idx: int):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(idx == 0)
            act.triggered.connect(lambda checked, i=idx: self._on_channel_selected(i))
            self._channel_group.addAction(act)
            if self._channel_menu is None:
                print("self._channel_menu failed to load properly")
                return
            self._channel_menu.addAction(act)

        _make_action("RGB (composite)", 0)
        self._channel_menu.addSeparator()
        for i, ch in enumerate(channel_info):
            name    = ch.get("name", "") or ""
            raw_idx = raw_indices[i] if raw_indices is not None else i
            # Show the true absolute index (e.g. "Ch 20: AF647-T1")
            label   = f"Ch {raw_idx}: {name}" if name else f"Ch {raw_idx}"
            _make_action(label, i + 1)

    def _on_channel_selected(self, combo_idx: int):
        self._current_channel = combo_idx
        z = self.zviewer_panel.z_slider.value()
        self._display_z_slice(z)
        self._update_overview(preserve_view=True)

    # ------------------------------------------------------------------
    # Overview projection
    # ------------------------------------------------------------------

    def _update_overview(self, preserve_view: bool = False):
        if self._slice_reader is None:
            return

        if hasattr(self, "_overview_worker") and self._overview_worker is not None:
            try:
                self._overview_worker.cancel()
                self._overview_worker.done.disconnect()
                self._overview_worker.quit()
                self._overview_worker.wait(300)
            except RuntimeError:
                pass
            self._overview_worker = None

        n_c          = self._czi_data.get("n_c", 1)
        n_z          = self._czi_data.get("n_z", 1)
        channel_info = self._czi_data.get("channel_info", [])
        subset       = self._channel_subset if self._channel_subset else list(range(n_c))
        combo_idx    = self._current_channel
        reader       = self._slice_reader

        class _OverviewWorker(QThread):
            done = pyqtSignal(object, bool)

            def __init__(self, reader, combo_idx, n_c, n_z,
                         channel_info, subset, preserve_view):
                super().__init__()
                self._reader       = reader
                self._combo_idx    = combo_idx
                self._n_c          = n_c
                self._n_z          = n_z
                self._channel_info = channel_info
                self._subset       = subset
                self._preserve     = preserve_view
                self._cancelled    = False

            def cancel(self):
                self._cancelled = True

            def run(self):
                try:
                    if self._combo_idx == 0:
                        s0   = self._reader.get_composite_rgb(
                            0, self._n_c, self._channel_info,
                            channel_subset=self._subset)
                        H, W = s0.shape[0], s0.shape[1]
                        proj = np.zeros((H, W, 3), dtype=np.float32)
                        for z in range(self._n_z):
                            if self._cancelled:
                                return
                            frame = self._reader.get_composite_rgb(
                                z, self._n_c, self._channel_info,
                                channel_subset=self._subset
                            ).astype(np.float32)
                            proj = np.maximum(proj, frame)
                        proj_u8 = np.clip(proj, 0, 255).astype(np.uint8)
                    else:
                        s0   = self._reader.get_coloured_slice(
                            0, self._combo_idx - 1, self._channel_info,
                            channel_subset=self._subset)
                        H, W = s0.shape[0], s0.shape[1]
                        proj = np.zeros((H, W, 3), dtype=np.float32)
                        for z in range(self._n_z):
                            if self._cancelled:
                                return
                            frame = self._reader.get_coloured_slice(
                                z, self._combo_idx - 1, self._channel_info,
                                channel_subset=self._subset
                            ).astype(np.float32)
                            proj = np.maximum(proj, frame)
                        proj_u8 = np.clip(proj, 0, 255).astype(np.uint8)

                    if not self._cancelled:
                        self.done.emit(proj_u8, self._preserve)
                except Exception as e:
                    print(f"[Scope] Overview update error: {e}")

        worker = _OverviewWorker(reader, combo_idx, n_c, n_z,
                                 channel_info, subset, preserve_view)
        worker.done.connect(self._on_overview_ready)
        worker.finished.connect(worker.deleteLater)
        self._overview_worker = worker
        worker.start()

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

        if self._slice_reader is not None:
            try:
                self._slice_reader.close()
            except Exception:
                pass

        self._slice_reader = SliceReader(
            czi_data["path"],
            scene_key=czi_data.get("scene_key"),
            dims=czi_data.get("dims"),
        )

        # Use the channel_subset emitted by CziLoader directly — it is the
        # last 4 of the TRUE channel count (e.g. [20, 21, 22, 23]).
        self._channel_subset = czi_data.get("channel_subset", [])
        print(f"[Scope] _on_file_loaded: using channel_subset = {self._channel_subset}")

        self.zviewer_panel.configure(czi_data["n_z"])
        self._current_channel = 0

        # Build the Channel menu using the metadata for the subset channels only
        full_info = czi_data.get("channel_info", [])
        subset_info = [
            full_info[c] if c < len(full_info) else {"name": f"Ch{c}", "color_hex": ""}
            for c in self._channel_subset
        ]
        self._rebuild_channel_menu(subset_info, self._channel_subset)

        mid_z = czi_data["n_z"] // 2
        self._display_z_slice(mid_z, force_channel=0, preserve_view=False)
        # Show the static projection from CziLoader immediately, then
        # kick off the live overview (same channels, better normalisation).
        self.overview_panel.canvas.show_image(czi_data["projection"], preserve_view=False)
        self._update_overview(preserve_view=False)

        self.status.showMessage(
            f"Loaded: {self._current_file}  |  "
            f"Z={czi_data['n_z']}  C_total={czi_data['n_c']}  "
            f"fluorescence={self._channel_subset}  "
            f"|  {czi_data['um_per_px']:.4f} µm/px  |  "
            "Use Z-slider (right panel) to scroll Z layers"
        )

    # ------------------------------------------------------------------
    # Z / channel display
    # ------------------------------------------------------------------

    def _display_z_slice(self, z: int, force_channel: int = -1,
                         preserve_view: bool = True):
        if self._slice_reader is None:
            return

        n_c          = self._czi_data.get("n_c", 1)
        channel_info = self._czi_data.get("channel_info", [])
        subset       = self._channel_subset if self._channel_subset else list(range(n_c))
        combo_idx    = self._current_channel if force_channel == -1 else force_channel

        if combo_idx == 0:
            try:
                img = self._slice_reader.get_composite_rgb(
                    z, n_c, channel_info, channel_subset=subset)
            except Exception as e:
                print(f"[Scope] Z-slice read error (composite, z={z}): {e}")
                self.status.showMessage(f"Could not read Z={z} (composite): {e}")
                return
        else:
            try:
                img = self._slice_reader.get_coloured_slice(
                    z, combo_idx - 1, channel_info, channel_subset=subset)
            except Exception as e:
                print(f"[Scope] Z-slice read error (channel, z={z}): {e}")
                self.status.showMessage(f"Could not read Z={z}: {e}")
                return

        self.zviewer_panel.canvas.show_slice(img, preserve_view=preserve_view)

        um  = self._czi_data.get("um_per_px", 1.0)
        n_z = self._czi_data.get("n_z", 1)
        if combo_idx == 0:
            ch_label = "RGB"
        else:
            raw_idx  = subset[combo_idx - 1] if combo_idx - 1 < len(subset) else combo_idx - 1
            ch_label = (channel_info[raw_idx].get("name", "") or f"Ch {raw_idx}") \
                if raw_idx < len(channel_info) else f"Ch {raw_idx}"
        self.status.showMessage(
            f"{self._current_file}  |  Z = {z} / {n_z - 1}  "
            f"|  channel: {ch_label}  |  {um:.4f} µm/px"
        )

    @pyqtSlot(object, bool)
    def _on_overview_ready(self, proj_u8: np.ndarray, preserve_view: bool):
        self.overview_panel.canvas.show_image(proj_u8, preserve_view=preserve_view)

    @pyqtSlot(int)
    def _on_z_changed(self, z: int):
        self._display_z_slice(z)

    # ------------------------------------------------------------------
    # Marker management
    # ------------------------------------------------------------------

    def _on_cell_clicked(self, idx: int):
        if idx >= len(self._markers):
            return
        m = self._markers[idx]
        self.overview_panel.canvas.highlight_marker(idx)
        self.table_panel.table.selectRow(idx)
        self.status.showMessage(
            f"Cell {m['num']}  |  x={m['x']:.1f}  y={m['y']:.1f}  |  "
            "use Z-slider to step through slices"
        )

    def _on_marker_added(self, x: float, y: float):
        num = max((m["num"] for m in self._markers), default=0) + 1
        z   = self.zviewer_panel.z_slider.value()
        self._markers.append({"x": x, "y": y, "num": num, "accepted": True})
        self.overview_panel.canvas.set_markers(self._markers)
        self.table_panel.add_row(self._current_file or "unknown", num,
                                  round(x, 1), round(y, 1), z, "—")
        self.overview_panel.reset_mode()
        self._update_cell_count()
        self.status.showMessage(f"Cell {num} added at ({x:.1f}, {y:.1f})  z={z}")

    def _on_markers_deleted(self, indices: list):
        for i in sorted(indices, reverse=True):
            self._markers.pop(i)
        for i, m in enumerate(self._markers):
            m["num"] = i + 1
        self.overview_panel.canvas.set_markers(self._markers)
        self.table_panel.remove_rows(indices)
        self.overview_panel.reset_mode()
        self._update_cell_count()
        self.status.showMessage(f"{len(indices)} cell(s) removed")

    def _on_measurement_done(self, length_px: float):
        um_per_px = self._czi_data.get("um_per_px", 1.0)
        length_um = length_px * um_per_px
        selected = self.table_panel.table.selectedItems()
        if selected:
            row  = self.table_panel.table.currentRow()
            item = QTableWidgetItem(f"{length_um:.2f} µm")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_panel.table.setItem(row, 5, item)
        self.status.showMessage(
            f"Measurement: {length_px:.1f} px  ≈  {length_um:.2f} µm"
        )

    def _update_cell_count(self):
        n = sum(1 for m in self._markers if m["accepted"])
        self.overview_panel.status_lbl.setText(f"{n} cells")

    def closeEvent(self, a0: object) -> None:
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