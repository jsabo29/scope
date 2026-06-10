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
  • Uses `czifile` (pure-Python, no CMake/C++ build required).
  • Raw pixel data is read dimension-by-dimension — never the whole file at once.
  • Z-max projection is built slice-by-slice so peak RAM ≈ 2 × one_slice.
  • Each Z slice is processed (contrast-stretch + Gaussian denoise) on demand
    inside ZSliceCanvas.show_slice(), keeping RAM flat regardless of file size.
  • Channel compositing uses only the current slice.

Install:
    pip install czifile scikit-image numpy matplotlib PyQt6
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
    QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QObject
from PyQt6.QtGui import QFont

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


# ---------------------------------------------------------------------------
# CZI loader  (runs in a background QThread)
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Stretch a 2-D array to uint8 [0, 255]."""
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def _process_slice(arr2d: np.ndarray) -> np.ndarray:
    """
    Per-layer processing pipeline (uint8 in, uint8 out):
      1. Percentile contrast stretch (p2 / p98)
      2. Mild Gaussian denoise (σ=1)
    Runs entirely on a 2-D array — no full-stack memory needed.
    """
    from skimage import filters, exposure
    a = arr2d.astype(np.float32)
    p2, p98 = float(np.percentile(a, 2)), float(np.percentile(a, 98))
    if p98 > p2:
        a = exposure.rescale_intensity(a, in_range=(p2, p98))  # type: ignore[arg-type]
    a = filters.gaussian(a, sigma=1)
    return _to_uint8(a)


class CziSignals(QObject):
    progress   = pyqtSignal(str)          # status message
    pct        = pyqtSignal(int)          # 0-100
    finished   = pyqtSignal(object)       # CziData dict on success
    error      = pyqtSignal(str)          # error message


class CziLoader(QThread):
    """
    Background thread that:
      1. Opens the .czi file with czifile.
      2. Reads metadata (dims, shape, pixel size).
      3. Builds the max-Z projection channel-by-channel, slice-by-slice.
      4. Does NOT load the full stack into RAM.
    """

    def __init__(self, path: str, channel: int = 0):
        super().__init__()
        self.path    = path
        self.channel = channel          # which channel to use for projection
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

            # -----------------------------------------------------------
            # Axes / shape discovery
            # czifile does NOT expose .axes / .shape on CziFile itself.
            # The array is obtained via asarray(); shape is inferred from
            # it plus the XML metadata (SizeZ, Channel count, pixel size).
            # -----------------------------------------------------------
            self.signals.progress.emit("Reading array structure…")

            # asarray() for large files returns a memory-mapped view so
            # no pixels are loaded until individual elements are accessed.
            data: np.ndarray = czi.asarray()
            raw_shape: tuple = data.shape   # e.g. (1, 1, C, Z, Y, X, 1)
            ndim: int = data.ndim

            # ---- Parse metadata XML -----------------------------------
            n_z: int        = 1
            n_c: int        = 1
            um_per_px: float = 1.0
            try:
                meta_xml: str = czi.metadata()  # type: ignore[call-arg]
                root = ET.fromstring(meta_xml)

                ch_els = list(root.iter("Channel"))
                if ch_els:
                    n_c = len(ch_els)

                for el in root.iter("SizeZ"):
                    if el.text:
                        n_z = max(1, int(el.text))
                        break

                for dist in root.iter("Distance"):
                    id_attr = dist.get("Id", "")
                    val_el  = dist.find("Value")
                    if id_attr == "X" and val_el is not None and val_el.text is not None:
                        um_per_px = float(val_el.text) * 1e6   # m → µm
                        break
            except Exception:
                pass   # fall back to shape inference

            # ---- Fallback: infer C/Z from array shape -----------------
            squeezed  = np.squeeze(data)
            sq_shape  = squeezed.shape
            if n_z == 1 and n_c == 1:
                if squeezed.ndim == 4:
                    n_c, n_z = int(sq_shape[0]), int(sq_shape[1])
                elif squeezed.ndim == 3:
                    n_z = int(sq_shape[0])

            # Clamp to what the array actually contains
            if ndim >= 4:
                n_z = max(1, min(n_z, raw_shape[ndim - 4]))
            if ndim >= 5:
                n_c = max(1, min(n_c, raw_shape[ndim - 5]))

            self.signals.progress.emit(
                f"Shape: {raw_shape}  |  Z={n_z}  C={n_c}  "
                f"|  {um_per_px:.4f} µm/px"
            )
            self.signals.pct.emit(15)

            # ---- Per-slice extractor ----------------------------------
            # Standard czifile dim order (7-D): (S, T, C, Z, Y, X, A)
            # For ndim < 7 trailing dims are dropped.
            # We address from the END so the code works for any ndim >= 2.
            #   ndim-1 = A (or X if 6-D), ndim-2 = X (or Y), ...
            # For a 7-D array: C=ndim-5, Z=ndim-4, Y=ndim-3, X=ndim-2
            def _raw_slice(c_idx: int, z_idx: int) -> np.ndarray:
                idx: list = [0] * ndim
                # Y and X are always the last spatial dims
                y_ax = ndim - 3 if ndim >= 7 else ndim - 2
                x_ax = ndim - 2 if ndim >= 7 else ndim - 1
                idx[y_ax] = slice(None)
                idx[x_ax] = slice(None)
                if ndim >= 5:
                    c_ax = ndim - 5
                    idx[c_ax] = min(c_idx, raw_shape[c_ax] - 1)
                if ndim >= 4:
                    z_ax = ndim - 4
                    idx[z_ax] = min(z_idx, raw_shape[z_ax] - 1)
                return np.squeeze(data[tuple(idx)])  # type: ignore[index]

            # ---- Max-Z projection -------------------------------------
            self.signals.progress.emit("Building max-Z projection…")
            ch  = min(self.channel, n_c - 1)
            s0  = _raw_slice(ch, 0)
            H, W = int(s0.shape[-2]), int(s0.shape[-1])
            proj = np.zeros((H, W), dtype=np.float32)
            for z in range(n_z):
                if self._abort:
                    return
                sl = _raw_slice(ch, z).astype(np.float32)
                h, w = sl.shape[-2], sl.shape[-1]
                proj[:h, :w] = np.maximum(proj[:h, :w], sl[-H:, -W:])
                self.signals.pct.emit(15 + int(80 * (z + 1) / n_z))

            proj_u8 = _to_uint8(proj)
            czi.close()

            self.signals.pct.emit(100)
            self.signals.finished.emit({
                "path":       self.path,
                "raw_shape":  raw_shape,
                "ndim":       ndim,
                "n_z":        n_z,
                "n_c":        n_c,
                "um_per_px":  um_per_px,
                "projection": proj_u8,
                "axes":       "",        # not used by new SliceReader
                "shape":      raw_shape, # alias for compatibility
            })

        except ImportError:
            self.signals.error.emit(
                "czifile not installed.\n\nRun:\n  pip install czifile scikit-image"
            )
        except Exception:
            import traceback
            self.signals.error.emit(traceback.format_exc())


class SliceReader:
    """
    Keeps a czifile handle open for on-demand Z-slice reads.
    Uses the same positional ndim-based indexing as CziLoader.
    Call get_slice(z, channel) → processed uint8 (H, W).
    """

    def __init__(self, path: str, axes: str, shape: tuple):
        import czifile
        self._czi   = czifile.CziFile(path)
        self._data  = self._czi.asarray()
        self._ndim  = self._data.ndim
        self._shape = self._data.shape

    def _raw_slice(self, c_idx: int, z_idx: int) -> np.ndarray:
        ndim  = self._ndim
        shape = self._shape
        idx: list = [0] * ndim
        y_ax = ndim - 3 if ndim >= 7 else ndim - 2
        x_ax = ndim - 2 if ndim >= 7 else ndim - 1
        idx[y_ax] = slice(None)
        idx[x_ax] = slice(None)
        if ndim >= 5:
            c_ax = ndim - 5
            idx[c_ax] = min(c_idx, shape[c_ax] - 1)
        if ndim >= 4:
            z_ax = ndim - 4
            idx[z_ax] = min(z_idx, shape[z_ax] - 1)
        return np.squeeze(self._data[tuple(idx)])

    def get_slice(self, z: int, channel: int) -> np.ndarray:
        raw = self._raw_slice(channel, z)
        return _process_slice(raw)

    def get_composite_rgb(self, z: int, n_c: int) -> np.ndarray:
        channels = []
        for c in range(min(n_c, 3)):
            channels.append(self._raw_slice(c, z).astype(np.float32))
        # Normalize each channel independently
        H, W = channels[0].shape[-2], channels[0].shape[-1]
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for i, ch in enumerate(channels):
            a = ch.astype(np.float32)
            lo, hi = a.min(), a.max()
            if hi > lo:
                a = (a - lo) / (hi - lo)
            rgb[:, :, i] = a
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def close(self):
        self._czi.close()


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

        self._markers: list = []
        self._mode = MODE_NORMAL

        self._press_xy   = None
        self._rect_patch = None
        self._lasso_xy: list  = []
        self._lasso_line = None
        self._pan_start  = None
        self._xlim_start = None
        self._ylim_start = None

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

    def show_image(self, image_array):
        self.ax.cla()
        self.ax.axis("off")
        if image_array is not None:
            self.ax.imshow(image_array, cmap="gray", interpolation="nearest")
        self._draw_markers()
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
            self._lasso_xy = [(event.xdata, event.ydata)]
            self._lasso_line, = self.ax.plot(
                [event.xdata], [event.ydata],
                color="#ff4444", linewidth=1.5, zorder=10
            )

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or self._press_xy is None:
            return
        if self._mode == MODE_NORMAL and self._pan_start and self._xlim_start and self._ylim_start is not None:
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

    def show_slice(self, slice_array, cell_mask=None, cilia_mask=None):
        self.ax.cla()
        self.ax.axis("off")
        if slice_array is not None:
            if slice_array.ndim == 3:
                self.ax.imshow(slice_array, interpolation="nearest")
            else:
                self.ax.imshow(slice_array, cmap="gray", interpolation="nearest")
        if cell_mask is not None:
            ov = np.zeros((*cell_mask.shape, 4))
            ov[cell_mask > 0] = [0.1, 0.4, 0.9, 0.35]
            self.ax.imshow(ov, interpolation="nearest")
        if cilia_mask is not None:
            ov = np.zeros((*cilia_mask.shape, 4))
            ov[cilia_mask > 0] = [0.9, 0.1, 0.6, 0.6]
            self.ax.imshow(ov, interpolation="nearest")
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
        elif not self._measure_mode and self._pan_start and self._xlim_start and self._ylim_start is not None:
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
    # Emitted after a file is successfully loaded
    file_loaded = pyqtSignal(dict)   # payload = CziData dict from CziLoader

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header row
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
        self.btn_plus  = QPushButton("+")
        self.btn_minus = QPushButton("−")
        for btn in (self.btn_plus, self.btn_minus):
            btn.setFixedSize(28, 28)
            btn.setFont(QFont("Courier New", 12, QFont.Weight.Bold))
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            btn.setEnabled(False)
        self.btn_plus.clicked.connect(self._on_plus)
        self.btn_minus.clicked.connect(self._on_minus)
        header.addWidget(self.btn_plus)
        header.addWidget(self.btn_minus)
        layout.addLayout(header)

        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)

        self.status_lbl = QLabel("")
        self.status_lbl.setFont(QFont("Courier New", 8))
        self.status_lbl.setStyleSheet("color: #666;")
        layout.addWidget(self.status_lbl)

        self._open_cid  = None
        self._loader    = None
        self._progress  = None
        self._show_open_prompt()

    # ------------------------------------------------------------------
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

        # Kill any previous loader
        if self._loader is not None:
            self._loader.abort()
            self._loader.wait()

        # Progress dialog (modal)
        self._progress = QProgressDialog("Opening CZI file…", "Cancel", 0, 100, self)
        self._progress.setWindowTitle("Loading")
        self._progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._progress.setMinimumDuration(300)
        self._progress.setValue(0)

        self._loader = CziLoader(path, channel=0)
        progress = self._progress
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
        # Reset any pan bleed-through
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

        self.file_loaded.emit(czi_data)

    @pyqtSlot(str)
    def _on_load_error(self, msg: str):
        if self._progress:
            self._progress.close()
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "CZI Load Error", msg)
        self._show_open_prompt()

    # ------------------------------------------------------------------
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
    measure_toggled   = pyqtSignal(bool)
    z_changed         = pyqtSignal(int)      # emitted when slider moves
    channel_changed   = pyqtSignal(int)      # emitted when channel combo changes

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Tool buttons (left strip)
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

        # Channel selector
        self.ch_label = QLabel("Ch")
        self.ch_label.setFont(QFont("Courier New", 7))
        self.ch_label.setStyleSheet("color: #666;")
        self.ch_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.ch_combo = QComboBox()
        self.ch_combo.addItem("RGB")
        self.ch_combo.setFont(QFont("Courier New", 8))
        self.ch_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a; color: #e0e0e0;
                border: 1px solid #444; border-radius: 3px;
                padding: 2px 4px; min-width: 48px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a; color: #e0e0e0;
                selection-background-color: #003355;
            }
        """)
        self.ch_combo.currentIndexChanged.connect(
            lambda idx: self.channel_changed.emit(idx)
        )
        tool_col.addWidget(self.ch_label)
        tool_col.addWidget(self.ch_combo)
        tool_col.addStretch()
        layout.addLayout(tool_col)

        # Canvas
        self.canvas = ZSliceCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        # Z-slider (right strip)
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

    def configure(self, n_z: int, n_c: int):
        """Called after a file loads to set slider range and channel options."""
        self.z_slider.setMaximum(max(0, n_z - 1))
        self.z_slider.setValue(n_z // 2)
        self.z_slider.setTickInterval(max(1, n_z // 10))

        # Rebuild channel combo
        self.ch_combo.blockSignals(True)
        self.ch_combo.clear()
        self.ch_combo.addItem("RGB (composite)")
        for c in range(n_c):
            self.ch_combo.addItem(f"Ch {c}")
        self.ch_combo.blockSignals(False)
        self.ch_combo.setCurrentIndex(0)

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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scope")
        self.setMinimumSize(1280, 800)

        self._markers: list    = []
        self._current_file: str = ""
        self._czi_data: dict   = {}
        self._slice_reader: SliceReader | None = None

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
        v_split = QSplitter(Qt.Orientation.Vertical)

        h_top = QSplitter(Qt.Orientation.Horizontal)
        self.overview_panel = OverviewPanel()
        self.zviewer_panel  = ZViewerPanel()
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

        # Signals
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
        self.zviewer_panel.channel_changed.connect(self._on_channel_changed)

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
        self._czi_data    = czi_data
        self._current_file = os.path.basename(czi_data["path"])

        # Close previous reader if any
        if self._slice_reader is not None:
            try:
                self._slice_reader.close()
            except Exception:
                pass

        # Open a persistent slice reader for on-demand Z access
        self._slice_reader = SliceReader(
            czi_data["path"],
            czi_data["axes"],
            czi_data["shape"]
        )

        # Configure the Z-slider and channel combo
        self.zviewer_panel.configure(czi_data["n_z"], czi_data["n_c"])

        # Show the middle Z slice immediately
        mid_z = czi_data["n_z"] // 2
        self._display_z_slice(mid_z, channel_idx=0)

        self.status.showMessage(
            f"Loaded: {self._current_file}  |  "
            f"Z={czi_data['n_z']}  C={czi_data['n_c']}  "
            f"|  {czi_data['um_per_px']:.4f} µm/px  |  "
            "Use Z-slider (right panel) to scroll Z layers"
        )

    # ------------------------------------------------------------------
    # Z / channel navigation
    # ------------------------------------------------------------------

    def _current_channel(self) -> int:
        """0 = composite RGB, 1..N = channel index 0..N-1"""
        return self.zviewer_panel.ch_combo.currentIndex()

    def _display_z_slice(self, z: int, channel_idx: int = -1):
        if self._slice_reader is None:
            return
        n_c = self._czi_data.get("n_c", 1)
        ch  = self._current_channel() if channel_idx == -1 else channel_idx

        if ch == 0 or n_c == 1:
            # Composite RGB
            img = self._slice_reader.get_composite_rgb(z, n_c)
        else:
            # Single channel (combo index 1 → channel 0, etc.)
            img = self._slice_reader.get_slice(z, ch - 1)

        self.zviewer_panel.canvas.show_slice(img)

        um = self._czi_data.get("um_per_px", 1.0)
        self.status.showMessage(
            f"{self._current_file}  |  Z = {z} / {self._czi_data.get('n_z', 1) - 1}  "
            f"|  {um:.4f} µm/px"
        )

    @pyqtSlot(int)
    def _on_z_changed(self, z: int):
        self._display_z_slice(z)

    @pyqtSlot(int)
    def _on_channel_changed(self, _: int):
        z = self.zviewer_panel.z_slider.value()
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
            row = self.table_panel.table.currentRow()
            item = QTableWidgetItem(f"{length_um:.2f} µm")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_panel.table.setItem(row, 5, item)
        self.status.showMessage(
            f"Measurement: {length_px:.1f} px  ≈  {length_um:.2f} µm"
        )

    def _update_cell_count(self):
        n = sum(1 for m in self._markers if m["accepted"])
        self.overview_panel.status_lbl.setText(f"{n} cells")

    def closeEvent(self, a0: object) -> None:  # matches QWidget.closeEvent(self, a0)
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