"""
Scope — Main application window
  Top-left:    Z-projection overview with cell markers
               + button → rectangle-draw mode (add cell)
               - button → freehand-lasso mode (delete cells inside)
  Top-right:   Z-slice viewer with Z-slider, ruler, zoom+/zoom- buttons
               Pan by click-dragging the image
  Bottom-left: Navigation controls (x/y/z position + zoom slider)
  Bottom-right: Data table (img_name, num, x, y, z, length) + Export
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QGridLayout, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QSlider, QLabel,
    QTableWidget, QTableWidgetItem, QFrame,
    QDoubleSpinBox, QSpinBox, QHeaderView,
    QSizePolicy, QToolButton,
    QFileDialog, QStatusBar,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath


# ---------------------------------------------------------------------------
# Mode constants for ImageCanvas
# ---------------------------------------------------------------------------
MODE_NORMAL = "normal"
MODE_ADD    = "add"     # drag rectangle -> add cell at centre
MODE_DELETE = "delete"  # freehand lasso -> delete markers inside


# ---------------------------------------------------------------------------
# ImageCanvas
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

        self._image_data  = None
        self._markers: list = []
        self._mode = MODE_NORMAL

        self._press_xy   = None
        self._rect_patch = None
        self._lasso_xy:  list = []
        self._lasso_line = None
        self._pan_start  = None
        self._xlim_start = None
        self._ylim_start = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def set_mode(self, mode: str):
        self._mode = mode
        cursors = {
            MODE_NORMAL: Qt.CursorShape.ArrowCursor,
            MODE_ADD:    Qt.CursorShape.CrossCursor,
            MODE_DELETE: Qt.CursorShape.ForbiddenCursor,
        }
        self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))

    def show_image(self, image_array, title=""):
        self.ax.cla()
        self.ax.axis("off")
        if image_array is not None:
            self.ax.imshow(image_array, cmap="gray", interpolation="nearest")
        self._image_data = image_array
        self._draw_markers()
        self.fig.canvas.draw_idle()

    def set_markers(self, markers: list):
        self._markers = markers
        self._draw_markers()
        self.fig.canvas.draw_idle()

    def _draw_markers(self):
        for coll in list(self.ax.collections):
            coll.remove()
        if not self._markers:
            return
        acc_x = [m["x"] for m in self._markers if m["accepted"]]
        acc_y = [m["y"] for m in self._markers if m["accepted"]]
        rej_x = [m["x"] for m in self._markers if not m["accepted"]]
        rej_y = [m["y"] for m in self._markers if not m["accepted"]]
        if acc_x:
            self.ax.scatter(acc_x, acc_y, c="#00d4ff", s=60, marker="o",
                            linewidths=1.2, edgecolors="#ffffff", zorder=5, alpha=0.9)
        if rej_x:
            self.ax.scatter(rej_x, rej_y, c="#ff4444", s=60, marker="o",
                            linewidths=1.2, edgecolors="#ffffff", zorder=5, alpha=0.6)

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._press_xy = (event.xdata, event.ydata)

        if self._mode == MODE_NORMAL:
            for i, m in enumerate(self._markers):
                if (event.xdata - m["x"])**2 + (event.ydata - m["y"])**2 < 100:
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

        if self._mode == MODE_NORMAL and self._pan_start is not None and self._xlim_start is not None and self._ylim_start is not None:
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
# ZSliceCanvas
# ---------------------------------------------------------------------------

class ZSliceCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1a1a1a")
        self.ax  = self.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self.ax.set_facecolor("#1a1a1a")
        self.ax.axis("off")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._measure_mode   = False
        self._measure_points: list = []
        self._measure_line   = None
        self._pan_start  = None
        self._xlim_start = None
        self._ylim_start = None

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("motion_notify_event",  self._on_motion)
        self.mpl_connect("button_release_event", self._on_release)

    def show_slice(self, slice_array, cell_mask=None, cilia_mask=None):
        self.ax.cla()
        self.ax.axis("off")
        if slice_array is not None:
            self.ax.imshow(slice_array, cmap="gray", interpolation="nearest")
        if cell_mask is not None:
            overlay = np.zeros((*cell_mask.shape, 4))
            overlay[cell_mask > 0] = [0.1, 0.4, 0.9, 0.35]
            self.ax.imshow(overlay, interpolation="nearest")
        if cilia_mask is not None:
            overlay = np.zeros((*cilia_mask.shape, 4))
            overlay[cilia_mask > 0] = [0.9, 0.1, 0.6, 0.6]
            self.ax.imshow(overlay, interpolation="nearest")
        self.fig.canvas.draw_idle()

    def set_measure_mode(self, enabled: bool):
        self._measure_mode   = enabled
        self._measure_points = []
        if self._measure_line:
            try:
                self._measure_line.remove()
            except Exception:
                pass
            self._measure_line = None
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.fig.canvas.draw_idle()

    def zoom(self, factor: float):
        """factor > 1 zooms in (shrinks visible range), < 1 zooms out."""
        ax = self.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xc = (xlim[0] + xlim[1]) / 2
        yc = (ylim[0] + ylim[1]) / 2
        new_xh = (xlim[1] - xlim[0]) / 2 / factor
        new_yh = (ylim[1] - ylim[0]) / 2 / factor
        ax.set_xlim(xc - new_xh, xc + new_xh)
        ax.set_ylim(yc - new_yh, yc + new_yh)
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._measure_mode:
            self._measure_points.append((event.xdata, event.ydata))
            if len(self._measure_points) == 2:
                p1, p2 = self._measure_points
                if self._measure_line:
                    try:
                        self._measure_line.remove()
                    except Exception:
                        pass
                self._measure_line, = self.ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]],
                    color="#ff0099", linewidth=2, linestyle="--", zorder=10
                )
                dist = ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5
                self.ax.text(
                    (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 - 6,
                    f"{dist:.1f} px", color="#ff0099",
                    fontsize=8, ha="center", zorder=11
                )
                self._measure_points = []
                self.fig.canvas.draw_idle()
        else:
            self._pan_start  = (event.xdata, event.ydata)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _on_motion(self, event):
        if (self._measure_mode or self._pan_start is None
                or event.inaxes != self.ax or event.xdata is None 
                or self._xlim_start is None or self._ylim_start is None):
            return
        dx = event.xdata - self._pan_start[0]
        dy = event.ydata - self._pan_start[1]
        self.ax.set_xlim(self._xlim_start[0] - dx, self._xlim_start[1] - dx)
        self.ax.set_ylim(self._ylim_start[0] - dy, self._ylim_start[1] - dy)
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        self._pan_start = None
        if not self._measure_mode:
            self.setCursor(Qt.CursorShape.ArrowCursor)


# ---------------------------------------------------------------------------
# OverviewPanel
# ---------------------------------------------------------------------------

class OverviewPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        lbl = QLabel("Z-Projection (max intensity)")
        lbl.setFont(QFont("Courier New", 9))
        lbl.setStyleSheet("color: #00aaff;")
        header.addWidget(lbl)
        header.addStretch()

        btn_style = """
            QPushButton {
                background: #2a2a2a; color: #e0e0e0;
                border: 1px solid #444; border-radius: 4px;
            }
            QPushButton:hover   { background: #3a3a3a; border-color: #00aaff; }
            QPushButton:pressed { background: #00aaff; color: #000; }
            QPushButton:checked { background: #00aaff; color: #000; border-color: #00aaff; }
        """
        self.btn_plus  = QPushButton("+")
        self.btn_minus = QPushButton("−")
        for btn in (self.btn_plus, self.btn_minus):
            btn.setFixedSize(28, 28)
            btn.setFont(QFont("Courier New", 12, QFont.Weight.Bold))
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)

        self.btn_plus.clicked.connect(self._on_plus)
        self.btn_minus.clicked.connect(self._on_minus)
        header.addWidget(self.btn_plus)
        header.addWidget(self.btn_minus)
        layout.addLayout(header)

        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)

        self.status_lbl = QLabel("No file loaded")
        self.status_lbl.setFont(QFont("Courier New", 8))
        self.status_lbl.setStyleSheet("color: #666;")
        layout.addWidget(self.status_lbl)

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
            self.status_lbl.setText("Delete mode — freehand draw around cells to remove")
        else:
            self.canvas.set_mode(MODE_NORMAL)
            self.status_lbl.setText("Normal mode")

    def reset_mode(self):
        self.btn_plus.setChecked(False)
        self.btn_minus.setChecked(False)
        self.canvas.set_mode(MODE_NORMAL)

    def load_demo(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 80, (256, 256), dtype=np.uint8)
        cell_positions = [(60, 80), (130, 50), (90, 150), (170, 120), (200, 60), (50, 200)]
        for (cy, cx) in cell_positions:
            yy, xx = np.ogrid[:256, :256]
            mask = (yy - cy)**2 + (xx - cx)**2 < 15**2
            img[mask] = np.clip(img[mask] + rng.integers(120, 200), 0, 255)
        self.canvas.show_image(img)
        markers = [
            {"x": cx, "y": cy, "num": i + 1, "accepted": True}
            for i, (cy, cx) in enumerate(cell_positions)
        ]
        self.canvas.set_markers(markers)
        self.status_lbl.setText(f"{len(markers)} cells detected")
        return markers, img


# ---------------------------------------------------------------------------
# ZViewerPanel
# ---------------------------------------------------------------------------

class ZViewerPanel(QWidget):
    measure_toggled = pyqtSignal(bool)

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
        self.btn_ruler.setToolTip("Measure mode — click two points")
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
        self.z_slider.setMaximum(20)
        self.z_slider.setValue(10)
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
        self.z_val = QLabel("10")
        self.z_val.setFont(QFont("Courier New", 8))
        self.z_val.setStyleSheet("color: #00aaff;")
        self.z_val.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.z_slider.valueChanged.connect(lambda v: self.z_val.setText(str(v)))
        slider_col.addWidget(z_top)
        slider_col.addWidget(self.z_slider, stretch=1)
        slider_col.addWidget(self.z_val)
        layout.addLayout(slider_col)

        self.btn_ruler.toggled.connect(
            lambda checked: (self.canvas.set_measure_mode(checked),
                             self.measure_toggled.emit(checked))
        )
        self.btn_zoomin.clicked.connect(lambda: self.canvas.zoom(1.3))
        self.btn_zoomout.clicked.connect(lambda: self.canvas.zoom(1 / 1.3))

        self._load_demo_slice()

    def _load_demo_slice(self):
        rng = np.random.default_rng(7)
        sl = rng.integers(0, 60, (128, 128), dtype=np.uint8)
        yy, xx = np.ogrid[:128, :128]
        cell = (yy - 64)**2 / 25**2 + (xx - 64)**2 / 20**2 < 1
        sl[cell] = np.clip(sl[cell] + 160, 0, 255)
        cell_mask = cell.astype(np.uint8)
        cilia = np.zeros((128, 128), dtype=np.uint8)
        for i in range(40):
            cy = int(45 + i * 0.6 + 4 * np.sin(i * 0.4))
            cx = int(68 + i)
            if 0 <= cy < 128 and 0 <= cx < 128:
                cilia[cy, cx] = 1
                if cy + 1 < 128:
                    cilia[cy + 1, cx] = 1
        self.canvas.show_slice(sl, cell_mask, cilia)


# ---------------------------------------------------------------------------
# DataTablePanel
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
        for i, w in enumerate([120, 45, 60, 60, 45, 70]):
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

    def load_demo_rows(self):
        for row in [
            ("demo.czi", 1, 60.0,  80.0,  10, 8.34),
            ("demo.czi", 2, 130.0, 50.0,   9, 6.71),
            ("demo.czi", 3, 90.0,  150.0, 11, 9.02),
            ("demo.czi", 4, 170.0, 120.0,  8, 7.45),
            ("demo.czi", 5, 200.0, 60.0,  12, 5.88),
            ("demo.czi", 6, 50.0,  200.0, 10, 10.13),
        ]:
            self.add_row(*row)

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
        self._apply_theme()
        self._build_layout()
        self._load_demo()

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

        h_bot = QSplitter(Qt.Orientation.Horizontal)
        self.table_panel = DataTablePanel()
        h_bot.addWidget(self._framed(self.table_panel))
        h_bot.setSizes([350, 750])

        v_split.addWidget(h_top)
        v_split.addWidget(h_bot)
        v_split.setSizes([480, 320])
        self.setCentralWidget(v_split)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.overview_panel.canvas.cell_clicked.connect(self._on_cell_clicked)
        self.overview_panel.canvas.marker_added.connect(self._on_marker_added)
        self.overview_panel.canvas.markers_deleted.connect(self._on_markers_deleted)
        self.zviewer_panel.measure_toggled.connect(
            lambda on: self.status.showMessage(
                "Measure mode ON — click two points on the cilia" if on
                else "Measure mode OFF"
            )
        )

    def _framed(self, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { border: 1px solid #2a2a2a; border-radius: 2px; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        return frame

    def _load_demo(self):
        self._markers, _ = self.overview_panel.load_demo()
        self.table_panel.load_demo_rows()
        self.status.showMessage(
            "Demo loaded  |  drag to pan  |  + rect-add  |  − lasso-delete  |  click marker to select"
        )

    def _on_cell_clicked(self, idx: int):
        if idx >= len(self._markers):
            return
        m = self._markers[idx]
        self.table_panel.table.selectRow(idx)
        self.status.showMessage(
            f"Cell {m['num']}  |  x={m['x']:.1f}  y={m['y']:.1f}  |  "
            "use Z-slider to step through slices"
        )

    def _on_marker_added(self, x: float, y: float):
        num = max((m["num"] for m in self._markers), default=0) + 1
        self._markers.append({"x": x, "y": y, "num": num, "accepted": True})
        self.overview_panel.canvas.set_markers(self._markers)
        self.table_panel.add_row("demo.czi", num, round(x, 1), round(y, 1), 0, "—")
        self.overview_panel.reset_mode()
        self.overview_panel.status_lbl.setText(
            f"{sum(1 for m in self._markers if m['accepted'])} cells detected"
        )
        self.status.showMessage(f"Cell {num} added at ({x:.1f}, {y:.1f})")

    def _on_markers_deleted(self, indices: list):
        for i in sorted(indices, reverse=True):
            self._markers.pop(i)
        for i, m in enumerate(self._markers):
            m["num"] = i + 1
        self.overview_panel.canvas.set_markers(self._markers)
        self.table_panel.remove_rows(indices)
        self.overview_panel.reset_mode()
        self.overview_panel.status_lbl.setText(
            f"{sum(1 for m in self._markers if m['accepted'])} cells detected"
        )
        self.status.showMessage(f"{len(indices)} cell(s) removed")


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