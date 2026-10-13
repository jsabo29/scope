# Scope

Desktop tool for finding **GAD1⁺ neurons that carry a primary cilium** in confocal
Z-stacks, and measuring each cilium's length in microns.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Opens `.czi` (Zeiss) and `.tif` / `.ome.tif`.

## Layout

| File | What it does |
|---|---|
| `run.py` | Entry point → `scope.app.launch()` |
| `scope/app.py` | Qt window, file loading, threading, display compositor |
| `scope/analysis.py` | The detection pipeline + its Qt dialog and worker |
| `scope/cilia_finder.py` | 3-D cilia detection (connected components with dim bridging) |
| `scope/analysis_backup.py` | Superseded per-slice 2-D pipeline, kept for reference |

## The window

- **Top left** — max-Z projection with cell markers. `+` drag a box to add a cell,
  `−` lasso to delete, drag to pan. Clicking a marker jumps the right panel to
  that cell's Z and frames a 30 µm window.
- **Top right** — Z-slice view. 📏 is the ruler; 🔍± zoom; drag pans.
- **Bottom** — the results table, with CSV import/export.
- **Channel** menu toggles channels and switches to colocalisation (pixelwise
  minimum) view. **Image → Process** gives per-channel brightness / contrast /
  floor. **Analyze → Detect** runs the pipeline.

### Measuring by hand

The ruler accumulates in 3-D. Trace the part of the cilium visible on the current
slice, move the Z slider, trace the next part — the segments join up and the
running total accounts for the axial jump using the file's Z spacing. Toggling
the ruler off and on starts a new measurement. Select a table row first and the
result is written to it.

If the file carries no Z spacing in its metadata, enter it in the detection
dialog; the status bar says so when axial travel cannot be converted.

## The pipeline (Analyze → Detect)

1. **Read & prepare** — each channel is read, normalised and reduced to its
   leanest form one at a time (nuclei → boolean masks, GAD1 and cilia → uint16).
   Cilia are deliberately *not* Gaussian-denoised; the blur erases them. They are
   normalised against the 99.99th percentile instead.
2. **Nucleus volume filter** — 3-D connected components, drop anything below the
   volume of the minimum nucleus diameter.
3. **Soma test** — a nucleus counts as GAD1⁺ when the median GAD1 brightness in
   the shell just outside it exceeds the median in the comparison shell beyond
   that by the differential %. Local, so it tolerates uneven illumination.
   Neighbouring nuclei are excluded from both zones.
4. **Cilia detection** — bright voxels are grouped into components; dim voxels
   adjacent to bright ones bridge small gaps, so a cilium broken by a dim stretch
   stays one object while background haze cannot chain together.
5. **Assignment** — largest cilium first, one per cell, matched by the shortest
   distance from the cilium to the cell's nucleus surface.
6. **Length** — 3-D skeleton, longest geodesic path through it, weighted by true
   voxel spacing.

Optionally saves a processed volume mirroring the original channels, with a
graded cilia channel for checking: dim = every detection, medium = the one
matched to a cell, bright = the skeleton actually measured.

### Tuning

Every threshold is a heuristic and is meant to be tuned on your data. The
terminal prints per-stage counts — nuclei surviving each filter, cilia detected
and matched, the distribution of lengths and anchor gaps — which is the fastest
way to see which stage is throwing your cells away. Two defaults worth knowing:

- **Cilium → cell matching** defaults to *proximal* (cilium to nucleus surface).
  The legacy *centroid* mode measures centre to centre, which grows with the
  cilium's own length and so discards long cilia for being long.
- **Cilia detection** defaults to the deterministic component labelling. The
  legacy flood fill is kept only to reproduce older runs; its result depends on
  voxel visit order.
