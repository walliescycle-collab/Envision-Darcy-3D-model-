# Envision groundwater — Streamlit

An interactive web version of the saturated PyGIMLi model in
[the original notebook](<Envision_3D_PyGIMLi_Groundwater v1.ipynb>).
The notebook is unchanged and remains independently runnable.

## Install and run

Use a dedicated environment with Python 3.11 or newer and a compatible PyGIMLi
compiled core. For example, in a Windows terminal with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Alternatively, use a Miniforge/Anaconda Prompt:

```text
conda create -n envision-web -c gimli -c conda-forge python=3.11 pygimli=1.6.1 pip
conda activate envision-web
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
Choose parameters and click **Run model**. The default mesh takes roughly a
minute on the development machine; finer meshes require more memory and time.
The initial page does not run a solve. Opening tabs, rotating plots and downloading
results do not solve again. Up to two configurations are cached in process memory;
the cache is lost when the server restarts.

## Features

- Editable reservoir heads, strip/fracture widths, conductivities, effective porosities and mesh preset.
- Turn on **Adjust hydraulic properties** in the sidebar to expose paired conductivity
  (m/s) and effective-porosity (fraction) controls for bedrock, aquiclude, deep sands,
  aquitard, compacted sands, loose sands/recharge strip, clay, fractures and boreholes.
  The strip shares both loose-sand properties. Porosity must be greater than zero
  and at most 1; it changes pore velocity (`q/n`), not steady head or Darcy flux.
  The open-water borehole default is 1.0. Submit edits with
  **Run model** before hiding them. Hiding controls retains the last submitted
  values; it does not reset them or trigger a solve. Casing K and porosity remain
  exactly zero. All selected properties are included in JSON, VTU and NPZ exports.
- Browser-based, rotatable Plotly geology/head/path views; no desktop VTK window,
  Trame server, Jupyter kernel or OpenGL screenshot renderer is used.
- Actual material sections, head/flux direction and pressure diagnostics.
- All 11 reservoir budgets, all nine borehole balances and hydraulic checks.
- ZIP downloads containing the full-resolution VTU, NPZ, JSON and sections PNG.
  Browser geometry/path simplification does not alter the downloadable solution.
- Per-session displayed results and isolated temporary exports. The existing
  notebook's `envision_outputs` directory is never overwritten by the app.

The model retains the 100 × 20 × 60 m geology, 1 m square borehole interiors,
0.5 m zero-K casing, bottom/top-only openings, bedrock fracture connections, and
rightmost full-depth conductive strip. The strip always uses loose-sand K.
Boreholes 1–3 connect to recharge; 4–9 connect to the spring/river reservoir through
ideal external plumbing. Only boreholes 1, 7 and 9 reach bedrock.

Controls are submitted together. Changed controls are not applied until **Run
model** is clicked. If a run fails validation, the app reports the failure and
retains any earlier successful result, clearly marking it as stale when applicable.
Parameters are included in every downloaded summary.

## Numerical and physical scope

[groundwater_model.py](groundwater_model.py) contains a standalone extraction of
the notebook equations and checks; it does **not** execute notebook cells.
[groundwater_views.py](groundwater_views.py) prepares browser figures and exports.
[streamlit_app.py](streamlit_app.py) handles controls, caching and presentation.
Native PyGIMLi/VTK calculations are serialized across sessions to avoid concurrent
access to process-global native state. Expensive results are not stored globally
as mutable VTK objects.

This is a conceptual, steady, fully saturated Darcy model, not a calibrated
prediction, unconfined solution, discrete-fracture formulation or real pipe-flow
law. Negative pressure is retained; the red zero-pressure contour is not a solved
water table. Fracture bands have finite model width, not measured apertures.
Horizontal bands follow bedrock slope. Well-bottom faces follow local mesh slopes
while matching the specified centreline elevation. Casing-only heads are NaN by
design; conductivity and flux there are exactly zero.

The river shares the spring/outlet reservoir rather than having its own riverbed
exchange law. Lake, Spring 2 and other illustrative surface objects are not hydraulic
boundaries. The browser views prioritize the solved materials and boreholes rather
than reproducing every decorative notebook object.

Geometry/connectivity, positive volumes, prescribed heads, independent stiffness
and gradients, total conservation, casing flux, all borehole balances and the
requested flow signs are checked before publishing a run. Unusable geometry,
nonfinite values or failed budgets raise explicit errors. Mesh convergence and
field calibration remain necessary for scientific use.

## Tests

Run in the same environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The integration tests run a full default solve, compare it with the validated
notebook's exported default result, verify figure/export content, and exercise the
Streamlit controls/run/failure/stale-result paths with Streamlit AppTest.
Keep the original default files in `envision_outputs` for the regression comparison.
