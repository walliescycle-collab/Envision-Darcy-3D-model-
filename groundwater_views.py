"""Browser figures and isolated, downloadable exports; no desktop VTK windows."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import plotly.graph_objects as go
import pyvista as pv

from groundwater_model import ModelResult, horizons


@dataclass
class AppResults:
    summary: dict
    geometry: go.Figure
    flow: go.Figure
    sections_png: bytes
    archive: bytes
    summary_json: str


def _mesh_trace(
    surface: pv.PolyData,
    name: str,
    color: str | None = None,
    opacity: float = 1.0,
    show_head: bool = False,
) -> go.Mesh3d:
    triangles = surface.triangulate()
    faces = triangles.faces.reshape(-1, 4)[:, 1:]
    options = {}
    if show_head:
        options = dict(
            intensity=np.asarray(triangles["Head_m"]),
            colorscale="Viridis",
            colorbar=dict(title="Head (m)"),
            showscale=True,
        )
    else:
        options = dict(color=color)
    return go.Mesh3d(
        x=triangles.points[:, 0], y=triangles.points[:, 1], z=triangles.points[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, opacity=opacity, showlegend=True,
        hovertemplate=name + "<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f} m<extra></extra>",
        **options,
    )


def _decorate(figure: go.Figure, result: ModelResult, title: str) -> None:
    geometry = result.summary["geometry"]
    for well in geometry["wells"]:
        x = well["x_m"]
        label = well["label"]
        bottom = well["bottom_centre_m"]
        top = float(horizons(x)[-1])
        flow = result.summary["borehole_budgets"][label]["top_inflow_m3_day"]
        direction = "downward recharge" if flow > 0 else "upward discharge"
        figure.add_trace(go.Scatter3d(
            x=[x, x], y=[geometry["well_y_m"]]*2, z=[bottom, top],
            mode="lines+text", text=["", label], textposition="top center",
            line=dict(color="#008f7a" if flow > 0 else "#00aee8", width=6),
            name=f"Borehole {label}", showlegend=False,
            hovertemplate=f"Borehole {label}: {direction}<br>Top inflow {flow:.3f} m3/day<extra></extra>",
        ))
    figure.update_layout(
        title=title, height=660, margin=dict(l=0, r=0, b=0, t=45),
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="Elevation (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=-2.0, z=1.1)),
        ),
        legend=dict(itemsizing="constant"), uirevision="envision-camera",
    )


def geometry_figure(result: ModelResult) -> go.Figure:
    grid = result.grid
    materials = result.summary["materials"]
    well_y = result.summary["geometry"]["well_y_m"]
    figure = go.Figure()
    cut = grid.slice(normal="y", origin=(50, well_y + 2, 30))
    for marker in range(1, 8):
        selected = cut.extract_cells(np.asarray(cut["Material"]) == marker)
        if selected.n_cells:
            material = materials[marker]
            figure.add_trace(_mesh_trace(
                selected.extract_surface(algorithm="dataset_surface"),
                material["name"], material["color"],
            ))
    for marker in [8, 9, 10, 11]:
        selected = grid.extract_cells(np.asarray(grid["Material"]) == marker)
        if marker in [10, 11]:
            selected = selected.clip(normal="y", origin=(0, well_y, 0), invert=False)
        surface = selected.extract_surface(algorithm="dataset_surface").triangulate().clean()
        surface.clear_data()
        if surface.n_cells > 25000:
            surface = surface.decimate_pro(
                reduction=0.8, preserve_topology=True, boundary_vertex_deletion=False,
            )
        material = materials[marker]
        figure.add_trace(_mesh_trace(
            surface, material["name"], material["color"], opacity=0.2 if marker == 8 else 1.0,
        ))
    _decorate(figure, result, "Resolved geology, fracture grid and cased boreholes")
    return figure


def flow_figure(result: ModelResult, flowgrid: pv.UnstructuredGrid) -> go.Figure:
    figure = go.Figure()
    y = result.summary["geometry"]["well_y_m"]
    section = flowgrid.slice(normal="y", origin=(50, y, 30))
    figure.add_trace(_mesh_trace(section, "Central head section", show_head=True))
    geometry = result.summary["geometry"]
    seeds = []
    for x in np.linspace(20, 95, 9):
        hs = horizons(x)
        for yy in [5, 10, 15]:
            for layer in [0, 2, 4, 5]:
                seeds.append([x, yy, (hs[layer] + hs[layer + 1]) / 2])
    for well in geometry["wells"]:
        x = well["x_m"]
        seeds.append([x, y, (well["bottom_centre_m"] + float(horizons(x)[-1])) / 2])
    for x in geometry["fracture_x_m"]:
        for fraction in geometry["fracture_horizontal_fractions"]:
            seeds.append([x, 5, fraction*float(horizons(x)[1])])
    for z in np.linspace(1, 59, 12):
        seeds.append([np.mean(geometry["strip_x_m"]), y, z])
    paths = flowgrid.streamlines_from_source(
        pv.PolyData(np.asarray(seeds)), vectors="Darcy_flux_m_s",
        integration_direction="both", max_steps=3000, initial_step_length=0.1,
        terminal_speed=1e-16, compute_vorticity=False,
    )
    if paths.n_cells:
        coordinates = []
        offset = 0
        while offset < len(paths.lines):
            count = paths.lines[offset]
            point_ids = paths.lines[offset+1:offset+1+count]
            # Limit browser payload only; retain every numerical cell in the downloads.
            step = max(1, int(count)//250)
            reduced = np.unique(np.r_[np.arange(0, count, step), count-1])
            coordinates.extend(paths.points[point_ids[reduced]].tolist())
            coordinates.append([None, None, None])
            offset += int(count)+1
        xyz = list(zip(*coordinates))
        figure.add_trace(go.Scatter3d(
            x=xyz[0], y=xyz[1], z=xyz[2], mode="lines", name="Qualitative flow paths",
            line=dict(color="#e4572e", width=2), hoverinfo="skip",
        ))
    _decorate(figure, result, "Hydraulic head and qualitative flow paths")
    return figure


def sections_png(result: ModelResult, flowgrid: pv.UnstructuredGrid) -> bytes:
    materials = result.summary["materials"]
    colors = [m["color"] for m in materials.values()]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, 12), 11)
    fig = Figure(figsize=(14, 18), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(4, 1, sharex=True)
    for ax, y in zip(axes[:2], [8.0, 10.0]):
        section = result.grid.slice(normal="y", origin=(50, y, 30)).triangulate()
        ax.tripcolor(
            section.points[:, 0], section.points[:, 2],
            section.faces.reshape(-1, 4)[:, 1:],
            facecolors=np.asarray(section["Material"]), cmap=cmap, norm=norm,
        )
        ax.set_title(
            f"Fracture grid and bedrock at y={y:g} m" if y == 8
            else "Boreholes and transverse fracture sheet at y=10 m"
        )
    axes[0].legend(
        handles=[Patch(color=m["color"], label=m["name"]) for m in materials.values()],
        loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8,
    )
    xs = np.linspace(0, 100, 400)
    zs = np.linspace(0, 60, 240)
    x, z = np.meshgrid(xs, zs)
    probe = pv.PolyData(np.c_[x.ravel(), np.full(x.size, 10.0), z.ravel()]).sample(flowgrid)
    valid = np.asarray(probe["vtkValidPointMask"]).reshape(x.shape).astype(bool)
    head = np.asarray(probe["Head_m"]).reshape(x.shape).copy()
    pressure = np.asarray(probe["Pressure_head_m"]).reshape(x.shape).copy()
    flux = np.asarray(probe["Darcy_flux_m_s"]).reshape(x.shape+(3,)).copy()
    head[~valid] = np.nan
    pressure[~valid] = np.nan
    flux[~valid] = np.nan
    levels = np.linspace(result.config.spring_head,
                         result.config.spring_head+result.config.head_difference, 21)
    image = axes[2].contourf(x, z, head, levels=levels, cmap="viridis", extend="both")
    fig.colorbar(image, ax=axes[2], label="Total head (m)")
    s = (slice(5, None, 18), slice(5, None, 20))
    qx, qz = flux[s][..., 0], flux[s][..., 2]
    speed = np.maximum(np.hypot(qx, qz), 1e-30)
    axes[2].quiver(x[s], z[s], qx/speed, qz/speed, color="white", scale=40, width=0.0025)
    axes[2].set_title("Head and normalised Darcy-flux direction")
    limit = float(np.nanmax(np.abs(pressure)))
    image = axes[3].pcolormesh(x, z, pressure, cmap="RdBu", vmin=-limit,
                               vmax=limit, shading="auto")
    fig.colorbar(image, ax=axes[3], label="Pressure head h-z (m)")
    if np.nanmin(pressure) < 0 < np.nanmax(pressure):
        axes[3].contour(x, z, pressure, levels=[0], colors="red", linewidths=1.5)
    axes[3].set_title("Red zero-pressure contour is NOT a solved water table")
    for ax in axes:
        for boundary in horizons(xs)[1:]:
            ax.plot(xs, boundary, color="#222", lw=0.5, alpha=0.5)
        ax.set(ylabel="Elevation (m)", aspect="equal", xlim=(0, 100), ylim=(0, 60))
    axes[-1].set_xlabel("Distance x (m)")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    fig.clear()
    return buffer.getvalue()


def export_archive(result: ModelResult, sections: bytes, summary_json: str) -> bytes:
    grid = result.grid
    arrays = io.BytesIO()
    np.savez_compressed(
        arrays, points=grid.points, tetrahedra=grid.cells.reshape(-1, 5)[:, 1:],
        material=grid["Material"], borehole_id=grid["Borehole_ID"],
        hydraulically_active_cells=grid["Hydraulically_active"].astype(bool),
        hydraulically_active_nodes=result.active_nodes,
        head_m=grid["Head_m"], pressure_head_m=grid["Pressure_head_m"],
        conductivity_m_s=grid["K_m_s"], effective_porosity=grid["Effective_porosity"],
        darcy_flux_m_s=grid["Darcy_flux_m_s"], pore_velocity_m_s=grid["Pore_velocity_m_s"],
    )
    archive = io.BytesIO()
    with TemporaryDirectory(prefix="envision-streamlit-") as directory:
        mesh_path = Path(directory) / "envision_flow.vtu"
        grid.save(mesh_path)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
            output.write(mesh_path, arcname=mesh_path.name)
            output.writestr("envision_solution.npz", arrays.getvalue())
            output.writestr("run_summary.json", summary_json)
            output.writestr("03_sections.png", sections)
    return archive.getvalue()


def prepare_results(result: ModelResult) -> AppResults:
    active = result.grid.extract_cells(result.grid["Hydraulically_active"].astype(bool))
    flowgrid = active.cell_data_to_point_data(pass_cell_data=True)
    sections = sections_png(result, flowgrid)
    summary_json = json.dumps(result.summary, indent=2, allow_nan=False)
    return AppResults(
        summary=result.summary,
        geometry=geometry_figure(result),
        flow=flow_figure(result, flowgrid),
        sections_png=sections,
        archive=export_archive(result, sections, summary_json),
        summary_json=summary_json,
    )
