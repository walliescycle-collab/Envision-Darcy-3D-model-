"""Run with: python -m streamlit run streamlit_app.py"""
from dataclasses import asdict
import logging
import threading

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning
import streamlit as st

from groundwater_model import HYDRAULIC_MATERIALS, ModelConfig, ModelValidationError, run_model
from groundwater_views import AppResults, prepare_results

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="Envision Groundwater", layout="wide")


@st.cache_resource
def solver_lock() -> threading.Lock:
    # PyGIMLi/VTK have process-global native state; serialize work across sessions.
    return threading.Lock()


@st.cache_data(max_entries=2, show_spinner=False)
def calculate(config: ModelConfig) -> AppResults:
    with solver_lock():
        model = run_model(config, progress=st.write)
        st.write("Preparing browser figures and downloads")
        return prepare_results(model)


def conductivity(label: str, value: float, key: str) -> float:
    return float(st.number_input(
        label, min_value=1e-12, max_value=1.0, value=value, format="%.2e", key=key,
        help="Illustrative scalar Darcy conductivity in metres per second.",
    ))


def main() -> None:
    st.title("Envision 3D Groundwater")
    st.caption("Steady saturated Darcy flow | 100 × 20 × 60 m | PyGIMLi finite elements")
    st.warning(
        "Conceptual model, not a calibrated prediction. Elevated borehole tops are connected "
        "to ideal external reservoirs. Fractures and pipes are finite-width Darcy surrogates; "
        "negative pressure does not represent a solved unsaturated zone."
    )
    with st.sidebar:
        st.header("Model controls")
        st.caption("Edit parameters, then click Run model. Downloads always match the displayed run.")
        show_hydraulics = st.toggle("Adjust hydraulic properties", value=False)
        properties = st.session_state.setdefault(
            "hydraulic_properties",
            dict(rock_k=1e-9, fracture_k=2e-4, sand_k=1e-4, pipe_k=1e-2),
        )
        defaults = asdict(ModelConfig())
        properties.setdefault("bedrock_k", properties.get("rock_k", defaults["bedrock_k"]))
        properties.setdefault("loose_sand_k", properties.get("sand_k", defaults["loose_sand_k"]))
        properties = {
            field: properties.get(field, defaults[field])
            for _, k_field, porosity_field in HYDRAULIC_MATERIALS
            for field in (k_field, porosity_field)
        }
        with st.form("model_parameters"):
            spring = st.number_input("Spring / river total head (m)", value=40.0, step=1.0)
            difference = st.number_input("Recharge head above outlet (m)", min_value=0.1,
                                         max_value=100.0, value=10.0, step=0.5)
            strip = st.slider("Right-edge strip width (m)", 0.5, 4.0, 2.0, 0.5)
            fracture = st.select_slider("Fracture-band thickness (m)",
                                         options=[0.25, 0.5, 0.75], value=0.5)
            if show_hydraulics:
                st.subheader("Conductivity and effective porosity")
                st.caption(
                    "K is in m/s; porosity is a fraction (0 < n <= 1). "
                    "Porosity rescales pore velocity q/n, not steady head or Darcy flux. "
                    "The recharge strip shares loose-sand properties."
                )
                for name, k_field, porosity_field in HYDRAULIC_MATERIALS:
                    with st.expander(name):
                        properties[k_field] = conductivity(
                            f"{name} K (m/s)", properties[k_field], k_field,
                        )
                        properties[porosity_field] = float(st.number_input(
                            f"{name} effective porosity",
                            min_value=0.000001, max_value=1.0,
                            value=properties[porosity_field], format="%.6f",
                            key=porosity_field,
                            help="Dimensionless water-conducting volume fraction; 0.30 means 30%.",
                        ))
                st.caption("Impermeable casing: K = 0 and porosity = 0 (fixed).")
            else:
                st.caption("Hydraulic controls hidden; defaults or last submitted values remain active.")
            resolution = st.selectbox(
                "Background mesh", ["Notebook default (50 × 6)", "Coarse (30 × 4)", "Fine (70 × 8)"],
            )
            st.caption("Feature edges remain explicit at every resolution. Fine runs use more memory.")
            submitted = st.form_submit_button("Run model", type="primary", width="stretch")
        if submitted:
            st.session_state["hydraulic_properties"] = properties
        st.caption("Click Run model to submit hydraulic edits before hiding the controls.")
        st.caption("No desktop windows, Trame service or Jupyter kernel required.")
    nx, ny = {
        "Notebook default (50 × 6)": (50, 6),
        "Coarse (30 × 4)": (30, 4),
        "Fine (70 × 8)": (70, 8),
    }[resolution]
    config = ModelConfig(
        spring_head=float(spring), head_difference=float(difference),
        strip_width=float(strip), fracture_width=float(fracture),
        nx=nx, ny=ny,
        **properties,
    )
    if submitted:
        with st.status("Building and validating model...", expanded=True) as status:
            try:
                result = calculate(config)
            except (ModelValidationError, np.linalg.LinAlgError, MatrixRankWarning,
                    RuntimeError, OSError) as error:
                LOGGER.exception("Groundwater model run failed")
                status.update(label="Run failed; no new results published", state="error")
                st.error(str(error))
                st.info("Review parameters or choose the notebook defaults. Any earlier result below is unchanged.")
            else:
                st.session_state["results"] = result
                status.update(label="Model solved; hydraulic checks passed", state="complete", expanded=False)
    result = st.session_state.get("results")
    if result is None:
        st.info("Choose parameters in the sidebar and click Run model. The default run can take about a minute.")
        with st.expander("What does this model include?", expanded=True):
            st.markdown(
                "- Six geological layers and a local clay lens.\n"
                "- A full-depth right-edge recharge strip with loose-sand conductivity.\n"
                "- A connected fracture grid within low-conductivity bedrock.\n"
                "- Nine 1 m square boreholes at y=10 m with 0.5 m impermeable casing.\n"
                "- Boreholes 1–3 feed aquifers; boreholes 4–9 discharge through ideal plumbing.\n"
                "- Independent stiffness, water-budget, connectivity and pipe-balance checks."
            )
        return
    summary = result.summary
    if asdict(config) != summary["config"]:
        st.warning("Controls differ from the displayed run. Click Run model to apply them; current downloads retain the previous run.")
    for message in summary["warnings"]:
        st.warning(message)
    columns = st.columns(4)
    columns[0].metric("Recharge (m³/day)", f'{summary["recharge_inflow_m3_day"]:.3f}')
    columns[1].metric("Spring / river outflow (m³/day)", f'{summary["spring_outflow_m3_day"]:.3f}')
    columns[2].metric("Relative budget imbalance", f'{summary["relative_budget_imbalance"]:.2e}')
    columns[3].metric("Tetrahedra", f'{summary["tetrahedra"]:,}')
    geology, flow, sections, budget, downloads = st.tabs(
        ["3D geology", "Head & flow", "Sections", "Budgets & checks", "Downloads & assumptions"]
    )
    with geology:
        st.caption("Drag to rotate; scroll to zoom. Use the legend to toggle materials. Rendering simplification affects display only.")
        st.plotly_chart(result.geometry, width="stretch", key="geometry")
    with flow:
        st.plotly_chart(result.flow, width="stretch", key="flow")
        st.caption("Cell-to-point averaged streamlines are qualitative, not particle travel times or connectivity evidence. Borehole hover shows verified budget direction.")
    with sections:
        st.image(result.sections_png, caption="Actual material sections, hydraulic head and pressure diagnostic.")
    with budget:
        st.subheader("Reservoir budgets")
        st.caption("Positive = inflow to the model; negative = outflow. Totals include surface and borehole connections.")
        st.dataframe(list(summary["boundary_budgets"].values()), hide_index=True, width="stretch")
        st.subheader("Borehole top / bottom budgets")
        st.dataframe(
            [dict(borehole=label, **values) for label, values in summary["borehole_budgets"].items()],
            hide_index=True, width="stretch",
        )
        st.subheader("Passed checks")
        st.json(summary["checks"])
        st.write(f'Free-node residual: {summary["free_node_relative_residual"]:.3e}')
        st.write(f'Active nodes with negative pressure: {summary["negative_pressure_node_fraction"]:.1%}')
    with downloads:
        st.download_button("Download complete results ZIP", result.archive,
                           file_name="envision_results.zip", mime="application/zip")
        st.download_button("Download run summary JSON", result.summary_json,
                           file_name="run_summary.json", mime="application/json")
        st.download_button("Download sections PNG", result.sections_png,
                           file_name="03_sections.png", mime="image/png")
        st.subheader("Parameters for this result")
        st.json(summary["config"])
        st.markdown(
            "**ZIP contents:** full-resolution VTU mesh, compressed NPZ arrays, JSON properties/"
            "budgets and sections PNG. No shared output files are overwritten.\n\n"
            "**Interpretation:** 50 m recharge and 40 m outlet heads are defaults, not recharge "
            "rates. Only bottom/top borehole faces exchange water; intermediate aquifers cannot "
            "leak through casing. Boreholes 1, 7 and 9 reach bedrock. The river shares the "
            "outlet reservoir; there is no separate riverbed exchange law. Lake and Spring 2 "
            "remain visual/conceptual features, not flow boundaries.\n\n"
            "**Resolution:** borehole bottoms follow local mesh slopes with exact centreline "
            "elevations. Horizontal fracture bands follow the bedrock slope. Conductivity, "
            "fracture width and pipe-flow assumptions need calibration and mesh-convergence studies. "
            "Casing-only node heads are undefined (NaN); their flux is exactly zero."
        )


if __name__ == "__main__":
    main()
