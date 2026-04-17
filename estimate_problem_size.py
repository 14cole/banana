"""
Estimate the number of unknowns (DOFs) and peak memory for a given
geometry snapshot without actually solving.

Usage:
    python estimate_problem_size.py my_geometry.grim 3.0
    # or
    from estimate_problem_size import estimate_problem_size
    est = estimate_problem_size(snapshot, frequency_ghz=3.0, polarization='TE')
    print(est)

Returns a dict with:
    'panels'           — total number of boundary elements
    'nodes'            — number of unique mesh nodes (usually panels+1 per open
                         strip, panels per closed body)
    'dofs'             — solver unknowns (depends on formulation: = nodes for
                         sheet / Robin-BIE / MFIE paths; = 2*nodes for
                         coupled-trace and transmission formulations)
    'formulation'      — which formulation the dispatcher will pick
    'dense_matrix_gb'  — peak memory for dense LU (system matrix + factors +
                         region operators)
    'recommended'      — solver_method suggestion based on DOF count
"""
from __future__ import annotations
import math
import os
import sys
from typing import Dict, Any

# Solver constants, kept in sync with rcs_solver.py defaults.
C0 = 299_792_458.0
DEFAULT_PANELS_PER_WAVELENGTH = 20
GMRES_NODE_THRESHOLD = 3000       # from rcs_solver.py
DENSE_MEMORY_WARN_GB = 8.0        # rcs_solver warns above this
DENSE_MEMORY_HARD_GB = 32.0       # rcs_solver refuses above this


def _panel_count_for_primitive(primitive_len_m: float,
                               n_property: int,
                               wavelength_m: float) -> int:
    """Replicate rcs_solver._panel_count_from_n without importing the solver."""
    if primitive_len_m <= 1e-12:
        return 1
    if n_property > 0:
        return max(1, n_property)
    if n_property < 0:
        n_per_wave = max(1, abs(n_property))
        target = max(wavelength_m / n_per_wave, primitive_len_m / 2000.0)
        return max(1, int(math.ceil(primitive_len_m / target)))
    # n_property == 0: use the default density
    target = wavelength_m / DEFAULT_PANELS_PER_WAVELENGTH
    return max(1, int(math.ceil(primitive_len_m / target)))


def _primitive_length_m(pt_pair: Dict[str, Any], unit_scale: float,
                        ang_deg: float = 0.0) -> float:
    """Euclidean length (or arc length if ang_deg != 0) in meters."""
    dx = (pt_pair['x2'] - pt_pair['x1']) * unit_scale
    dy = (pt_pair['y2'] - pt_pair['y1']) * unit_scale
    chord = math.hypot(dx, dy)
    if abs(ang_deg) < 1e-9:
        return chord
    # Arc primitives: arc_length = chord * ang_rad / (2 sin(ang/2))
    ang_rad = math.radians(ang_deg)
    half = 0.5 * abs(ang_rad)
    if math.sin(half) < 1e-12:
        return chord
    return chord * abs(ang_rad) / (2.0 * math.sin(half))


def _count_panels_and_closed(snapshot: Dict[str, Any],
                             wavelength_m: float,
                             unit_scale: float = 1.0) -> tuple[int, bool, set]:
    """
    Return (total_panel_count, any_closed_body, seg_types_present).
    """
    total = 0
    closed_bodies = False
    seg_types_present: set[int] = set()
    for seg in snapshot.get("segments", []):
        props = seg.get("properties", [])
        seg_type = int(props[0]) if len(props) > 0 else 2
        n_prop = int(props[1]) if len(props) > 1 else 0
        seg_types_present.add(seg_type)
        # Is this segment closed?  Check if first and last point coincide.
        pairs = seg.get("point_pairs", [])
        if len(pairs) >= 2:
            p_first = (pairs[0]['x1'], pairs[0]['y1'])
            p_last = (pairs[-1]['x2'], pairs[-1]['y2'])
            if abs(p_first[0] - p_last[0]) < 1e-9 and abs(p_first[1] - p_last[1]) < 1e-9:
                closed_bodies = True
        for pair in pairs:
            ang_deg = float(pair.get("ang_deg", 0.0))
            prim_len = _primitive_length_m(pair, unit_scale, ang_deg)
            total += _panel_count_for_primitive(prim_len, n_prop, wavelength_m)
    return total, closed_bodies, seg_types_present


def _predict_formulation(seg_types: set[int],
                         ibc_flags_used: set[int],
                         pol: str) -> str:
    """
    Mirror rcs_solver's dispatcher logic for the common cases.
    """
    has_sheet = 1 in seg_types
    has_pec_or_ibc = 2 in seg_types
    has_dielectric = 3 in seg_types or 5 in seg_types
    has_coated_pec = 4 in seg_types
    if has_sheet and has_pec_or_ibc and not has_dielectric and not has_coated_pec:
        return "mixed sheet+PEC (unified SLP/DLP)"
    if has_sheet and not has_pec_or_ibc and not has_dielectric:
        return f"dedicated sheet BIE ({pol})"
    if has_sheet:
        return "REJECTED (sheet + dielectric/coated not supported)"
    if has_dielectric and not has_pec_or_ibc and not has_coated_pec:
        return "single-dielectric two-density indirect"
    if has_coated_pec or (has_dielectric and has_pec_or_ibc):
        return "multi-region indirect (layered)"
    if has_pec_or_ibc and pol == "TM":
        return "TM Robin MFIE"
    if has_pec_or_ibc and pol == "TE":
        return "TE Robin-BIE (SLP)"
    return "coupled trace (general)"


def _dofs_per_node_for(formulation: str) -> int:
    """Most solver paths have 1 DOF/node; coupled and multi-region have 2."""
    if "coupled" in formulation or "multi-region" in formulation \
       or "two-density" in formulation or "dielectric" in formulation:
        return 2
    return 1


def estimate_problem_size(snapshot: Dict[str, Any],
                          frequency_ghz: float,
                          polarization: str = "TE",
                          geometry_units: str = "meters") -> Dict[str, Any]:
    """
    Estimate panel/node/DOF counts and dense-solve memory for a geometry
    snapshot, without running the solver.
    """
    pol = polarization.strip().upper()
    # meters per input unit
    unit_scale = {"meters": 1.0, "inches": 0.0254, "mm": 1e-3, "cm": 1e-2,
                  "feet": 0.3048}.get(geometry_units.strip().lower(), 1.0)

    wavelength_m = C0 / (frequency_ghz * 1e9)

    total_panels, has_closed, seg_types = _count_panels_and_closed(
        snapshot, wavelength_m, unit_scale)

    # Node count approximation.
    # For continuous linear Galerkin:
    #   - Closed contours: nodes == panels (same node at start and end)
    #   - Open strips:     nodes == panels + 1 (extra node at free end)
    # Signature-split effects can add a few extra nodes at material junctions
    # but the total is typically within 5-10% of this estimate.
    n_segments = len(snapshot.get("segments", []))
    if has_closed:
        approx_nodes = total_panels   # closed bodies dominate
    else:
        approx_nodes = total_panels + n_segments  # each open segment adds 1

    ibc_flags = {row[0] for row in snapshot.get("ibcs", []) if row}
    formulation = _predict_formulation(seg_types, ibc_flags, pol)
    dofs_per_node = _dofs_per_node_for(formulation)
    dofs = approx_nodes * dofs_per_node

    # Dense memory estimate (matches rcs_solver._estimate_memory_gb).
    bytes_per_complex = 16
    sys_size = 2 * approx_nodes     # always 2N in _estimate_memory_gb
    sys_bytes = 2 * sys_size * sys_size * bytes_per_complex
    # Assume 1 region and no CFIE for a conservative lower-bound estimate;
    # bump to 2 regions if dielectric present.
    n_regions = 2 if (3 in seg_types or 4 in seg_types or 5 in seg_types) else 1
    use_cfie_est = has_closed and 2 in seg_types
    ops_per_region = 4 if not use_cfie_est else 8
    region_bytes = n_regions * ops_per_region * approx_nodes * approx_nodes * bytes_per_complex
    misc_bytes = 4 * sys_size * bytes_per_complex * 1000
    total_bytes = sys_bytes + region_bytes + misc_bytes
    dense_gb = total_bytes / (1024 ** 3)

    # Recommend a solver method based on DOF count.
    if dofs < GMRES_NODE_THRESHOLD:
        recommended = "direct (dense LU)"
    elif dense_gb < DENSE_MEMORY_WARN_GB and dofs < 15000:
        recommended = "auto (GMRES above 3000 DOFs)"
    elif dense_gb < DENSE_MEMORY_HARD_GB:
        recommended = "gmres or fmm (large-memory solve)"
    else:
        recommended = "fmm (dense A would not fit)"

    return {
        "frequency_ghz": frequency_ghz,
        "wavelength_m": wavelength_m,
        "panels": total_panels,
        "approx_nodes": approx_nodes,
        "dofs_per_node": dofs_per_node,
        "dofs": dofs,
        "formulation": formulation,
        "seg_types": sorted(seg_types),
        "dense_matrix_gb": round(dense_gb, 3),
        "dense_warn_gb": DENSE_MEMORY_WARN_GB,
        "dense_hard_gb": DENSE_MEMORY_HARD_GB,
        "recommended_solver_method": recommended,
    }


def format_estimate(est: Dict[str, Any]) -> str:
    lines = [
        f"Frequency:               {est['frequency_ghz']:.3f} GHz  "
        f"(lambda = {est['wavelength_m']*100:.2f} cm)",
        f"Panels (boundary elems): {est['panels']}",
        f"Nodes (approximate):     {est['approx_nodes']}",
        f"DOFs per node:           {est['dofs_per_node']}",
        f"Total DOFs (unknowns):   {est['dofs']}",
        f"Segment types present:   {est['seg_types']}",
        f"Predicted formulation:   {est['formulation']}",
        f"Dense-solve memory:      {est['dense_matrix_gb']} GB  "
        f"(warn > {est['dense_warn_gb']}, refuse > {est['dense_hard_gb']})",
        f"Suggested solver_method: {est['recommended_solver_method']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python estimate_problem_size.py <snapshot.json|.grim> <freq_ghz>")
        sys.exit(1)
    import json
    path = sys.argv[1]
    freq = float(sys.argv[2])
    with open(path) as f:
        snap = json.load(f)
    print(format_estimate(estimate_problem_size(snap, freq)))
