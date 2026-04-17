"""
Estimate the number of unknowns (DOFs) and peak memory for a given
geometry (.geo text file or snapshot dict) without actually solving.

Usage:
    # From a .geo file:
    python estimate_problem_size.py my_geometry.geo 3.0

    # From JSON (legacy):
    python estimate_problem_size.py my_geometry.json 3.0

    # From Python:
    from estimate_problem_size import estimate_problem_size, load_geo_file
    snap = load_geo_file("my_geometry.geo")
    est = estimate_problem_size(snap, frequency_ghz=3.0, polarization='TE')
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
from typing import Any, Dict, List, Tuple

# Solver constants, kept in sync with rcs_solver.py defaults.
C0 = 299_792_458.0
DEFAULT_PANELS_PER_WAVELENGTH = 20
GMRES_NODE_THRESHOLD = 3000       # from rcs_solver.py
DENSE_MEMORY_WARN_GB = 8.0        # rcs_solver warns above this
DENSE_MEMORY_HARD_GB = 32.0       # rcs_solver refuses above this


# ─── .geo file reader (inlined from geometry_io.py so the script is standalone) ──

def load_geo_file(path: str) -> Dict[str, Any]:
    """Parse a .geo text file and return a snapshot dict.

    Format (matches geometry_io.parse_geometry):
        Title: optional title
        Segment: <name> <type?>
        properties: t1 t2 t3 t4 t5 t6
        x1 y1 x2 y2
        ... more point pairs ...
        Segment: ...
        IBCS:
        flag  R  X          (or: flag taper <kind> R1 X1 R2 X2)
        ...
        Dielectrics:
        ...

    Comments start with '#'.  Blank lines are ignored.
    """
    with open(path) as f:
        text = f.read()

    lines = [ln.strip() for ln in text.splitlines()]
    title = "Geometry"
    segments: List[Dict[str, Any]] = []
    ibcs_entries: List[List[str]] = []
    dielectric_entries: List[List[str]] = []

    state = "segments"
    cur_name = None
    cur_type = None
    cur_props: List[str] = []
    cur_x: List[float] = []
    cur_y: List[float] = []

    def flush_segment() -> None:
        if cur_name is None:
            return
        point_pairs: List[Dict[str, float]] = []
        for i in range(0, min(len(cur_x), len(cur_y)), 2):
            if i + 1 >= len(cur_x) or i + 1 >= len(cur_y):
                break
            point_pairs.append({
                "x1": cur_x[i], "y1": cur_y[i],
                "x2": cur_x[i + 1], "y2": cur_y[i + 1],
            })
        props_out = list(cur_props)
        effective_type = props_out[0] if props_out and str(props_out[0]).strip() else cur_type
        segments.append({
            "name": cur_name,
            "seg_type": effective_type,
            "properties": props_out,
            "point_pairs": point_pairs,
        })

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("title"):
            title = ln.split(":", 1)[1].strip() or title
            continue
        if state == "segments" and low.startswith("ibcs:"):
            flush_segment()
            cur_name = None
            state = "ibcs"
            continue
        if low.startswith("dielectrics:"):
            if state == "segments":
                flush_segment()
                cur_name = None
            state = "dielectrics"
            continue

        if state == "segments":
            if low.startswith("segment:"):
                flush_segment()
                parts = ln.split(":", 1)[1].strip().split()
                if not parts:
                    cur_name, cur_type = "Unnamed", None
                elif len(parts) == 1:
                    cur_name, cur_type = parts[0], None
                else:
                    cur_name, cur_type = parts[0], parts[1]
                cur_props = []
                cur_x = []
                cur_y = []
                continue
            if low.startswith("properties:"):
                cur_props = ln.split(":", 1)[1].strip().split()
                continue
            tokens = ln.split()
            if len(tokens) != 4:
                raise ValueError(
                    f"Geometry line must have 4 numbers, got {len(tokens)}: {ln!r}"
                )
            try:
                x1, y1, x2, y2 = map(float, tokens)
            except ValueError as e:
                raise ValueError(f"Geometry line must contain valid numbers: {ln!r}") from e
            cur_x.extend([x1, x2])
            cur_y.extend([y1, y2])
        elif state == "ibcs":
            tokens = ln.split()
            if tokens:
                ibcs_entries.append(tokens)
        elif state == "dielectrics":
            tokens = ln.split()
            if tokens:
                dielectric_entries.append(tokens)

    if state == "segments":
        flush_segment()

    return {
        "title": title,
        "segments": segments,
        "ibcs": ibcs_entries,
        "dielectrics": dielectric_entries,
    }


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
                          geometry_units: str = "inches") -> Dict[str, Any]:
    """
    Estimate panel/node/DOF counts and dense-solve memory for a geometry
    snapshot, without running the solver.

    NOTE: geometry_units defaults to "inches" to match the solver's default
    in solve_monostatic_rcs_2d / solve_bistatic_rcs_2d.  If your .geo file
    coordinates are actually in meters, pass geometry_units="meters"
    explicitly — otherwise the panel count estimate will be off by 25.4x
    in the wrong direction (reported count way too low if data is in mm,
    way too high if data is actually in inches).
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
        "geometry_units": geometry_units,
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
        f"Geometry units:          {est['geometry_units']}  "
        f"(assumed — override if wrong!)",
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
        print("usage: python estimate_problem_size.py <file.geo|file.json> "
              "<freq_ghz> [TE|TM] [inches|meters|mm|cm|feet]")
        print()
        print("NOTE: geometry units default to 'inches' to match the solver.")
        print("      Pass units explicitly if your .geo file is in meters etc.")
        sys.exit(1)
    path = sys.argv[1]
    freq = float(sys.argv[2])
    pol = sys.argv[3] if len(sys.argv) > 3 else "TE"
    units = sys.argv[4] if len(sys.argv) > 4 else "inches"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".geo":
        snap = load_geo_file(path)
    elif ext in (".json", ".grim"):
        import json
        with open(path) as f:
            snap = json.load(f)
    else:
        # Try .geo first as a fallback — it's the common case.
        try:
            snap = load_geo_file(path)
        except Exception:
            import json
            with open(path) as f:
                snap = json.load(f)
    print(format_estimate(estimate_problem_size(snap, freq, pol, units)))
