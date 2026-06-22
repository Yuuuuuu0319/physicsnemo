"""Audit HEC-RAS internal-face features for edge-local conservation.

The edge-flux head failed to learn native HEC-RAS Face Flow magnitude from the
current compact feature set. This script checks which physical edge features are
available, writes a per-face feature table, and reports simple correlations
against the HEC-RAS internal face-flow target statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - optional diagnostic dependency
    h5py = None


def read_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def read_matrix(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"{path} must be a 2D text matrix, got shape {values.shape}.")
    return values


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def safe_log1p_positive(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0)
    output[mask] = np.log1p(values[mask])
    return output


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": float(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
    }


def load_landcover_lookup(path: Path | None) -> dict[float, float]:
    if path is None or h5py is None or not path.exists():
        return {}
    with h5py.File(path, "r") as handle:
        if "Variables" not in handle:
            return {}
        variables = handle["Variables"][:]
    if variables.dtype.names is None:
        return {}
    names = variables.dtype.names
    manning_name = next(
        (name for name in names if name == "ManningsN" or "manning" in name.lower()),
        None,
    )
    ip_name = next(
        (
            name
            for name in names
            if name == "Percent Impervious" or "impervious" in name.lower()
        ),
        None,
    )
    if manning_name is None or ip_name is None:
        return {}
    return {
        round(float(row[manning_name]), 6): float(row[ip_name])
        for row in variables
    }


def write_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.10g",
    )


def compute_surface_gradient_rms(
    data_dir: Path,
    ids: list[str],
    src: np.ndarray,
    dst: np.ndarray,
    face_length: np.ndarray,
    elevation: np.ndarray,
    max_events: int | None,
) -> tuple[np.ndarray, int, int]:
    sum_sq = np.zeros(src.shape[0], dtype=np.float64)
    count = 0
    used_events = ids if max_events is None else ids[:max_events]
    length = np.maximum(face_length, 1e-6)
    for hydrograph_id in used_events:
        wd = read_matrix(data_dir / f"M80_WD_{hydrograph_id}.txt")
        if wd.shape[1] != elevation.shape[0]:
            raise ValueError(
                f"M80_WD_{hydrograph_id}.txt has {wd.shape[1]} nodes; "
                f"expected {elevation.shape[0]}."
            )
        surface = wd + elevation.reshape(1, -1)
        # Targets are transition deltas, so use surface at the current state.
        grad = (surface[:-1, dst] - surface[:-1, src]) / length.reshape(1, -1)
        sum_sq += np.sum(grad * grad, axis=0)
        count += grad.shape[0]
    if count == 0:
        return np.full(src.shape[0], np.nan), 0, 0
    return np.sqrt(sum_sq / count), len(used_events), count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit HEC-RAS internal-face features for HydroGraphNet."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--face-graph-file", required=True, type=Path)
    parser.add_argument("--face-stats-file", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--zone-label-file", type=Path)
    parser.add_argument("--landcover-hdf", type=Path)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    face_graph = np.load(args.face_graph_file)
    face_index = face_graph["internal_face_index"].astype(np.int64)
    src = face_index[0]
    dst = face_index[1]
    face_length = face_graph["internal_face_length"].astype(np.float64)
    normal = face_graph["internal_normal_unit"].astype(np.float64)

    stats = np.load(args.face_stats_file)
    face_rms = stats["face_rms"].astype(np.float64).reshape(-1)
    face_mean = stats["face_mean"].astype(np.float64).reshape(-1)
    face_std = stats["face_std"].astype(np.float64).reshape(-1)

    xy = read_matrix(args.data_dir / "M80_XY.txt")
    elevation = read_vector(args.data_dir / "M80_CE.txt")
    manning = read_vector(args.data_dir / "M80_N.txt")
    ip = read_vector(args.data_dir / "M80_IP.txt")
    area = read_vector(args.data_dir / "M80_CA.txt")
    zone = (
        read_vector(args.zone_label_file).astype(np.int64)
        if args.zone_label_file is not None and args.zone_label_file.exists()
        else np.zeros(elevation.shape[0], dtype=np.int64)
    )

    if face_rms.shape[0] != src.shape[0]:
        raise ValueError(
            f"face stats contain {face_rms.shape[0]} faces but graph has {src.shape[0]}."
        )

    landcover_lookup = load_landcover_lookup(args.landcover_hdf)
    rounded_manning = np.round(manning, 6)
    landcover_ip = np.full_like(ip, np.nan, dtype=np.float64)
    for key, value in landcover_lookup.items():
        landcover_ip[np.isclose(rounded_manning, key, rtol=0.0, atol=1e-6)] = value
    landcover_matched = np.isfinite(landcover_ip)

    ids = load_ids(args.ids_file)
    surface_grad_rms, used_events, used_transitions = compute_surface_gradient_rms(
        args.data_dir,
        ids,
        src,
        dst,
        face_length,
        elevation,
        args.max_events,
    )

    dx = xy[dst, 0] - xy[src, 0]
    dy = xy[dst, 1] - xy[src, 1]
    center_distance = np.sqrt(dx * dx + dy * dy)
    elevation_diff = elevation[dst] - elevation[src]
    bed_slope = elevation_diff / np.maximum(center_distance, 1e-6)
    manning_mean = 0.5 * (manning[src] + manning[dst])
    manning_diff = manning[dst] - manning[src]
    ip_mean = 0.5 * (ip[src] + ip[dst])
    ip_diff = ip[dst] - ip[src]
    area_mean = 0.5 * (area[src] + area[dst])
    area_ratio = np.maximum(area[src], area[dst]) / np.maximum(
        np.minimum(area[src], area[dst]), 1e-6
    )
    high_touching = ((zone[src] == 3) | (zone[dst] == 3)).astype(np.float64)
    high_internal = ((zone[src] == 3) & (zone[dst] == 3)).astype(np.float64)
    same_zone = (zone[src] == zone[dst]).astype(np.float64)

    columns = [
        "face_id",
        "src",
        "dst",
        "face_length",
        "normal_x",
        "normal_y",
        "center_distance",
        "elevation_src",
        "elevation_dst",
        "elevation_diff",
        "bed_slope",
        "manning_src",
        "manning_dst",
        "manning_mean",
        "manning_diff",
        "ip_src",
        "ip_dst",
        "ip_mean",
        "ip_diff",
        "landcover_ip_src",
        "landcover_ip_dst",
        "area_src",
        "area_dst",
        "area_mean",
        "area_ratio",
        "zone_src",
        "zone_dst",
        "same_zone",
        "zone3_touching",
        "zone3_internal",
        "surface_gradient_rms",
        "face_target_mean",
        "face_target_std",
        "face_target_rms",
    ]
    rows = np.column_stack(
        [
            np.arange(src.shape[0], dtype=np.float64),
            src,
            dst,
            face_length,
            normal[:, 0],
            normal[:, 1],
            center_distance,
            elevation[src],
            elevation[dst],
            elevation_diff,
            bed_slope,
            manning[src],
            manning[dst],
            manning_mean,
            manning_diff,
            ip[src],
            ip[dst],
            ip_mean,
            ip_diff,
            landcover_ip[src],
            landcover_ip[dst],
            area[src],
            area[dst],
            area_mean,
            area_ratio,
            zone[src],
            zone[dst],
            same_zone,
            high_touching,
            high_internal,
            surface_grad_rms,
            face_mean,
            face_std,
            face_rms,
        ]
    )
    write_csv(args.output_csv, columns, rows)

    target = np.log1p(np.abs(face_rms))
    correlations = {
        "log1p(face_target_rms) vs log1p(face_length)": pearson(
            target, safe_log1p_positive(face_length)
        ),
        "log1p(face_target_rms) vs abs(elevation_diff)": pearson(
            target, np.abs(elevation_diff)
        ),
        "log1p(face_target_rms) vs abs(bed_slope)": pearson(
            target, np.abs(bed_slope)
        ),
        "log1p(face_target_rms) vs manning_mean": pearson(target, manning_mean),
        "log1p(face_target_rms) vs ip_mean": pearson(target, ip_mean),
        "log1p(face_target_rms) vs log1p(area_mean)": pearson(
            target, safe_log1p_positive(area_mean)
        ),
        "log1p(face_target_rms) vs log1p(surface_gradient_rms)": pearson(
            target, safe_log1p_positive(surface_grad_rms)
        ),
        "log1p(face_target_rms) vs zone3_touching": pearson(target, high_touching),
        "log1p(face_target_rms) vs zone3_internal": pearson(target, high_internal),
    }

    zone_lines = []
    for label, mask in [
        ("all", np.ones_like(high_touching, dtype=bool)),
        ("zone3_touching", high_touching.astype(bool)),
        ("zone3_internal", high_internal.astype(bool)),
        ("cross_zone", ~same_zone.astype(bool)),
    ]:
        summary = summarize(face_rms[mask])
        zone_lines.append(
            "| {label} | {count:.0f} | {median:.3f} | {mean:.3f} | {p90:.3f} | {p99:.3f} | {max:.3f} |".format(
                label=label,
                **summary,
            )
        )

    top_idx = np.argsort(face_rms)[-10:][::-1]
    top_lines = [
        (
            f"| {idx} | {int(src[idx])} | {int(dst[idx])} | "
            f"{int(zone[src[idx]])}-{int(zone[dst[idx]])} | "
            f"{face_rms[idx]:.3f} | {surface_grad_rms[idx]:.6g} | "
            f"{face_length[idx]:.3f} | {manning_mean[idx]:.6f} | {ip_mean[idx]:.3f} |"
        )
        for idx in top_idx
    ]

    feature_status = [
        ("HEC-RAS internal face connectivity", "available", str(args.face_graph_file)),
        ("Face length and normal", "available", str(args.face_graph_file)),
        ("Cell elevation/area/Manning/IP", "available", str(args.data_dir)),
        (
            "LandCover Manning-to-IP lookup",
            "available" if landcover_lookup else "missing",
            str(args.landcover_hdf) if args.landcover_hdf else "",
        ),
        (
            "Dynamic water-surface gradient proxy",
            "available",
            f"{used_events} events / {used_transitions} transitions",
        ),
        ("Native internal Face Flow target stats", "available", str(args.face_stats_file)),
        ("Boundary face flow supervision", "not in this audit", "edge NPZ stores node boundary/source delta"),
        ("HEC-RAS face hydraulic conveyance/open area", "missing", "not exported to current NPZ"),
        ("Autoregressive predicted edge state", "missing", "not a deployed model feature yet"),
    ]

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_md.open("w") as handle:
        handle.write("# Minxiong HEC-RAS Edge Feature Audit\n\n")
        handle.write("## Inputs\n\n")
        handle.write(f"- data dir: `{args.data_dir}`\n")
        handle.write(f"- face graph: `{args.face_graph_file}`\n")
        handle.write(f"- face stats: `{args.face_stats_file}`\n")
        handle.write(f"- ids file: `{args.ids_file}`\n")
        handle.write(f"- landcover HDF: `{args.landcover_hdf}`\n")
        handle.write(f"- output CSV: `{args.output_csv}`\n\n")

        handle.write("## Feature Availability\n\n")
        handle.write("| Feature | Status | Source / note |\n")
        handle.write("|---|---|---|\n")
        for name, status, note in feature_status:
            handle.write(f"| {name} | {status} | `{note}` |\n")
        handle.write("\n")

        handle.write("## LandCover / IP Check\n\n")
        handle.write(f"- LandCover classes read: `{len(landcover_lookup)}`\n")
        handle.write(
            f"- Cells matched by Manning's n lookup: `{int(landcover_matched.sum())}` / `{landcover_matched.size}`\n"
        )
        handle.write(f"- HGN `M80_IP` unique values: `{np.unique(ip).tolist()}`\n")
        if landcover_lookup:
            handle.write(f"- LandCover Manning to IP mapping: `{landcover_lookup}`\n")
        handle.write("\n")

        handle.write("## Face Target RMS by Zone Group\n\n")
        handle.write("| Group | Faces | Median ft^3 | Mean ft^3 | P90 ft^3 | P99 ft^3 | Max ft^3 |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        handle.write("\n".join(zone_lines))
        handle.write("\n\n")

        handle.write("## Feature Correlations\n\n")
        handle.write("Pearson correlation uses `log1p(face_target_rms)` as the target.\n\n")
        handle.write("| Feature comparison | Pearson r |\n")
        handle.write("|---|---:|\n")
        for name, value in correlations.items():
            handle.write(f"| {name} | {value:.6f} |\n")
        handle.write("\n")

        handle.write("## Top 10 High-RMS Faces\n\n")
        handle.write("| Face | Src | Dst | Zone pair | Target RMS ft^3 | Surface grad RMS | Face length | Manning mean | IP mean |\n")
        handle.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        handle.write("\n".join(top_lines))
        handle.write("\n\n")

        handle.write("## Interpretation\n\n")
        handle.write(
            "- The current data now has enough topology and static hydraulic context "
            "to build a better edge feature matrix: face length, normal, cell "
            "area, bed elevation difference, Manning's n, IP, zone pair, and a "
            "dynamic water-surface gradient proxy.\n"
        )
        handle.write(
            "- The current edge head did not consume this full audited feature set; "
            "it only used a subset through node features and optional face "
            "normal / previous face flow. This audit identifies the next concrete "
            "feature block for a publishable edge-local conservation branch.\n"
        )
        handle.write(
            "- Missing pieces for a DUALFloodGNN-level edge method remain: "
            "autoregressive edge state, stronger edge message passing, and direct "
            "boundary-face flow representation rather than only node-level "
            "boundary/source residuals.\n"
        )


if __name__ == "__main__":
    main()
