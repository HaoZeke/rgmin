#!/usr/bin/env python3
"""Create analyzed JSON and Markdown from an unmodified raw benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import mannwhitneyu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--energy-tolerance", type=float, required=True)
    return parser.parse_args()


def finite_values(
    records: Iterable[dict[str, Any]],
    key: str,
    *,
    converged_only: bool = True,
) -> list[float]:
    values = []
    for record in records:
        value = record.get(key)
        if (
            (record["converged"] or not converged_only)
            and value is not None
            and np.isfinite(value)
        ):
            values.append(float(value))
    return values


def quartiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q1": None, "q3": None}
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    return {"median": float(median), "q1": float(q1), "q3": float(q3)}


def mw_p(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    result = float(mannwhitneyu(left, right, alternative="two-sided", method="auto").pvalue)
    if np.isfinite(result):
        return result
    pooled = np.asarray(left + right, dtype=np.float64)
    return 1.0 if np.ptp(pooled) == 0.0 else None


def aa_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    odd = [row for row in records if row["round"] % 2 == 1]
    even = [row for row in records if row["round"] % 2 == 0]
    odd_wall = finite_values(odd, "wall_seconds", converged_only=False)
    even_wall = finite_values(even, "wall_seconds", converged_only=False)
    odd_median = quartiles(odd_wall)["median"]
    even_median = quartiles(even_wall)["median"]
    band = None
    relative_width = None
    if odd_median is not None and even_median is not None:
        band = [min(odd_median, even_median), max(odd_median, even_median)]
        centre = 0.5 * (odd_median + even_median)
        relative_width = 0.0 if centre == 0.0 else (band[1] - band[0]) / centre
    return {
        "even_n": len(even_wall),
        "odd_n": len(odd_wall),
        "even_median_seconds": even_median,
        "odd_median_seconds": odd_median,
        "noise_band_seconds": band,
        "relative_band_width": relative_width,
        "mann_whitney_p": mw_p(even_wall, odd_wall),
    }


def grouped_records(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["system"], record["method"], record["implementation"])].append(record)
    return groups


def summarize(groups: dict[tuple[str, str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for (system, method, implementation), records in sorted(groups.items()):
        successes = sum(bool(record["converged"]) for record in records)
        rows.append(
            {
                "system": system,
                "method": method,
                "implementation": implementation,
                "n": len(records),
                "successes": successes,
                "success_rate": successes / len(records),
                "wall_seconds": quartiles(
                    finite_values(records, "wall_seconds", converged_only=False)
                ),
                "outside_potential_seconds": quartiles(
                    finite_values(
                        records,
                        "outside_potential_seconds",
                        converged_only=False,
                    )
                ),
                "force_calls": quartiles(finite_values(records, "force_calls")),
                "energy_ev": quartiles(finite_values(records, "energy_ev")),
                "fmax_ev_per_angstrom": quartiles(
                    finite_values(records, "fmax_ev_per_angstrom")
                ),
                "stress_max_ev_per_angstrom3": quartiles(
                    finite_values(records, "stress_max_ev_per_angstrom3")
                ),
                "aa_wall": aa_summary(records),
            }
        )
    return rows


def paired_energy(
    reference: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    ref = {
        (record["seed"], record["round"]): record
        for record in reference
        if record["converged"] and record["energy_ev"] is not None
    }
    other = {
        (record["seed"], record["round"]): record
        for record in comparator
        if record["converged"] and record["energy_ev"] is not None
    }
    deltas = [
        abs(float(ref[key]["energy_ev"]) - float(other[key]["energy_ev"]))
        for key in sorted(ref.keys() & other.keys())
    ]
    return {
        "pairs": len(deltas),
        "agreements": sum(delta <= tolerance for delta in deltas),
        "tolerance_ev": tolerance,
        "max_abs_delta_ev": max(deltas) if deltas else None,
        "median_abs_delta_ev": float(np.median(deltas)) if deltas else None,
    }


def comparisons(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    systems_methods = sorted({(key[0], key[1]) for key in groups})
    for system, method in systems_methods:
        reference = groups.get((system, method, "rgmin"))
        if reference is None:
            continue
        for implementation in ("ase", "eon"):
            comparator = groups.get((system, method, implementation))
            if comparator is None:
                continue
            rows.append(
                {
                    "system": system,
                    "method": method,
                    "reference": "rgmin",
                    "comparator": implementation,
                    "wall_mann_whitney_p": mw_p(
                        finite_values(reference, "wall_seconds", converged_only=False),
                        finite_values(comparator, "wall_seconds", converged_only=False),
                    ),
                    "outside_potential_mann_whitney_p": mw_p(
                        finite_values(
                            reference,
                            "outside_potential_seconds",
                            converged_only=False,
                        ),
                        finite_values(
                            comparator,
                            "outside_potential_seconds",
                            converged_only=False,
                        ),
                    ),
                    "energy_agreement": paired_energy(reference, comparator, tolerance),
                }
            )
    return rows


def number(value: float | None, scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value * scale:.{digits}g}"


def interval(
    stats: dict[str, float | None],
    scale: float = 1.0,
    digits: int = 3,
) -> str:
    if stats["median"] is None:
        return "—"
    return (
        f"{number(stats['median'], scale, digits)} "
        f"[{number(stats['q1'], scale, digits)}, {number(stats['q3'], scale, digits)}]"
    )


def markdown_report(document: dict[str, Any]) -> str:
    summaries = {
        (row["system"], row["method"], row["implementation"]): row
        for row in document["summaries"]
    }
    comparisons_by_key = {
        (row["system"], row["method"], row["comparator"]): row
        for row in document["comparisons"]
    }
    systems_methods = sorted({(key[0], key[1]) for key in summaries})
    machine = document["runtime"]["machine"]
    thread_vars = document["runtime"].get("env_threads", {})
    threads = (
        ", ".join(f"{name}={value}" for name, value in sorted(thread_vars.items()))
        if thread_vars
        else "unset"
    )
    lines = [
        "# Optimizer coverage benchmark",
        "",
        (
            f"Release builds, {document['protocol']['thread_count']} pinned thread, "
            f"{document['protocol']['rounds']} order-flipped rounds, no warm-up and no rescue reruns. "
            f"Convergence is {document['protocol']['fmax_ev_per_angstrom']} eV/A; "
            f"cell rows also require {document['protocol']['stress_max_ev_per_angstrom3']} eV/A^3. "
            f"Energy agreement tolerance is "
            f"{document['protocol']['energy_agreement_tolerance_ev']} eV."
        ),
        "",
        (
            f"Machine: {machine['cpu']}; flags: {machine['cpu_flags_summary']}; "
            f"architecture: {machine['architecture']}; available logical CPUs: "
            f"{machine['available_logical_cpus']}/{machine['logical_cpus']}; kernel: "
            f"{document['runtime'].get('kernel', document['runtime']['release'])}; "
            f"Python: {document['runtime']['python']}; Rust: {document['build']['rust']}; "
            f"C++: {document['build']['cxx']}; BLAS: {document['runtime'].get('blas', 'unknown')}; "
            f"thread env: {threads}."
        ),
        "",
        (
            "P-values below are within-run timing-noise tests only "
            "(Mann-Whitney on wall and outside-potential timings)."
        ),
        "",
        "| System | Method | Code | Success | Wall ms, median [IQR] | Outside potential ms | Force calls to converge | Final energy eV | Wall timing p | Outside timing p | Energy agreement |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gaps = {(gap["method"], gap["implementation"]): gap["reason"] for gap in document["gaps"]}
    for system, method in systems_methods:
        for implementation in ("rgmin", "ase", "eon"):
            row = summaries.get((system, method, implementation))
            if row is None:
                gap = gaps.get((method, implementation))
                if gap:
                    lines.append(
                        f"| {system} | {method} | {implementation} | gap | — | — | — | — | — | — | {gap} |"
                    )
                continue
            comparison = comparisons_by_key.get((system, method, implementation))
            energy = "—"
            wall_p = outside_p = "—"
            if comparison is not None:
                agreement = comparison["energy_agreement"]
                energy = (
                    "n/a"
                    if agreement["pairs"] == 0
                    else f"{agreement['agreements']}/{agreement['pairs']}; "
                    f"max Δ={number(agreement['max_abs_delta_ev'])}"
                )
                wall_p = number(comparison["wall_mann_whitney_p"])
                outside_p = number(comparison["outside_potential_mann_whitney_p"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        system,
                        method,
                        implementation,
                        f"{row['successes']}/{row['n']}",
                        interval(row["wall_seconds"], 1000.0),
                        interval(row["outside_potential_seconds"], 1000.0),
                        interval(row["force_calls"]),
                        interval(row["energy_ev"]),
                        wall_p,
                        outside_p,
                        energy,
                    ]
                )
                + " |"
            )
    failures = document["failures"]
    failure_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        failure_groups[(row["system"], row["method"], row["implementation"])].append(row)
    failure_text = "\n".join(
        f"- {system} {method} {implementation}: {len(rows)} nonconverged; "
        f"force calls/attempt "
        f"{interval(quartiles([row['force_calls'] for row in rows]), digits=6)}."
        for (system, method, implementation), rows in sorted(failure_groups.items())
    )
    widest = max(
        document["summaries"],
        key=lambda row: row["aa_wall"]["relative_band_width"]
        if row["aa_wall"]["relative_band_width"] is not None
        else -1.0,
    )
    variable_cell = []
    for system in ("Cu108-cell", "Si64-cell"):
        rgmin_cell = summaries.get((system, "cell-lbfgs", "rgmin"))
        ase_cell = summaries.get((system, "cell-lbfgs", "ase"))
        if (
            rgmin_cell is not None
            and ase_cell is not None
        ):
            variable_cell.append(
                f"- {system}: rgmin converged {rgmin_cell['successes']}/{rgmin_cell['n']}; "
                f"ASE converged {ase_cell['successes']}/{ase_cell['n']}."
            )
    eon_sd_records = [
        row
        for row in document["records"]
        if row["implementation"] == "eon" and row["method"] == "sd"
    ]
    eon_budget = sum(bool(row.get("hit_iteration_budget")) for row in eon_sd_records)
    eon_sd_note = "No eOn SD rows were recorded."
    if eon_sd_records:
        sample = eon_sd_records[0].get("eon_config", {})
        eon_sd_note = (
            f"eOn SD uses opt_convergence_metric={sample.get('opt_convergence_metric')} "
            f"with opt_converged_force={sample.get('opt_converged_force')} eV/A, "
            f"opt_step_size={sample.get('opt_step_size')}, opt_max_move={sample.get('opt_max_move')}, "
            f"and opt_max_iterations={sample.get('opt_max_iterations')}. "
            f"{eon_budget}/{len(eon_sd_records)} SD runs hit the iteration budget."
        )
    lines.extend(
        [
            "",
            "The result JSON contains the odd/even A/A wall-time noise band and its Mann-Whitney p-value for every system/method/code cell. "
            f"The widest relative band is {number(widest['aa_wall']['relative_band_width'], 100.0)}% "
            f"for {widest['system']} {widest['method']} {widest['implementation']}.",
            "",
            "## Failures and gaps",
            "",
            f"Nonconverged or failed runs: {len(failures)}.",
            failure_text,
            "The eOn client has no variable-cell optimizer. A generic rgpot EMT Python binding was absent and is delivered as a separate source revision; the selected Cu systems use the predeclared Morse potential.",
            eon_sd_note,
            "",
            "## Variable-cell nonconvergence",
            "",
            "\n".join(variable_cell),
            "- Minimal reproduction: `bench/optimizers/repro_cell.py` runs the same capped L-BFGS and filter without the benchmark scheduler.",
            "",
            "Within-run caveat: endpoint and iteration-count comparisons are deterministic under this protocol; timing statistics quantify noise only within this run's machine and software stack.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("analysis output must be separate from the raw input")
    document = json.loads(args.input.read_text(encoding="utf-8"))
    generated_keys = {"analysis", "summaries", "comparisons", "failures"}
    present = generated_keys & document.keys()
    if present:
        raise ValueError(f"raw input contains analysis keys: {sorted(present)}")
    tolerance = float(document["protocol"]["energy_agreement_tolerance_ev"])
    if tolerance != args.energy_tolerance:
        raise ValueError("analysis tolerance differs from the declared benchmark tolerance")
    records = document["records"]
    groups = grouped_records(records)
    document["analysis"] = {
        "continuous_test": "two-sided Mann-Whitney U",
        "energy_agreement_tolerance_ev": tolerance,
        "input": args.input.name,
        "p_value_scope": "within-run timing noise only",
        "schema": 1,
    }
    document["summaries"] = summarize(groups)
    document["comparisons"] = comparisons(groups, tolerance)
    document["failures"] = [
        {
            "system": row["system"],
            "method": row["method"],
            "implementation": row["implementation"],
            "seed": row["seed"],
            "round": row["round"],
            "error": row["error"],
            "force_calls": row["force_calls"],
            "wall_seconds": row["wall_seconds"],
            "fmax_ev_per_angstrom": row["fmax_ev_per_angstrom"],
            "stress_max_ev_per_angstrom3": row["stress_max_ev_per_angstrom3"],
        }
        for row in records
        if not row["converged"]
    ]
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(markdown_report(document), encoding="utf-8")


if __name__ == "__main__":
    main()
