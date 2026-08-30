"""Export the estimator accuracy report used by the 3D demo's Accuracy tab.

Runs tests/test_estimation_accuracy.py's scenario harness and writes
frontend/accuracy_report.json, then regenerates the inline blob that gets
embedded in frontend/tyre3d.html (fetch() is blocked on file:// URLs, and the
demo must stay a single file, so the data is inlined rather than loaded).

    python scripts/export_accuracy.py

Diagnostic columns worth knowing about:

  error_sd          standard deviation of the tread error across timesteps.
                    Zero means the estimate never moved.
  offset_fraction   |bias| / MAE. 1.0 means the entire error is a constant
                    offset, i.e. the MAE is measuring the distance from the
                    prior to ground truth rather than estimator accuracy.
"""

from __future__ import annotations

import datetime
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_estimation_accuracy as T  # noqa: E402
from src.evtyre.simulation.scenarios import ScenarioType  # noqa: E402

PRIOR_TREAD, PRIOR_SIG, N_STEPS = 4.8, 2.5, 20


def stats(m) -> dict:
    sd = math.sqrt(max(0.0, m.rmse**2 - m.bias**2))
    return {
        "mae": round(m.mae, 4), "rmse": round(m.rmse, 4), "bias": round(m.bias, 4),
        "max": round(m.max_error, 4), "error_sd": round(sd, 5),
        "offset_fraction": round(abs(m.bias) / m.mae, 4) if m.mae > 0 else 1.0,
        "coverage": round(m.coverage_2sigma, 4), "n": m.n_samples,
    }


def main() -> None:
    report = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": "tests/test_estimation_accuracy.py (441-test suite)",
        "prior_tread_mm": PRIOR_TREAD, "prior_tread_sigma_mm": PRIOR_SIG,
        "tread_new_mm": 8.0, "tread_legal_mm": 1.6, "n_steps": N_STEPS, "scenarios": [],
    }
    rows = []
    for st in ScenarioType:
        r = T._run_scenario_accuracy(st, n_steps=N_STEPS)
        tpc = {c: stats(m) for c, m in r.tread_metrics.items()}
        ppc = {c: stats(m) for c, m in r.pressure_metrics.items()}
        gt = {c: round(PRIOR_TREAD - m.bias, 3) for c, m in r.tread_metrics.items()}
        report["scenarios"].append({
            "name": st.value, "tread_mae": round(r.mean_tread_mae, 4),
            "press_mae": round(r.mean_pressure_mae, 4),
            "coverage_2sigma": round(r.mean_tread_coverage, 4),
            "tread_error_sd_max": max(v["error_sd"] for v in tpc.values()),
            "tread_offset_fraction_min": round(min(v["offset_fraction"] for v in tpc.values()), 4),
            "implied_gt_tread_mm": gt, "tread_per_corner": tpc, "press_per_corner": ppc,
        })
        rows.append({
            "n": st.value, "tm": round(r.mean_tread_mae, 4), "pm": round(r.mean_pressure_mae, 4),
            "cv": round(r.mean_tread_coverage, 4),
            "sd": max(v["error_sd"] for v in tpc.values()),
            "off": round(min(v["offset_fraction"] for v in tpc.values()), 4), "gt": gt,
            "tc": {c: [v["mae"], v["rmse"], v["bias"], v["max"], v["error_sd"]] for c, v in tpc.items()},
            "pc": {c: [v["mae"], v["rmse"], v["bias"], v["max"], v["error_sd"]] for c, v in ppc.items()},
        })

    out = ROOT / "frontend" / "accuracy_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    blob = {
        "gen": report["generated_utc"], "src": report["source"], "prior": PRIOR_TREAD,
        "priorSig": PRIOR_SIG, "treadNew": 8.0, "treadLegal": 1.6, "steps": N_STEPS, "rows": rows,
    }
    inline = "const ACCURACY_DATA = " + json.dumps(blob, separators=(",", ":")) + ";"

    html_path = ROOT / "frontend" / "tyre3d.html"
    html = html_path.read_text(encoding="utf-8")
    start = html.find("const ACCURACY_DATA = ")
    if start == -1:
        raise SystemExit("ACCURACY_DATA block not found in tyre3d.html")
    end = html.index(";\n", start) + 1
    html_path.write_text(html[:start] + inline + html[end:], encoding="utf-8")

    print(f"wrote {out.relative_to(ROOT)} and re-inlined {len(rows)} scenarios into tyre3d.html")
    sd = max(s["tread_error_sd_max"] for s in report["scenarios"])
    off = min(s["tread_offset_fraction_min"] for s in report["scenarios"])
    print(f"max tread error SD {sd:.5f} mm | min offset fraction {off * 100:.1f}%")


if __name__ == "__main__":
    main()
