#!/usr/bin/env python3
"""Convenience wrapper for the reproducibility scripts."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MTM-Bench analysis scripts.")
    parser.add_argument("--scored_dir", default="outputs_scored/haiku judge")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--summary_dir", default="summary")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--mismatch_n_boot", type=int, default=5000)
    parser.add_argument("--skip_bootstrap", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    if not args.skip_summary:
        scored_files = sorted(Path(args.scored_dir).glob("*_tri_main_scored.csv"))
        if not scored_files:
            raise FileNotFoundError(f"No scored model files in {args.scored_dir}")
        summary_dir = Path(args.summary_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            long_path = Path(temp_dir) / "tri_main_analysis_long.csv"
            wide_path = Path(temp_dir) / "tri_main_analysis_wide.csv"
            for scored_file in scored_files:
                run([py, "scripts/analyze_results.py", "--infile", str(scored_file),
                     "--long_out", str(long_path), "--wide_out", str(wide_path)])
            shutil.copyfile(long_path, summary_dir / long_path.name)
            shutil.copyfile(wide_path, summary_dir / wide_path.name)

    run([py, "scripts/make_tables.py", "--scored_dir", args.scored_dir, "--out_dir", args.out_dir])
    if not args.skip_bootstrap:
        run([py, "scripts/bootstrap_axis_ranges.py", "--scored_dir", args.scored_dir,
             "--out", str(out_dir / "bootstrap_axis_ranges.csv"), "--n_boot", str(args.n_boot)])
        run([py, "scripts/bootstrap_resp_only_vs_full_mismatch.py", "--scored_dir", args.scored_dir,
             "--out", str(out_dir / "bootstrap_resp_only_vs_full_mismatch.csv"),
             "--n_boot", str(args.mismatch_n_boot)])
    run([py, "scripts/regression_analysis.py", "--scored_dir", args.scored_dir,
         "--out", str(out_dir / "regression_table.csv")])
    run([py, "scripts/make_judge_sensitivity_table.py",
         "--out", str(out_dir / "table8_judge_sensitivity.csv"), "--n_boot", str(args.n_boot)])
    run([py, "scripts/audit_summary.py", "--out", str(out_dir / "audit_summary.csv")])
    print("\nDone.")


if __name__ == "__main__":
    main()
