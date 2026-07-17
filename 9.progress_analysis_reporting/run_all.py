#!/usr/bin/env python3
r"""
run_all.py
============
Convenience wrapper: runs step1 (plots) + step2 (per-patient report) for
every patient, then step3 (overall report) once at the end. Equivalent to
calling all three scripts yourself, in order.

USAGE:
  # ALL patients + overall report
  python run_all.py --dataset D:\NahidW\Dataset\face_burn_dataset

  # one patient only (still safe to run any time; overall report is
  # skipped since it's a whole-dataset view)
  python run_all.py --dataset D:\NahidW\Dataset\face_burn_dataset --patient PAT01
"""

import argparse
from pathlib import Path

from step1_generate_plots import generate_for_patient as gen_plots
from step2_generate_patient_report import build_patient_report
from step3_generate_overall_report import build_overall_report
from analysis_common import discover_patients


def main():
    p = argparse.ArgumentParser(description="Run the full analysis+reporting pipeline")
    p.add_argument("--dataset", required=True, help="Dataset root")
    p.add_argument("--patient", default=None,
                    help="Limit to one patient (skips the overall report). "
                         "Omit to process ALL patients + the overall report.")
    args = p.parse_args()
    dataset_dir = Path(args.dataset)

    if args.patient:
        gen_plots(dataset_dir, args.patient)
        out = build_patient_report(dataset_dir, args.patient)
        print(f"  ✓ {args.patient} report → {out}")
        return

    patients = discover_patients(dataset_dir)
    if not patients:
        print(f"  No patient folders found under {dataset_dir}")
        return

    print(f"=== Generating plots + reports for {len(patients)} patient(s) ===\n")
    for pd in patients:
        gen_plots(dataset_dir, pd.name)
        out = build_patient_report(dataset_dir, pd.name)
        print(f"    → {out}\n")

    print("=== Generating overall (all-patients) report ===")
    out = build_overall_report(dataset_dir)
    print(f"  ✓ Overall report → {out}")


if __name__ == "__main__":
    main()
