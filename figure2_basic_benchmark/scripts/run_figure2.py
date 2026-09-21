from __future__ import annotations

"""Figure 2 — end-to-end. Regenerates every Figure 2 output with one command.

    conda run -n hector python run_figure2.py

Produces, under result/:
    figure/figure2_main.pdf / .png                  main figure, human kidney, panels A-D
    figure/supplemental/supplementary_figure_2_human_kidney_detail.pdf / .png
                                                   inference timing + per-class F1
    figure/supplemental/supplementary_figure_<n>_<tissue>.pdf / .png
                                                   full benchmark figure for each of the
                                                   five generalization datasets, numbered
                                                   3-7 by the order the draft cites them
    table/kidney_per_class_f1.csv
    table/kidney_confusion.csv, kidney_topology.csv
    table/benchmark_metrics.csv                    interpretable metrics, all datasets

Everything expensive is compute-if-missing into the local `.cache/`: the benchmark
predictions (HECTOR + the PopV suite) and the human-kidney deep-dive. A first run from
an empty cache does the full inference; later runs reuse it and are fast. Delete
`.cache/` to force a genuine from-scratch rebuild.

Flags:
    --main-only   stop after the main figure and the kidney tables (skips supplements)
"""

import argparse
import os
import shutil
import sys

# PYTHONHASHSEED pin is a redundant safety net; the ontology/confusion order is
# deterministic regardless (ontology_compare sorts each node's children by CL ID).
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

from pathlib import Path

from benchmark_modules import benchmark_core as core
from benchmark_modules import dataset_configs as dsc
from benchmark_modules import figure2_assembly as fig2

# figure2_basic_benchmark/, holding input/, scripts/ and result/. Anchoring here
# means the run does not depend on which folder it was started from.
FIGURE = Path(__file__).resolve().parents[1]
FIG_DIR = FIGURE / "result" / "figure"
SUPP_DIR = FIG_DIR / "supplemental"
TABLE_DIR = FIGURE / "result" / "table"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate every Figure 2 output (main figure, supplements, tables).")
    ap.add_argument("--main-only", action="store_true",
                    help="stop after the main figure and kidney tables")
    args = ap.parse_args()

    keys = ["kidney"] if args.main_only else ["kidney", *dsc.GENERALIZATION_KEYS]
    for d in (FIG_DIR, SUPP_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[1/4] Ensuring benchmark predictions (compute-if-missing)...")
    for key in keys:
        dsc.ensure_inference(key)

    print("[2/4] Ensuring human-kidney deep-dive (compute-if-missing)...")
    kidney_cfg = fig2.kidney_config()
    fig2.ensure_deepdive(kidney_cfg)

    print("[3/4] Assembling main figure...")
    fig2.assemble_figure2(kidney_cfg, out_stem=str(FIG_DIR / "figure2_main"))
    fig2.write_kidney_tables(kidney_cfg, str(TABLE_DIR))

    if args.main_only:
        print(f"Done (main only): {FIG_DIR}/figure2_main.* + {TABLE_DIR}/kidney_*.csv")
        return

    # Every supplement is written under the name its supplementary figure number gives
    # it, so a file can be matched to a citation without consulting a table. The numbers
    # come from dsc.SUPPLEMENTS.
    print("[4/4] Rendering supplements...")
    fig2.plot_supplement_detail_combined(
        kidney_cfg, out=str(SUPP_DIR / dsc.supplement_pdf_name("kidney")))

    # Each generalization dataset keeps its FULL benchmark figure (UMAP grid + macro
    # matrix + timing + per-class F1), rendered by the benchmark_core plotter.
    for key in dsc.GENERALIZATION_KEYS:
        print(f"  {key}...")
        cfg = dsc.benchmark_config(key)
        core.run_plots_for_dataset(cfg)
        # Moved, not copied: the plotter writes into the dataset's cache directory,
        # which otherwise ends up holding a second copy of every supplement (23 MB)
        # that nothing reads. The cache keeps the frozen predictions; the figures
        # belong under result/.
        source_pdf = Path(cfg["results_dir"]) / cfg["figure_file_name"]
        source_png = source_pdf.with_suffix(".png")
        destination_pdf = SUPP_DIR / source_pdf.name
        destination_png = SUPP_DIR / source_png.name

        shutil.move(str(source_pdf), str(destination_pdf))
        shutil.move(str(source_png), str(destination_png))

    fig2.write_benchmark_metrics_table(keys, dsc.CACHE_ROOT, str(TABLE_DIR))
    print(f"Done: {FIG_DIR}/figure2_main.*, {SUPP_DIR}/*, {TABLE_DIR}/*.csv")


if __name__ == "__main__":
    main()
