"""Regenerate figures 2, 4-9 from results_for_manuscript/figures/data into figures/regenerated/.

    python results_for_manuscript/figures/scripts/make_all.py

Figure numbers are the v4.2 manuscript numbers. Fig. 1 (hand-built from server panels) and
Fig. 3 (needs checkpoints) are not regenerated here. The output is a regeneration from the
corrected results, not a byte copy of HoloQPI_4.2/manuscript_images/: font/matplotlib version
and small layout differences are expected; the plotted data are the same.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
REPO = HERE.parents[2]
OUT = FIG / "regenerated"
OUT.mkdir(exist_ok=True)
# script -> (name the script writes, v4.2 figure name)
JOBS = [("fig2_learned_vs_classical.py", "figure_2.png", "figure_2.png"),
        ("fig4_agreement.py", "figure_6.png", "figure_4.png"),
        ("fig5_ablation.py", "figure_4.png", "figure_5.png"),
        ("fig6_weight_sweep.py", "figure_5.png", "figure_6.png"),
        ("fig7_forward_model.py", "figure_7.png", "figure_7.png"),
        ("fig9_efficiency.py", "figure_9.png", "figure_9.png")]
for script, written, final in JOBS:
    subprocess.run([sys.executable, script], cwd=HERE, check=True, env={**__import__("os").environ, "HOLOQPI_RUNS": str(FIG / "data")})
    shutil.move(OUT / written, OUT / ("_" + final))
    print(f"{script} -> regenerated/{final}")
for _, _, final in JOBS:
    (OUT / ("_" + final)).replace(OUT / final)
subprocess.run([sys.executable, "fig8_boundary_sensitivity.py", "--root", str(REPO), "--out", str(OUT)], cwd=HERE, check=True)
print("done")
