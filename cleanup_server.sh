#!/usr/bin/env bash
# =============================================================================
# cleanup_server.sh -- removes old results and working files from the server
# project folder, the same way UPDATE_AND_CLEANUP.cmd does on the PC.
#
#   cd /sda/usama/QPI_Extended_v2
#   bash cleanup_server.sh
#
# Run it AFTER run_server_figures.sh has said FINISHED and you have
# server_figs.zip on your PC. It lists what it will delete, asks once, then
# deletes and prints each item. Nothing outside this folder is touched.
#
# Kept: the code, configs, docs, run_v2.sh, the new runs/, figures/ and data/,
# the logs of the corrected study (logs/corrected_20260930_054214 and the
# logs/v2_20260930_* folders), weights and manuscript_images/.
# =============================================================================

cd "$(dirname "$0")" || exit 1
[ -f main.py ] && [ -d holoqpi ] || { echo "STOP: run this from the project folder (the one with main.py). Nothing was deleted."; exit 1; }

FILES=(
  apply_fixes.py run_corrected_study.sh run_everything.sh cleanup.sh
  run_server_figures.sh server_package.zip
  corrected_package.zip corrected_results.zip diagnostics_package.zip diagnostics_results.zip
  decompose_log.txt register_log.txt DIAGNOSTICS_ANSWERS.md FIX_README.md RESPONSE.md
  CLEANUP.cmd del_red.cmd docs/complete_documentation.pdf
  test/HoloQPI_Evaluation.EXAMPLE_RUN.ipynb runs/mass_uncertainty.json
)
DIRS=(
  archive logs_run test_outputs test/test_outputs
  runs/base_off_axis runs/base_s1337_off_axis runs/base_s2024_off_axis runs/base_gabor
  logs/corrected_20260930_045329
)

echo
echo "==========================================================================="
echo " cleanup_server in $(pwd)"
echo "==========================================================================="
echo " [1/4] Checking what is present"
TODO_F=(); TODO_D=()
for f in "${FILES[@]}"; do [ -f "$f" ] && TODO_F+=("$f"); done
for f in ./*.out; do [ -f "$f" ] && TODO_F+=("${f#./}"); done
for d in "${DIRS[@]}"; do [ -d "$d" ] && TODO_D+=("$d"); done
for d in logs/v2_20260921_* fix_backup_*; do [ -d "$d" ] && TODO_D+=("$d"); done
while IFS= read -r d; do TODO_D+=("$d"); done < <(find . -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -not -path "./.git/*" | sed 's#^\./##')
for f in "${TODO_F[@]}"; do echo "    file    $f"; done
for d in "${TODO_D[@]}"; do echo "    folder  $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"; done
echo "    found ${#TODO_F[@]} files and ${#TODO_D[@]} folders to delete"
if [ "${#TODO_F[@]}" -eq 0 ] && [ "${#TODO_D[@]}" -eq 0 ]; then
  echo " Nothing to delete. The folder is already clean."; exit 0
fi
echo
echo " archive/ holds the runs, figures, amplitude reference and aberration.json"
echo " of the study BEFORE the crop correction. It is deleted too."
read -r -p " Delete everything listed above? [y/N] " ans
case "$ans" in y|Y|yes|YES) ;; *) echo " Cancelled. Nothing was deleted."; exit 0;; esac

echo " [2/4] Deleting files"
n=0; for f in "${TODO_F[@]}"; do rm -f -- "$f" && { echo "    deleted $f"; n=$((n+1)); }; done
echo "    $n files deleted"
echo " [3/4] Deleting folders"
n=0; for d in "${TODO_D[@]}"; do [ -d "$d" ] && rm -rf -- "$d" && { echo "    deleted $d"; n=$((n+1)); }; done
echo "    $n folders deleted"

echo " [4/4] Checks (all must say YES)"
bad=0
chk() { if eval "$2"; then printf "    %-28s YES\n" "$1"; else printf "    %-28s NO\n" "$1"; bad=1; fi; }
chk "code kept"               "[ -f main.py ] && [ -f holoqpi/config.py ]"
chk "run_v2.sh kept"          "[ -f run_v2.sh ]"
chk "new results kept"        "[ -f runs/benchmark_results/results_arm_A.json ]"
chk "same-field table kept"   "[ -f runs/common_fields/common_fields_summary.csv ]"
chk "checkpoints kept"        "[ -f runs/v2_baseline_off_axis/best_model.pt ]"
chk "new figures kept"        "[ -f figures/fig05_modality_comparison.png ]"
chk "amplitude reference kept" "[ -d data/amplitude_reference ]"
chk "aberration.json kept"    "[ -f data/aberration.json ]"
chk "study logs kept"         "[ -d logs/corrected_20260930_054214 ]"
chk "split kept"              "[ -f data/splits.json ]"
echo "==========================================================================="
if [ "$bad" -eq 0 ]; then echo " Finished. Folder is clean."; else echo " Finished with problems: send this output to Claude."; fi
echo "==========================================================================="
