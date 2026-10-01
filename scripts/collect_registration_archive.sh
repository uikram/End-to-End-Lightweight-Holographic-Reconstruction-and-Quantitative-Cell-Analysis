#!/usr/bin/env bash
# Locate (or regenerate) the Gabor-registration diagnostics used by Sec. 4.1 and place them in runs/diagnostics/.
#
#   cd /sda/usama/QPI_Extended_v3
#   bash scripts/collect_registration_archive.sh                 # search /sda/usama for an existing archive
#   bash scripts/collect_registration_archive.sh --regenerate    # re-run scripts/register_holograms.py (needs data/)
#
# Files: registration_summary.json, hologram_registration.csv, classical_phase_shift.csv, hologram_inventory.csv
# (written by scripts/register_holograms.py to <paths.output_root>/<output_dir> = runs/diagnostics/).
# Existing results are never overwritten: files are copied only if they are missing from runs/diagnostics/.
set -u
FILES="registration_summary.json hologram_registration.csv classical_phase_shift.csv hologram_inventory.csv"
DEST="runs/diagnostics"
SEARCH_ROOT="${SEARCH_ROOT:-/sda/usama}"
mkdir -p "$DEST"

missing() { for f in $FILES; do [ -f "$DEST/$f" ] || echo "$f"; done; }

if [ "${1:-}" = "--regenerate" ]; then
  echo "Regenerating with scripts/register_holograms.py (about 1600 registrations x 2 variants; --skip-classical omits classical_phase_shift.csv)"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python scripts/register_holograms.py --config config/base.yaml --device cuda
else
  for f in $(missing); do
    echo "searching for $f under $SEARCH_ROOT ..."
    # newest copy that sits in a 'diagnostics' directory (original archive or an older project folder)
    src=$(find "$SEARCH_ROOT" -path '*/diagnostics/*' -name "$f" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$src" ]; then echo "  found: $src"; cp -n "$src" "$DEST/$f"; else echo "  NOT FOUND"; fi
  done
  # unzip archives created by the earlier phase, if present
  for z in $(find "$SEARCH_ROOT" -maxdepth 3 \( -name 'diagnostics_results.zip' -o -name 'diagnostics_package.zip' \) 2>/dev/null); do
    echo "archive present: $z   (unzip -l $z | grep -E 'registration|inventory|phase_shift')"
  done
fi
echo; echo "status in $DEST:"
for f in $FILES; do [ -f "$DEST/$f" ] && echo "  OK       $f" || echo "  MISSING  $f"; done
echo; echo "Then: git add $DEST/{registration_summary.json,hologram_registration.csv,classical_phase_shift.csv,hologram_inventory.csv} && git commit && git push"
