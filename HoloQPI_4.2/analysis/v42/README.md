# analysis/v42

These scripts produce and check every number in HoloQPI_4.2. They are meant to sit at `<repository>/analysis/v42/` and to be run from the repository root, because they read `runs/`, `logs/` and `config/`. The copies in `Claude outputs/HoloQPI_4.2/analysis/v42/` are for reference.

1. `python analysis/v42/extract_values.py`
   - Reads the corrected result files and writes `values.json`, with one entry per value: file, field, and formula for derived values.
   - Checks `runs/*/metrics_test.json` against the collector's `benchmark_results`, including the differences and thresholds.
2. `python analysis/v42/make_tables_v42.py --out "Claude outputs/HoloQPI_4.2"`
   - Writes tables 2–13.
   - Records each printed cell and its key in `table_cells.csv`.
3. `python analysis/v42/check_numbers.py --tex "Claude outputs/HoloQPI_4.2"`
   - Checks every number in the text and in the tables against `values.json`, a derived calculation, or a listed method parameter.
   - Writes `number_check.csv` with the columns: location, value, source file, key or formula, inputs, match.
4. `python analysis/v42/make_changelog_numbers.py > analysis/v42/changelog_numbers.md` writes the v4.1 → 4.2 number table used in CHANGELOG.md.

Resolution rule: a difference is resolved when |Δ| > 2·sqrt((s1² + s2²)/2), using the between-seed SD with ddof = 1. When either side has fewer than three runs, the comparison is not resolvable.
