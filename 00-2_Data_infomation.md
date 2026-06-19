# sequence_pattern_structures
Sequential Pattern Structures for Predicting Mortality in Oncology Patients 

Formal definitions of the current feature, state, and sequence pattern structures are documented in:

- `notes/formal_pattern_structure_definition.md`
- `notes/neighborhood_mining_experiment.md`

## Data preparation

Build the single-snapshot SPC/IPS dataset with:

```bash
python scripts/prepare_single_snapshot_dataset.py
```

This writes:

- `prepared/single_snapshot_120d/patient_snapshots.csv`
- `prepared/single_snapshot_120d/patient_states.csv`
- `prepared/single_snapshot_120d/patient_trajectories.jsonl`
- `prepared/single_snapshot_120d/pattern_structure_metadata.json`
- `prepared/single_snapshot_120d/validation_report.json`

Build the breast-cancer-only variant with stricter recent-support filtering with:

```bash
python scripts/prepare_breast_cancer_supported_single_snapshot_dataset.py
```

This writes:

- `prepared/breast_cancer_single_snapshot_120d_last_window_supported/patient_snapshots.csv`
- `prepared/breast_cancer_single_snapshot_120d_last_window_supported/patient_states.csv`
- `prepared/breast_cancer_single_snapshot_120d_last_window_supported/patient_trajectories.jsonl`
- `prepared/breast_cancer_single_snapshot_120d_last_window_supported/pattern_structure_metadata.json`
- `prepared/breast_cancer_single_snapshot_120d_last_window_supported/validation_report.json`

## Generated files

The same file layout is used for both prepared datasets. The breast-cancer-only variant adds one extra static feature:

- `cancer_subtype_code`: exact ICD-10 diagnosis code inside the `C50.*` family

Important representation rule for numeric pattern-structure features:

- missing numeric values are exported as the **top interval** of that feature
- the corresponding `*_missing` column is kept as a provenance marker so you can tell that the top interval came from missingness

### `patient_snapshots.csv`

One row per patient. This is the supervision table for the single-snapshot task.

Main columns:

- `patient_id`: patient identifier
- `diagnosis_date`: timeline anchor, `t = 0`
- `snapshot_date`: exact prediction date, defined as `max_date - 120 days`
- `max_date`: patient-specific observation end
- `death_date`: death date if available
- `will_die_next_120_days`: target label for the snapshot
- `age_bin`, `gender`, `cancer_type`, `stage_normalized`, `metastasis_flag`: static context
- `cancer_subtype_code`: exact diagnosis subtype code for the cohort-specific variant
- `window_count`: number of 30-day states included in the trajectory

Use this file when you need:

- labels for evaluation or supervised experiments
- cohort-level summaries
- a join target for `patient_states.csv` or `patient_trajectories.jsonl`

### `patient_states.csv`

One row per patient per 30-day window from diagnosis until the snapshot.
Windows are built forward from diagnosis. The final window may be shorter than 30 days so that the sequence ends exactly at `snapshot_date`.

Main columns:

- identifiers: `patient_id`, `window_index`, `window_start`, `window_end`
- static context repeated on each row: `age_bin`, `gender`, `cancer_type`, `stage_normalized`, `metastasis_flag`
- cohort-specific static context when available: `cancer_subtype_code`
- IPS-style lab columns for each selected test:
  - `*_min`
  - `*_max`
  - `*_missing`
- dynamic non-lab state:
  - `ecog_min`
  - `ecog_max`
  - `ecog_missing`
  - `chemo_active`
  - `radiotherapy_active`
  - `admissions` as an exact non-negative count

For labs and ECOG:

- if a value is observed in the window, `min` and `max` are the observed interval
- if a value is missing in the window, `min` and `max` are filled with the feature's top interval and `*_missing = 1`

Use this file when you need:

- a tabular representation for inspection and debugging
- to export states into another mining or ML pipeline
- to verify that windowed aggregation is correct before sequence mining

### `patient_trajectories.jsonl`

One JSON object per patient. This is the sequence-oriented representation for SPC-style mining.

Each object contains:

- `patient_id`
- `static_context`
- `states`: ordered list of window states from diagnosis to snapshot

For the breast-cancer-only variant, `static_context` also includes `cancer_subtype_code`.

Each state contains:

- `window_index`, `window_start`, `window_end`
- `labs`: per-test interval plus missing provenance marker
- `ecog`: interval plus missing provenance marker
- `chemo_active`
- `radiotherapy_active`
- `admissions`

Use this file when you need:

- the actual trajectory object for sequential pattern mining
- a patient-independent sequence representation
- to score mined patterns later by joining back to `patient_snapshots.csv`

### `validation_report.json`

Quality-control report produced after preparation.

It includes:

- cohort size and class balance
- sequence length summary
- leakage and label-validity checks
- aggregation statistics for labs, ECOG, treatments, admissions, states, and trajectories

Use this file to confirm that a run is valid before using the generated dataset.

### `pattern_structure_metadata.json`

Metadata describing the top elements used in the exported pattern-structure representation.

It includes:

- top intervals for all numeric lab features
- top interval for ECOG
- the top element for admissions
- notes explaining that top intervals are used for mathematical correctness and `*_missing` is the provenance marker

## Recommended workflow

1. Run `python scripts/prepare_single_snapshot_dataset.py`.
2. Check `validation_report.json` to confirm the build passed.
3. Use `patient_trajectories.jsonl` as the main SPC input.
4. Use `patient_states.csv` when you want a flat state table for inspection or conversion.
5. Use `patient_snapshots.csv` to attach labels to mined patterns after mining, not inside the state representation itself.
6. Use `pattern_structure_metadata.json` whenever you need to interpret top intervals or distinguish mathematically introduced top values from observed intervals.

## Cohort variants

### `single_snapshot_120d`

- adults with a diagnosis date
- exact snapshot at `max_date - 120 days`
- no recent-support filter
- all cancer types

### `breast_cancer_single_snapshot_120d_last_window_supported`

- adults with earliest diagnosis code in the `C50.*` family
- exact snapshot at `max_date - 120 days`
- patient is kept only if the last included window has at least one signal:
  - any selected lab present
  - or ECOG present
  - or active chemo/radiotherapy
  - or admission count > 0
- `cancer_type` is fixed to `breast_cancer`
- `cancer_subtype_code` keeps the exact ICD subtype such as `C50.4`, `C50.8`, etc.

## Neighborhood Mining PoC

Run the neighborhood-based proof-of-concept miner with:

```bash
python scripts/mine_neighborhood_interval_patterns.py --length 2 --k 10
python scripts/mine_neighborhood_interval_patterns.py --length 3 --k 10
```

Default assumptions:

- cohort: `prepared/breast_cancer_single_snapshot_120d_last_window_supported`
- seeds: positive suffix subsequences
- lengths handled separately
- features:
  - `albumin`, `hb`, `wbc`, `creatinine`, `total_bilirubin`, `ecog`, `admissions`, `chemo_active`
- neighborhood distance:
  - range-normalized interval midpoint + width distance for numeric interval features
  - normalized absolute difference for admissions
  - exact mismatch distance for binary features

Outputs are written to:

- `results/neighborhood_patterns/<cohort_name>/length_2_k10/`
- `results/neighborhood_patterns/<cohort_name>/length_3_k10/`

Each run writes:

- `patterns.jsonl`: full machine-readable patterns
- `patterns_top.csv`: concise ranked summary
- `run_metadata.json`: exact parameters and baseline statistics

The experiment design and defaults are documented in:

- `notes/neighborhood_mining_experiment.md`
