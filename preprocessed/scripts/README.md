# Preprocessing Job Scripts

This directory contains preprocessing job scripts designed to run in a controlled environment. By default, each
script derives the following paths from its own location:

```text
CRCEHR root = preprocessed/..
raw data    = CRCEHR root/rawdata
output      = CRCEHR root/preprocessed
```

No SSH, SCP, or machine-specific absolute paths are required.

## Data Processing Jobs

### `build_inhouse_key_values.py`

Materializes the raw structured In-house CSV files as split-aligned JSONL according to
`preprocessed/inhouse/*_registration.json`:

```bash
cd path/to/repo/preprocessed
python3 scripts/build_inhouse_key_values.py
```

By default, the script reads from `../rawdata/In-house` and writes to `inhouse/`.

### `build_ehrcon_gold.py`

Filters entity-level consistency gold records from the EHRCon annotations and aggregates them by note:

```bash
python3 scripts/build_ehrcon_gold.py
```

By default, the script reads from `../rawdata/EHRCon/ehrcon-1.0.1` and writes to `ehrcon/`.

### `rebuild_inhouse_registrations.py`

Normalizes field names and generates patient-level train, upstream-test, and downstream-test registrations:

```bash
python3 scripts/rebuild_inhouse_registrations.py
```

This script changes the splits and registrations and creates a backup directory. It is not required for standard
training or evaluation. Run it only when the data splits must be rebuilt; afterward, regenerate the In-house
key-value files and check for patient overlap across splits.

## Metadata Tools

### `generate_ehrcon_metadata.py`

Generates an EHRCon metadata workbook from `ehrcon/ehrcon_extraction_gold_field_stats.json` and a provided Excel
template:

```bash
python3 scripts/generate_ehrcon_metadata.py \
  --template /path/to/ColoReason.xlsx \
  --stats ehrcon/ehrcon_extraction_gold_field_stats.json \
  --output /path/to/EHRCon.xlsx
```

### `update_coloreason_metadata.py`

Updates the specified ColoReason metadata workbook:

```bash
python3 scripts/update_coloreason_metadata.py --workbook /path/to/ColoReason.xlsx
```

The release repository does not include workbooks, raw data, or preprocessed artifacts. When running the Excel
utilities, explicitly provide paths to the external template and statistics file.
