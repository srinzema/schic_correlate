# scHicCorr

Computes per-chromosome correlations between single-cell Hi-C contact matrices stored in `.cool` files. Supports preprocessing, weighted correlations, and flexible output formats.

## Features

- Symmetrizes and normalizes Hi-C contact matrices
- Applies a 2D mean filter to smooth matrices
- Extracts the first K diagonals for analysis
- Computes weighted correlations between multiple datasets
- Outputs compressed CSV (`.csv.gz`) or Parquet
- Optionally splits results per chromosome
- Parallel processing via multiple CPU cores

## Installation

```bash
pip install scHicCorr
```

## Usage

```bash
schiccorr input1.cool input2.cool --output_prefix results/output --format parquet --split --h 1 --K 5000000 --cores 4
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `input_files` | One or more `.cool` Hi-C input files | |
| `--output_prefix` | Prefix for output files (chromosome name appended if `--split`) | |
| `--format` | Output format: `parquet` or `csv.gz` | `parquet` |
| `--split` | Split output by chromosome into separate files | |
| `--h` | Mean filter size | `1` |
| `--K` | Number of diagonals to extract | `5000000` |
| `--cores` | Number of CPU cores for parallel processing | `1` |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--log-file` | Write logs to file in addition to stdout | |

### Examples

Single Parquet file:

```bash
schiccorr sample1.cool sample2.cool --output_prefix results/hic_corr
```

Split by chromosome, compressed CSV:

```bash
schiccorr sample1.cool sample2.cool --output_prefix results/hic_corr --format csv.gz --split
```

Log to file:

```bash
schiccorr sample1.cool sample2.cool --output_prefix results/hic_corr --log-file run.log
```

## Output

Each row in the output contains `reference`, `comparison`, `chromosome`, and `correlation`. When `--split` is used, one file is written per chromosome with the chromosome name appended to the prefix.
