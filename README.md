# Single Cell Hi-C Correlation Tool
This tool computes per-chromosome correlations between Hi-C contact matrices stored in .cool files. It supports preprocessing, weighted correlations, and flexible output formats.

## Features
- Symmetrizes and normalizes Hi-C contact matrices.
- Applies a 2D mean filter to smooth matrices.
- Extracts the first K diagonals for analysis.
- Computes weighted correlations between multiple datasets.
- Supports compressed CSV (.csv.gz) and Parquet outputs.
- Optionally splits results by chromosome for easier handling.
- Parallel processing using multiple CPU cores.

## Installation
Clone the repository:
```bash
git clone git@github.com:srinzema/schic_correlate.git
cd schic_correlate
```

## Usage
```bash
python main.py input1.cool input2.cool --output_prefix results/output --format parquet --split --h 1 --K 5000000 --cores 4
```

### Arguments
- input_files – One or more .cool Hi-C input files.
- --output_prefix – Prefix for output files. Chromosome names will be appended if  --split is used.
- --format – Output format: parquet (default) or csv.gz.
- --split – Split output by chromosome into separate files.
- --h – Mean filter size (default: 1).
- --K – Number of diagonals to extract (default: 5,000,000).
- --cores – Number of CPU cores for parallel processing (default: 1

### Examples
Single Parquet file:
```bash
python main.py sample1.cool sample2.cool --output_prefix results/hic_corr --format parquet
```

Split by chromosome, compressed CSV:
```bash
python main.py sample1.cool sample2.cool --output_prefix results/hic_corr --format csv.gz --split
```

## Output
- Parquet or CSV: Each row contains reference, comparison, chromosome, and correlation.
- When --split is used, each chromosome is saved in a separate file with the chromosome name appended to the prefix.