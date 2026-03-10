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
- --cores – Number of CPU cores for parallel processing (default: 1).
- --log-level – Logging level: DEBUG, INFO (default), WARNING, ERROR.

### Examples

Single Parquet file:

```bash
python main.py sample1.cool sample2.cool --output_prefix results/hic_corr --format parquet
```

Split by chromosome, compressed CSV:

```bash
python main.py sample1.cool sample2.cool --output_prefix results/hic_corr --format csv.gz --split
```

With debug logging piped to file:

```bash
python main.py sample1.cool sample2.cool --output_prefix results/hic_corr --log-level DEBUG > my_log.txt
```

With logging to both stdout and file:

```bash
python main.py sample1.cool sample2.cool --output_prefix results/hic_corr --log-file my_log.txt
```

## Output

- Parquet or CSV: Each row contains reference, comparison, chromosome, and correlation.
- When --split is used, each chromosome is saved in a separate file with the chromosome name appended to the prefix.
- Logs: Output to stdout by default (pipe to file as needed)

## Memory Optimization

This tool includes several optimizations to reduce RAM usage:

### Key Memory-Saving Features

- **Sequential processing**: Correlations are computed one chromosome at a time instead of loading all data simultaneously
- **Diagonal limiting**: The `--chunk-size` parameter limits the number of diagonals extracted (default: 1000)
- **Compressed storage**: Processed data is saved using `np.savez_compressed` for smaller file sizes
- **Early cleanup**: Matrices are deleted from memory as soon as they're no longer needed

### Recommended Settings for Large Datasets

```bash
# For memory-constrained systems
python main.py sample.cool --K 1000 --chunk-size 500 --cores 1

# For high-resolution data (small bins)
python main.py sample.cool --K 5000 --chunk-size 1000 --cores 2
```

### Memory Usage Estimation

- Each chromosome matrix: ~ (bins × bins) × 8 bytes (for float64)
- Diagonal storage: ~ (chunk_size × matrix_size) × 8 bytes per chromosome
- Total memory: ~ 2-3 × largest_chromosome_matrix during preprocessing

For a 10kb resolution genome with 3 billion base pairs:

- ~300,000 bins per chromosome
- Matrix size: ~90GB per chromosome (if loaded fully)
- With optimizations: ~1-2GB peak memory usage
