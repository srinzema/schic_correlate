import argparse, os, cooler, time, tempfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from modules import preprocess, correlate
from typing import List, Dict, Tuple
import numpy as np


def cool_file(path_str: str) -> Path:
    """Return the path if it exists and has a .cool extension, else raise an error."""
    path = Path(path_str)
    if not path.suffix == ".cool":
        raise argparse.ArgumentTypeError(f"File {path} is not a .cool file")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {path} does not exist")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hi-C correlation")

    # Command-line arguments
    parser.add_argument(
        "input_files", type=cool_file, nargs="+", help="Hi-C .cool input files"
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="File to save results"
    )
    parser.add_argument(
        "--h", type=int, default=1, help="Mean filter size for preprocessing"
    )
    parser.add_argument(
        "--K", type=int, default=5_000_000, help="Number of diagonals to extract"
    )
    parser.add_argument(
        "--cores", type=int, default=1, help="Number of CPU cores for multiprocessing"
    )

    args = parser.parse_args()

    # Determine output format
    output_file = Path(args.output_file)
    if output_file.suffix == ".parquet":
        output_format = "parquet"
    elif output_file.suffixes[-2:] == [".csv", ".gz"]:
        output_format = "csv.gz"
    else:
        raise ValueError("Output file must end with .parquet or .csv.gz")

    # Determine number of workers for parallel processing
    max_workers: int = min(args.cores, os.cpu_count())

    # Compute binsize and max_diagonal for preprocessing
    binsize: int = cooler.Cooler(str(args.input_files[0])).binsize
    max_diagonal: int = args.K // binsize + 1

    # Use a temporary directory to store per-chromosome processed data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Preprocess input files in parallel
        print(f"Preprocessing, using tempdir: {tmpdir}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    preprocess.preprocess_file, file, args.h, max_diagonal, tmpdir
                )
                for file in args.input_files
            ]
            normalized_paths: List[Path] = [future.result() for future in futures]

        # Calculate pairwise correlations in parallel
        print("Calculating correlation scores")
        start = time.time()
        scores: Dict[Tuple[str, str, str], np.float64] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for n, reference in enumerate(normalized_paths):
                comparisons = normalized_paths[n:]
                futures.append(
                    executor.submit(correlate.compare, reference, comparisons)
                )

        # Collect results as they complete
        for n, future in enumerate(futures, start=1):
            scores.update(future.result())
            print(f"{n} completed ({time.time() - start:.2f} seconds)")

    df = pd.DataFrame(
        {
            "reference": [ref for ref, _, _ in scores.keys()],
            "comparison": [comp for _, comp, _ in scores.keys()],
            "chromosome": [chrom for _, _, chrom in scores.keys()],
            "correlation": [round(corr, 12) for corr in scores.values()],
        }
    )
    if output_format == "parquet":
        df.to_parquet(output_file, index=False)
    elif output_format == "csv.gz":
        df.to_csv(output_file, index=False, compression="gzip")


if __name__ == "__main__":
    main()
