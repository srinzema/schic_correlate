import argparse, os, cooler, time, tempfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from modules import preprocess, correlate
from typing import List, Dict, Tuple
import numpy as np
import sys
import pickle
from loguru import logger


def cool_file(path_str: str) -> Path:
    """Return the path if it exists and has a .cool extension, else raise an error."""
    path = Path(path_str)
    if not path.suffix == ".cool":
        raise argparse.ArgumentTypeError(f"File {path} is not a .cool file")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {path} does not exist")
    return path


def save_df(df: pd.DataFrame, filename: Path, fmt: str) -> None:
    if fmt == "parquet":
        df.to_parquet(filename, index=False)
    elif fmt == "csv.gz":
        df.to_csv(filename, index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hi-C correlation")

    # Command-line arguments
    parser.add_argument(
        "input_files", type=cool_file, nargs="+", help="Hi-C .cool input files"
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Prefix for output files. Chromosome names will be appended if --split is used.",
    )
    parser.add_argument(
        "--format",
        choices=["csv.gz", "parquet"],
        default="parquet",
        help="Output format",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split output by chromosome into separate files",
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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )

    args = parser.parse_args()

    # Configure loguru: normal logs (DEBUG, INFO) to stdout, errors (WARNING+) to stderr
    level = args.log_level.upper()
    logger.remove()  # Remove default handler
    if level in ["DEBUG", "INFO"]:
        logger.add(
            sys.stdout,
            level=level,
            filter=lambda record: record["level"].no <= 20,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
    logger.add(
        sys.stderr,
        level=max(level, "WARNING"),
        filter=lambda record: record["level"].no >= 30,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    logger.info(
        f"Starting Hi-C correlation analysis with {len(args.input_files)} input files"
    )
    logger.info(
        f"Output prefix: {args.output_prefix}, Format: {args.format}, Split: {args.split}"
    )
    logger.info(f"Parameters: h={args.h}, K={args.K}, cores={args.cores}")

    # Determine number of workers for parallel processing
    max_workers: int = min(args.cores, os.cpu_count())
    logger.info(f"Using {max_workers} CPU cores for parallel processing")

    # Use a temporary directory to store per-chromosome processed data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logger.info(f"Using temporary directory: {tmpdir}")

        # Preprocess input files in parallel
        logger.info("Starting preprocessing of input files")
        try:
            normalized_paths: List[Path] = preprocess.preprocess_files(
                args.input_files, args.h, args.K, tmpdir, max_workers
            )
            logger.info(f"Preprocessing completed for {len(normalized_paths)} files")
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

        # Calculate pairwise correlations in parallel
        logger.info("Starting correlation calculation")
        start = time.time()
        result_files: List[Path] = []
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for n, reference in enumerate(normalized_paths):
                    comparisons = normalized_paths[n:]
                    future = executor.submit(correlate.compare, reference, comparisons)
                    futures.append(future)

                # Save results to temporary files as they complete
                total_complete = 0
                interval = len(futures) // 10 if len(futures) >= 10 else 1
                for future in as_completed(futures):
                    total_complete += 1
                    result_files.append(future.result())
                    # logger.info(f"Correlation batch {n} completed and saved")
                    if total_complete % interval == 0 or total_complete == len(futures):
                        logger.info(f"{total_complete}/{len(futures)} complete")

            logger.info(
                f"Correlation calculation completed in {time.time() - start:.2f} seconds"
            )
        except Exception as e:
            logger.error(f"Error during correlation calculation: {e}")
            raise

    # Load and combine all correlation results from temporary files
    logger.info("Loading and combining correlation results")
    scores: Dict[Tuple[str, str, str], np.float64] = {}
    for result_file in result_files:
        with open(result_file, "rb") as f:
            scores.update(pickle.load(f))

    keys = list(scores.keys())
    values = list(scores.values())

    df = pd.DataFrame(
        {
            "reference": [k[0] for k in keys],
            "comparison": [k[1] for k in keys],
            "chromosome": [k[2] for k in keys],
            "correlation": np.round(values, 12),
        }
    )
    logger.info(f"Created DataFrame with {len(df)} correlation entries")

    try:
        if args.split:
            # Loop over chromosomes and save one file per chromosome
            chromosomes = df["chromosome"].unique()
            logger.info(f"Splitting output by {len(chromosomes)} chromosomes")
            for chrom in chromosomes:
                filename = Path(f"{args.output_prefix}_{chrom}.{args.format}")
                filename.parent.mkdir(
                    parents=True, exist_ok=True
                )  # create directories if needed
                save_df(df[df["chromosome"] == chrom], filename, args.format)
                logger.info(f"Saved {filename}")
        else:
            # Save all results in a single file
            filename = Path(f"{args.output_prefix}.{args.format}")
            filename.parent.mkdir(
                parents=True, exist_ok=True
            )  # create directories if needed
            save_df(df, filename, args.format)
            logger.info(f"Saved {filename}")
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        raise


if __name__ == "__main__":
    main()
