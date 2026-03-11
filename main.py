import argparse, os, cooler, time, tempfile
import pandas as pd
from pathlib import Path
from modules import preprocess, correlate, utils
from typing import List, Dict, Tuple
import numpy as np
import sys
import pickle
from loguru import logger


def main() -> None:
    parser = utils.create_argument_parser()
    args = parser.parse_args()

    # Configure loguru
    utils.configure_logger(args.log_level.upper())

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
        try:
            result_files: List[Path] = correlate.compare_pairwise(
                normalized_paths, max_workers
            )
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
                utils.save_df(df[df["chromosome"] == chrom], filename, args.format)
                logger.info(f"Saved {filename}")
        else:
            # Save all results in a single file
            filename = Path(f"{args.output_prefix}.{args.format}")
            filename.parent.mkdir(
                parents=True, exist_ok=True
            )  # create directories if needed
            utils.save_df(df, filename, args.format)
            logger.info(f"Saved {filename}")
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        raise


if __name__ == "__main__":
    main()
