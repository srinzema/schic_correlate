import os
import tempfile
from pathlib import Path
from typing import List
from .modules import preprocess, correlate, utils, results
from loguru import logger


def main() -> None:
    parser = utils.create_argument_parser()
    args = parser.parse_args()

    # Configure loguru
    utils.configure_logger(args.log_level.upper())

    # Log analysis parameters
    logger.info(f"Starting analysis with {len(args.input_files)} input files")
    logger.info(f"Output: {args.output_prefix}.{args.format} (split: {args.split})")
    logger.info(f"Parameters: h={args.h}, K={args.K}, cores={args.cores}")

    # Determine number of workers for parallel processing
    num_workers: int = min(args.cores, os.cpu_count())
    logger.info(f"Using {num_workers} CPU cores for parallel processing")

    # Use a temporary directory to store per-chromosome processed data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        logger.info(f"Using temporary directory: {tmpdir}")

        # Preprocess input files in parallel
        logger.info("Starting preprocessing of input files")
        try:
            normalized_paths: List[Path] = preprocess.preprocess_files(
                args.input_files,
                args.h,
                args.K,
                tmpdir,
                num_workers,
            )
            logger.info(f"Preprocessing completed for {len(normalized_paths)} files")
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

        # Calculate pairwise correlations in parallel
        logger.info("Starting correlation calculation")
        try:
            result_files: List[Path] = correlate.compare_pairwise(
                normalized_paths,
                num_workers,
            )
            logger.info("Correlation calculation completed")
        except Exception as e:
            logger.error(f"Error during correlation calculation: {e}")
            raise

        # Load and combine all correlation results, create and save DataFrame
        try:
            logger.info("Loading and combining correlation results")
            filenames = results.save_results(
                result_files,
                args.output_prefix,
                args.format,
                args.split,
            )
            for file in filenames:
                logger.info(f"Saved results to {file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
