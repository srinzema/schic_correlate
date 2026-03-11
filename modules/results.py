import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from modules import utils


def save_results(
    result_files: List[Path], output_prefix: Path, fmt: str, split: bool = False
) -> None:
    """Load correlation results, create DataFrame, and save to disk.

    Can save all results in a single file or split by chromosome into separate files.

    Args:
    result_files: List of paths to pickle files containing correlation results.
    output_prefix: Path prefix for output files.
    fmt: Output format ('parquet' or 'csv.gz').
    split: If True, split output by chromosome into separate files.
    """
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

    if split:
        # Loop over chromosomes and save one file per chromosome
        chromosomes = df["chromosome"].unique()
        logger.info(f"Splitting output by {len(chromosomes)} chromosomes")
        for chrom in chromosomes:
            filename = Path(f"{output_prefix}_{chrom}.{fmt}")
            filename.parent.mkdir(parents=True, exist_ok=True)
            utils.save_df(df[df["chromosome"] == chrom], filename, fmt)
            logger.info(f"Saved {filename}")
    else:
        # Save all results in a single file
        filename = Path(f"{output_prefix}.{fmt}")
        filename.parent.mkdir(parents=True, exist_ok=True)
        utils.save_df(df, filename, fmt)
        logger.info(f"Saved {filename}")
    logger.info("Analysis completed successfully")
