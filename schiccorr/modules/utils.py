import argparse
import sys
from pathlib import Path
import pandas as pd
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
	"""Save a DataFrame to disk in the specified format."""
	if fmt == "parquet":
		df.to_parquet(filename, index=False)
	elif fmt == "csv.gz":
		df.to_csv(filename, index=False, compression="gzip")


def configure_logger(level: str) -> None:
	"""Configure loguru logger with stdout for normal logs and stderr for errors."""
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


def create_argument_parser() -> argparse.ArgumentParser:
	"""Create and return the argument parser for the Hi-C correlation analysis."""
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

	return parser
