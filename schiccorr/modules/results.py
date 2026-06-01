import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCHEMA = pa.schema(
	[
		("reference", pa.string()),
		("comparison", pa.string()),
		("chromosome", pa.string()),
		("correlation", pa.float64()),
	]
)


def results_to_arrow(result_file: Path) -> pa.Table:
	"""Load a single pickle result file and convert to Arrow table."""
	with open(result_file, "rb") as f:
		scores: Dict[Tuple[str, str, str], np.float64] = pickle.load(f)

	keys = list(scores.keys())
	values = list(scores.values())

	return pa.table(
		{
			"reference": [k[0] for k in keys],
			"comparison": [k[1] for k in keys],
			"chromosome": [k[2] for k in keys],
			"correlation": np.round(values, 12),
		},
		schema=SCHEMA,
	)


def parquet_to_csv_gz(parquet_path: Path, csv_gz_path: Path) -> None:
	reader = pq.ParquetFile(parquet_path)
	first = True
	for batch in reader.iter_batches():
		batch.to_pandas().to_csv(
			csv_gz_path,
			index=False,
			header=first,
			mode="wb" if first else "ab",
			compression="gzip",
		)
		first = False


def save_results(
	result_files: List[Path], output_prefix: Path, fmt: str, split: bool = False
) -> List[Path]:
	is_csv = fmt == "csv.gz"

	# Always write parquet first
	parquet_files: List[Path] = []

	if not split:
		pq_path = Path(f"{output_prefix}.parquet")
		pq_path.parent.mkdir(parents=True, exist_ok=True)
		writer = pq.ParquetWriter(pq_path, SCHEMA)
		for result_file in result_files:
			writer.write_table(results_to_arrow(result_file))
		writer.close()
		parquet_files.append(pq_path)
	else:
		writers: Dict[str, pq.ParquetWriter] = {}
		pq_paths: Dict[str, Path] = {}

		for result_file in result_files:
			table = results_to_arrow(result_file)
			for chrom in pc.unique(table.column("chromosome")).to_pylist():  # type: ignore[attr-defined]
				chrom_table = table.filter(pc.equal(table.column("chromosome"), chrom))  # type: ignore[attr-defined]
				if chrom not in writers:
					pq_path = Path(f"{output_prefix}_{chrom}.parquet")
					pq_path.parent.mkdir(parents=True, exist_ok=True)
					pq_paths[chrom] = pq_path
					writers[chrom] = pq.ParquetWriter(pq_path, SCHEMA)
				writers[chrom].write_table(chrom_table)

		for chrom, writer in writers.items():
			writer.close()
		parquet_files.extend(pq_paths.values())

	if not is_csv:
		return parquet_files

	# Convert parquet to csv.gz and remove parquet files
	csv_files: List[Path] = []
	for pq_path in parquet_files:
		csv_path = pq_path.with_suffix("").with_suffix(".csv.gz")
		parquet_to_csv_gz(pq_path, csv_path)
		pq_path.unlink()
		csv_files.append(csv_path)

	return csv_files
