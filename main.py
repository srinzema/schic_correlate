import argparse, os, cooler, time, tempfile
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from modules import preprocess, correlate


def cool_file(path_str):
    path = Path(path_str)
    if not path.suffix == ".cool":
        raise argparse.ArgumentTypeError(f"File {path} is not a .cool file")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File {path} does not exist")
    return path


def main():
    parser = argparse.ArgumentParser(description="Hi-C correlation")
    
    parser.add_argument("input_files", type=cool_file, nargs="+", help="Hi-C .cool input files")
    parser.add_argument("--output_file", type=Path, required=True, help="File to save results")
    parser.add_argument("--h", type=int, default=1, help="Mean filter size for preprocessing")
    parser.add_argument("--K", type=int, default=5_000_000, help="Number of diagonals to extract")
    parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores for multiprocessing")
    
    args = parser.parse_args()
    max_workers = min(args.cores, os.cpu_count())

    binsize = cooler.Cooler(str(args.input_files[0])).binsize
    max_diagonal = args.K // binsize + 1

    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
    
        # Preprocess
        print(f"Preprocessing, using tempdir: {tmpdir}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(preprocess.preprocess_file, file, args.h, max_diagonal, tmpdir) for file in args.input_files]
            normalized_paths = [future.result() for future in futures]

        # Correlate
        print("Calculating correlation scores")
        start = time.time()
        scores = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for n, reference in enumerate(normalized_paths):
                comparisons = normalized_paths[n:]
                futures.append(
                    executor.submit(
                        correlate.compare, reference, comparisons
                    )
                )
        
        for n, future in enumerate(futures, start=1):
            scores.update(future.result())
            print(f"{n} completed ({time.time() - start:.2f} seconds)")

    df = pd.DataFrame(
        [(ref, comp, chrom, round(corr, 12)) for (ref, comp, chrom), corr in scores.items()],
        columns=["reference", "comparison", "chromosome", "correlation"]
    )

    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
