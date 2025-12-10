from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import numba


@numba.njit
def weighted_correlation(diag1: np.ndarray, diag2: np.ndarray) -> Tuple[np.float64, np.float64]:
    """Compute the weighted correlation between two diagonals.

    Only non-zero entries in both diagonals are considered. Returns the correlation
    coefficient and a weight based on the number of contributing elements.

    Args:
    diag1: 1D NumPy array representing the first diagonal.
    diag2: 1D NumPy array representing the second diagonal.

    Returns:
    Tuple of (correlation, weight), both as floats.
    """
    mask = (diag1 != 0) & (diag2 != 0)
    n = mask.sum()
    if n == 0:
        return 0.0, 0.0  # corr, weight

    # Extract non-zero elements
    x = diag1[mask]
    y = diag2[mask]

    x_sum = np.float64(0.0)
    y_sum = np.float64(0.0)
    for i in range(n):
        x_sum += x[i]
        y_sum += y[i]
    
    x_mean = x_sum / n
    y_mean = y_sum / n

    cov = np.float64(0.0)
    x_var = np.float64(0.0)
    y_var = np.float64(0.0)
    for i in range(n):
        dx = x[i] - x_mean
        dy = y[i] - y_mean
        cov += dx * dy
        x_var += dx * dx
        y_var += dy * dy
    cov /= n
    x_std = np.sqrt(x_var / n)
    y_std = np.sqrt(y_var / n)

    if x_std == 0.0 or y_std == 0.0:
        corr = 0.0
    else:
        corr = cov / (x_std * y_std)

    weight = 0.0
    if n >= 2:
        weight = n * (1.0 + 1.0 / n) / 12.0

    return corr, weight


def compare(reference: Path, comparisons: List[Path]) -> Dict[Tuple[str, str, str], np.float64]:
    """Compute weighted correlations between per-chromosome data files.

    For each chromosome present in the reference directory, compares the corresponding
    files in one or more comparison directories. Extracts diagonals from the stored
    arrays, computes a weighted correlation for each diagonal, and combines them
    into a single weighted correlation per chromosome.

    Args:
    reference: Path to the directory containing reference .npz files.
    comparisons: List of directories with .npz files to compare against the reference.

    Returns:
    Dictionary mapping (reference_name, comparison_name, chromosome_name) to
    the computed weighted correlation as a float.
    """
    _chroms1 = list(reference.glob("*.npz"))
    results: Dict[Tuple[str, str, str], np.float64] = {}

    for comparison in comparisons:
        chrom_map2 = {p.stem: p for p in comparison.glob("*.npz")}

        for chrom1 in _chroms1:
            if chrom1.stem not in chrom_map2:
                print("error", chrom1.stem, chrom_map2)
                continue  # TODO raise error
            chrom2 = chrom_map2[chrom1.stem]

            with np.load(chrom1) as c1, np.load(chrom2) as c2:
                weighted_sum = np.float64(0.0)
                total_weight = np.float64(0.0)

                for k in range(len(c1.files)):
                    diag1 = np.ascontiguousarray(c1[f"arr_{k}"])
                    diag2 = np.ascontiguousarray(c2[f"arr_{k}"])
                    corr, weight = weighted_correlation(diag1, diag2)
                    weighted_sum += corr * weight
                    total_weight += weight

                correlation = weighted_sum / total_weight if total_weight > 0 else np.float64(0.0)
                results[(reference.stem, comparison.stem, chrom1.stem)] = correlation
    return results
