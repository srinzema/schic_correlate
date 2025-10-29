from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import pandas as pd
import numpy as np
import numba
import time

@numba.njit
def weighted_correlation(diag1, diag2):
    mask = (diag1 != 0) & (diag2 != 0)
    n = mask.sum()
    if n == 0:
        return 0.0, 0.0  # corr, weight

    x, y = diag1[mask], diag2[mask]
    xm, ym = x.mean(), y.mean()
    cov = np.mean((x - xm) * (y - ym))
    sx, sy = np.std(x), np.std(y)
    corr = 0.0 if sx == 0.0 or sy == 0.0 else cov / (sx * sy)

    weight = 0.0
    if n >= 2:
        weight = n * (1.0 + 1.0 / n) / 12.0

    return corr, weight


def compare(reference, comparisons):
    _chroms1 = list(reference.glob("*.npz"))
    results = {}

    for comparison in comparisons:
        chrom_map2 = {p.stem: p for p in comparison.glob("*.npz")}

        for chrom1 in _chroms1:
            if chrom1.stem not in chrom_map2:
                print("error", chrom1.stem, chrom_map2)
                continue  # TODO raise error
            chrom2 = chrom_map2[chrom1.stem]

            with np.load(chrom1) as c1, np.load(chrom2) as c2:
                weighted_sum = 0.0
                total_weight = 0.0

                for k in range(len(c1.files)):
                    diag1, diag2 = c1[f"arr_{k}"], c2[f"arr_{k}"]
                    corr, weight = weighted_correlation(diag1, diag2)
                    weighted_sum += corr * weight
                    total_weight += weight

                correlation = weighted_sum / total_weight if total_weight > 0 else 0.0
                results[(reference.stem, comparison.stem, chrom1.stem)] = correlation
                
    return results
