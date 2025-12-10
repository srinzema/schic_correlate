from pathlib import Path
from typing import List
import numpy as np
import cooler
import numba


@numba.njit
def mean_filter(mat: np.ndarray, h: int) -> np.ndarray:
    """
    Apply a 2D mean filter to a matrix.

    Replaces each entry with the average of values in a square neighborhood of
    radius h around it, clipped at matrix boundaries.

    Args:
    mat: 2D NumPy array to be filtered.
    h: Neighborhood radius in both dimensions.

    Returns:
    Filtered matrix with the same shape and dtype as the input.
    """
    rows, cols = mat.shape
    output = np.zeros((rows, cols), dtype=mat.dtype)
    
    for x in range(rows):
        for y in range(cols):
            x_min = max(x - h, 0)
            x_max = min(x + h + 1, rows)
            y_min = max(y - h, 0)
            y_max = min(y + h + 1, cols)

            n = 0
            neighborhood = 0.0
            for i in range(x_min, x_max):
                for j in range(y_min, y_max):
                    neighborhood += mat[i, j]
                    n += 1
            output[x, y] = neighborhood / n
    return output


@numba.njit
def unstack(mat: np.ndarray, K: int) -> List[np.ndarray]:
    """Extract the first K diagonals from a matrix.

    Collects the main diagonal and the next K−1 upper diagonals, each stored as a
    1D array with length limited by the matrix bounds.

    Args:
    mat: 2D NumPy array to extract diagonals from.
    K: Number of diagonals to extract, starting from the main diagonal.

    Returns:
    List of 1D NumPy arrays corresponding to successive upper diagonals.
    """

    rows, cols = mat.shape
    diagonals = []
    
    for k in range(K):
        length = min(rows, cols - k)
        diag = np.zeros(length, dtype=mat.dtype)

        for i in range(length):
            diag[i] = mat[i, i + k]
        diagonals.append(diag)
    return diagonals


def preprocess_file(file: Path, h: int, K: int, temp_dir: Path = None) -> Path:
    """Load a cooler file and write per-chromosome processed contact data.

    For each chromosome, the contact matrix is symmetrized, normalized by the total
    contact count, smoothed with a mean filter, the first K diagonals are extracted,
    and the result is saved as a NumPy archive in a temporary directory.

    Args:
    file: Path to the input .cool or .mcool file.
    h: Window size parameter for the mean filter.
    K: Number of diagonals to extract from each contact matrix.
    temp_dir: Base directory where intermediate per-chromosome results
    will be written.

    Returns:
    Path to the directory containing the saved per-chromosome .npz files.
    The temporary directory structure would be tmpdir/cell/chromosome/file
    """

    temp_dir = temp_dir / file.stem
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    clr: cooler.Cooler = cooler.Cooler(str(file))
    total_contacts = clr.matrix(balance=False)[:].sum()
    if total_contacts == 0 or np.isnan(total_contacts):
        total_contacts = 1  # avoid division by zero

    for chrom in clr.chromnames:
        # Load and symmetrize chrom matrix
        mat: np.ndarray = clr.matrix(balance=False, as_pixels=False, join=True).fetch(chrom)
        mat_t = mat.copy().T
        np.fill_diagonal(mat_t, 0)
        mat = mat + mat_t

        # Normalize, filter, unstack, and save
        mat = mat / total_contacts
        mat = mean_filter(mat, h)
        diagonals = unstack(mat, K)
        np.savez(temp_dir / f"{chrom}.npz", *diagonals)

    return temp_dir