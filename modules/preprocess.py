from pathlib import Path
import numpy as np
import cooler
import numba


@numba.njit
def mean_filter(mat, h):
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
def unstack(mat, K):
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
    temp_dir = temp_dir / file.stem
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    clr = cooler.Cooler(str(file))
    total_contacts = clr.matrix(balance=False)[:].sum()
    if total_contacts == 0 or np.isnan(total_contacts):
        total_contacts = 1  # avoid division by zero

    for chrom in clr.chromnames:
        # Load and symmetrize chrom matrix
        mat = clr.matrix(balance=False, as_pixels=False, join=True).fetch(chrom)
        mat_t = mat.copy().T
        np.fill_diagonal(mat_t, 0)
        mat = mat + mat_t

        # Normalize, filter, unstack, and save
        mat = mat / total_contacts
        mat = mean_filter(mat, h)
        diagonals = unstack(mat, K)
        np.savez(temp_dir / f"{chrom}.npz", *diagonals)

    return temp_dir