import numpy as np
from schiccorr.modules.correlate import weighted_correlation


def test_perfect_correlation():
    """Test that identical arrays return a correlation of 1.0"""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    corr, weight = weighted_correlation(a, b)

    assert corr == pytest.approx(1.0)
    assert weight == pytest.approx(5 / 12)


def test_inverse_correlation():
    """Test that perfectly opposite arrays return -1.0"""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)

    corr, weight = weighted_correlation(a, b)

    assert corr == pytest.approx(-1.0)
    assert weight == pytest.approx(5 / 12)


def test_zero_variance():
    """Test that constant arrays (std=0) return correlation 0.0 instead of crashing"""
    a = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    corr, weight = weighted_correlation(a, b)

    assert corr == 0.0
    assert weight == pytest.approx(5 / 12)


def test_all_zeros():
    """Test that arrays with no overlap return (0.0, 0.0)"""
    a = np.zeros(10, dtype=np.float64)
    b = np.zeros(10, dtype=np.float64)

    corr, weight = weighted_correlation(a, b)

    assert corr == 0.0
    assert weight == 0.0


def test_partial_overlap():
    """Test that only non-zero entries are considered"""
    # a has value at index 0 and 2
    # b has value at index 1 and 2
    # Mask should keep indices 0, 1, 2 (n=3)
    a = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 2.0, 3.0, 0.0], dtype=np.float64)

    # Manual check:
    # x = [1, 0, 3], y = [0, 2, 3]
    # means: x_m = 1.33, y_m = 1.66
    # weight for n=3: 3 * (1 + 1/3) / 12 = 3 * 1.33 / 12 = 4/12 = 0.333...

    corr, weight = weighted_correlation(a, b)

    assert weight == pytest.approx(4 / 12)
    assert -1.0 <= corr <= 1.0


def test_numba_types():
    """Ensure the function handles integer arrays by casting them internally"""
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 2, 3], dtype=np.int64)

    # This should not crash because of the np.float64 casts inside the function
    corr, weight = weighted_correlation(a, b)
    assert corr == pytest.approx(1.0)
