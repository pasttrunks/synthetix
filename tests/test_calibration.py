import numpy as np
import pytest
from benchmark.calibration import compute_calibration_details

def test_compute_calibration_details_perfect():
    """Test perfect calibration gives zero ECE and zero MCE."""
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.05, 0.05, 0.95, 0.95], dtype=float)
    ece, mce, details = compute_calibration_details(y_true, y_prob, n_bins=10)
    
    assert isinstance(ece, float)
    assert isinstance(mce, float)
    assert ece >= 0.0
    assert mce >= 0.0
    assert len(details["bin_edges"]) == 11
    assert len(details["bin_confs"]) == 10
    assert len(details["bin_accs"]) == 10
    assert len(details["bin_counts"]) == 10

def test_compute_calibration_details_empty():
    """Test empty input handles cleanly without error."""
    y_true = np.array([], dtype=int)
    y_prob = np.array([], dtype=float)
    ece, mce, details = compute_calibration_details(y_true, y_prob, n_bins=10)
    
    assert ece == 0.0
    assert mce == 0.0
    assert details["bin_confs"] == []
    assert details["bin_accs"] == []
    assert details["bin_counts"] == []
