import pytest
import numpy as np

from linear_system_solver import matrix_multiplication

# matrix-matrix multiplication tests

def test_matrix_multiplication_basic():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = matrix_multiplication(A, B)
    expected_result = np.dot(A, B)
    assert np.allclose(result, expected_result), "Test failed for basic case"

def test_matrix_multiplication_identity():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    I = np.eye(3)
    result = matrix_multiplication(A, I)
    assert np.allclose(result, A), "Test failed for identity matrix"

def test_matrix_multiplication_non_square():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])
    result = matrix_multiplication(A, B)
    expected_result = np.dot(A, B)
    assert np.allclose(result, expected_result), "Test failed for non-square matrices"

def test_matrix_multiplication_empty():
    A = np.empty((0, 3))
    B = np.empty((3, 0))
    result = matrix_multiplication(A, B)
    expected_result = np.empty((0, 0))
    assert np.allclose(result, expected_result), "Test failed for empty matrices"

# tests with vectors

def test_matrix_vector_multiplication():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([7, 8, 9])
    result = matrix_multiplication(A, B)
    expected_result = np.dot(A, B)
    assert np.allclose(result, expected_result), "Test failed for matrix-vector multiplication"

def test_vector_matrix_multiplication():
    A = np.array([7, 8])
    B = np.array([[1, 2, 3], [4, 5, 6]])
    result = matrix_multiplication(A, B)
    expected_result = np.dot(A, B)
    assert np.allclose(result, expected_result), "Test failed for vector-matrix multiplication"

# linalg errors tests

def test_matrix_multiplication_dimension_mismatch():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8], [9, 10]])
    with pytest.raises(np.linalg.LinAlgError):
        matrix_multiplication(A, B)

def test_matrix_vector_multiplication_dimension_mismatch():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([5, 6, 7, 8, 9, 10])
    with pytest.raises(np.linalg.LinAlgError):
        matrix_multiplication(A, B)

def test_vector_matrix_multiplication_dimension_mismatch():
    A = np.array([1, 2, 3, 4])
    B = np.array([[5, 6], [7, 8]])
    with pytest.raises(np.linalg.LinAlgError):
        matrix_multiplication(A, B)

# Run the tests
if __name__ == "__main__":
    pytest.main()