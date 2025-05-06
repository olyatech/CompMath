import numpy as np
import matplotlib.pyplot as plt
import time

from numba import jit

@jit
def richardson_method(A, b, tau, max_iter, tol):
    """
    Solve a linear system Ax = b using the Richardson iteration method.

    Parameters:
    - A (np.ndarray): A symmetric positive definite matrix of shape (n, n).
    - b (np.ndarray): The right-hand side vector of shape (n,).
    - tau (float): The iteration parameter.
    - max_iter (int): The maximum number of iterations.
    - tol (float): The tolerance for the stopping criterion based on the residual norm.

    Returns:
    - x (np.ndarray): The approximate solution vector of shape (n,).
    - residuals (np.ndarray): An array containing the residual norms at each iteration.
    """
    x = np.zeros_like(b, dtype=np.float64)
    residuals = []

    for i in range(max_iter):
        residual = b - np.dot(A, x)
        residual_norm = np.linalg.norm(residual)
        residuals.append(residual_norm)

        if residual_norm < tol:
            break

        if np.isinf(residual_norm):
            print("inf")
            break

        # x_{k+1} = x_k + tau (b - Ax_k)
        x = x + tau * residual

    return x, np.array(residuals)

def gershgorin_circles(A):
    """
    Estimate the eigenvalues of a matrix using Gershgorin circles.

    Parameters:
    - A (np.ndarray): A square matrix of shape (n, n).

    Returns:
    - min_gershgorin (np.ndarray): Array of lower bounds for eigenvalues.
    - max_gershgorin (np.ndarray): Array of upper bounds for eigenvalues.
    """
    centers = np.diag(A)
    radii = np.sum(np.abs(A), axis=1) - np.abs(centers)

    return centers - radii, centers + radii

def tau_values(A):
    """
    Compute different tau values for the Richardson method.

    Parameters:
    - A (numpy.ndarray): A symmetric positive definite matrix of shape (n, n).

    Returns:
    - tau_values (dict): A dictionary containing different tau values.
    """
    eigenvalues = np.linalg.eigvals(A)
    min_gershgorin, max_gershgorin = gershgorin_circles(A)

    return {
        "random": np.random.uniform(0, 2 / eigenvalues.max()),
        "estimated_optimal": 2 / (max(min_gershgorin.min(), 0) + max_gershgorin.max()),  # Optimal value based on estimates
        "exact_optimal": 2 / (eigenvalues.min() + eigenvalues.max())  # Optimal value based on exact eigenvalues
    }

def plot_convergence(results, tau_values):
    """
    Plot the convergence of the Richardson method for different tau values.

    Parameters:
    - A (numpy.ndarray): A symmetric positive definite matrix of shape (n, n).
    - b (numpy.ndarray): The right-hand side vector of shape (n,).
    - tau_values (dict): A dictionary containing different tau values.
    - max_iter (int): The maximum number of iterations.
    - tol (float): The tolerance for the stopping criterion based on the residual norm.
    """
    plt.figure(figsize=(12, 8))
    for name, tau in tau_values.items():
        plt.semilogx(results[name]["residuals"], label=f"{name} (tau={tau:.5f})"[:50])

    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence of Richardson Method')
    plt.legend()
    plt.grid(True)

    return plt

def analysis():
    # Init matrices
    np.random.seed(4)
    n = 100
    A = np.random.rand(n, n)
    A = A @ A.T  # A = A^T > 0
    b = np.random.rand(n)

    # Tau values
    A_tau_values = tau_values(A)

    # Richardson method parameters
    max_iter = 1e6
    tol = 1e-8

    # Exact solution
    start_time = time.time()
    exact_solution = np.linalg.solve(A, b)
    exact_time = time.time() - start_time

    # Solutions with Richardson method
    results = {}
    for name, tau in A_tau_values.items():
        start_time = time.time()
        solution, residuals = richardson_method(A, b, tau, max_iter, tol)
        end_time = time.time()
        results[name] = {
            "solution": solution,
            "residuals": residuals,
            "time": end_time - start_time
        }

    errors = {name: np.linalg.norm(results[name]["solution"] - exact_solution) for name in results}

    # Execution times
    execution_times = {name: results[name]["time"] for name in results}
    execution_times["exact_solution"] = exact_time

    # Plots
    plt = plot_convergence(results, A_tau_values)
    plt.show()

    print("errors:", errors)
    print("execution times:", execution_times)

if __name__ == "__main__":
    analysis()