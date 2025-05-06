import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.linalg import solve

def compute_b_coefficients(a, xl, xr):
    """
    Compute coefficients b_k for the best L2 approximation.
    
    Args:
        a: Array of coefficients for the original polynomial in phi basis
        xl: Left boundary of the interval
        xr: Right boundary of the interval
        
    Returns:
        Array of coefficients b for the psi basis approximation
    """
    # Define the basis functions
    def phi_functions(x):
        return np.array([1, np.log(x), x**-2, x**-1, x, x**2, x**3])
    
    def psi_functions(x):
        return np.array([1, x, np.sqrt(x), x**2, x**3, x**4, x**6])
    
    n = len(a)
    
    # Compute the Gram matrix for psi basis
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            def integrand(x):
                psi_i = psi_functions(x)[i]
                psi_j = psi_functions(x)[j]
                return psi_i * psi_j
            G[i, j], _ = quad(integrand, xl, xr)
    
    # Compute the right-hand side vector
    b_rhs = np.zeros(n)
    for i in range(n):
        def integrand(x):
            sum_phi = 0
            phi = phi_functions(x)
            for k in range(n):
                sum_phi += a[k] * phi[k]
            psi_i = psi_functions(x)[i]
            return sum_phi * psi_i
        b_rhs[i], _ = quad(integrand, xl, xr)
    
    # Solve the linear system to find b coefficients
    b = solve(G, b_rhs)
    return b

def evaluate_phi_polynomial(a, x_values):
    """
    Evaluate the original polynomial in phi basis at given points.
    
    Args:
        a: Array of coefficients for the phi basis
        x_values: Array of x values to evaluate at
        
    Returns:
        Array of function values
    """
    result = np.zeros_like(x_values)
    for x, idx in zip(x_values, range(len(x_values))):
        val = 0
        if x <= 0:
            continue  # phi functions are not defined for x <= 0
        val += a[0] * 1
        val += a[1] * np.log(x)
        val += a[2] * (x**-2)
        val += a[3] * (x**-1)
        val += a[4] * x
        val += a[5] * (x**2)
        val += a[6] * (x**3)
        result[idx] = val
    return result

def evaluate_psi_polynomial(b, x_values):
    """
    Evaluate the approximation polynomial in psi basis at given points.
    
    Args:
        b: Array of coefficients for the psi basis
        x_values: Array of x values to evaluate at
        
    Returns:
        Array of function values
    """
    result = np.zeros_like(x_values)
    for x, idx in zip(x_values, range(len(x_values))):
        if x <= 0:
            continue  # psi functions are not defined for x <= 0
        val = 0
        val += b[0] * 1
        val += b[1] * x
        val += b[2] * np.sqrt(x)
        val += b[3] * (x**2)
        val += b[4] * (x**3)
        val += b[5] * (x**4)
        val += b[6] * (x**6)
        result[idx] = val
    return result

def plot_approximation(a, b, xl, xr, num_points=1000):
    """
    Plot the original function, its approximation, and their difference.
    
    Args:
        a: Coefficients for original phi basis polynomial
        b: Coefficients for psi basis approximation
        xl: Left interval boundary
        xr: Right interval boundary
        num_points: Number of points for plotting
    """
    x_values = np.linspace(xl, xr, num_points)
    y_original = evaluate_phi_polynomial(a, x_values)
    y_approx = evaluate_psi_polynomial(b, x_values)
    
    plt.figure(figsize=(12, 8))
    
    # Plot original and approximation
    plt.subplot(2, 1, 1)
    plt.plot(x_values, y_original, label='Original function')
    plt.plot(x_values, y_approx, label='Approximation')
    plt.title(f'Function and its L2 approximation on [{xl}, {xr}]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Plot difference
    plt.subplot(2, 1, 2)
    plt.plot(x_values, y_original - y_approx, label='Difference', color='red')
    plt.title('Difference between original and approximation')
    plt.xlabel('x')
    plt.ylabel('Difference')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage and tests
if __name__ == "__main__":
    # Test case 1: Simple polynomial that can be exactly represented in both bases
    a_test1 = np.array([0, 0, 0, 0, 1, 0, 0])  # f(x) = x
    xl1, xr1 = 1, 2
    b_test1 = compute_b_coefficients(a_test1, xl1, xr1)
    print("Test 1 coefficients:", b_test1)
    plot_approximation(a_test1, b_test1, xl1, xr1)
    
    # Test case 2: More complex function
    a_test2 = np.array([1, 0.5, 0, -1, 0, 2, -0.3])  # f(x) = 1 + 0.5*ln(x) - 1/x + 2x^2 - 0.3x^3
    xl2, xr2 = 0.5, 3
    b_test2 = compute_b_coefficients(a_test2, xl2, xr2)
    print("Test 2 coefficients:", b_test2)
    plot_approximation(a_test2, b_test2, xl2, xr2)
    
    # Test case 3: Different interval
    a_test3 = np.array([0, 1, -1, 0, 0.5, 0, 0])  # f(x) = ln(x) - 1/x^2 + 0.5x
    xl3, xr3 = 1, 4
    b_test3 = compute_b_coefficients(a_test3, xl3, xr3)
    print("Test 3 coefficients:", b_test3)
    plot_approximation(a_test3, b_test3, xl3, xr3)