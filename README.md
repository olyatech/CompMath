# Implementation of Computational mathematics algorithms for MIPT CompMath course

## Strict methods for systems solving
Implementation of Gaussian elimination with partial pivoting for systems like $Ax = b$. 
Source code with usage example and pytest tests can be found in `linear_systems` folder.

Realization follows the next steps:
1. LU Decomposition with Partial Pivoting: decompose the matrix $A$ into $PA=LU$, where $P$ is the permutation matrix, $L$ is the lower triangular matrix, and $U$ is the upper triangular matrix.
2. Forward and Back Substitution: solve the system using forward substitution for $Ly=Pb$ and back substitution for $Ux=y$.

