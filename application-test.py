# %%
import numpy as np
from linear_solvers.hhl import HHL
from src_quantum import *
from quantum_approach_main import *
def priori_M(A):
    # Calculate the priori initial state of M, which is denoted as M^*_0
    diagonal_elements = np.diag(A)
    diagonal_elements_reciprocal = np.reciprocal(diagonal_elements)
    M0 = np.diag(diagonal_elements_reciprocal)
    M0[-1, :] = -np.diag(M0)  
    M0[-1, -1] = 1
    return M0

if __name__ == '__main__':
    transation_matrix = np.array([[-16, 10, 10, 0, 10, 0, 0, 0],
                       [4, -26, 0,  10,  0, 10,  0,  0],
                       [2,  0, -26, 10,  0,  0, 10,  0],
                       [0,  6,  6, -36,  0,  0,  0,  10],
                       [10, 0,  0,   0,-26, 10, 10,  0],
                       [0, 10,  0,   0,  9,-36,  0,  10],
                       [0,  0, 10,   0,  7,  0, -36, 10],
                       [0,  0,  0,  16,  0, 16,  16, -30]])
    transation_matrix[-1] = np.ones(8)
    b = np.zeros(8)
    b[-1] = 1
    expanded_A, expanded_b = Expand_A_and_b(transation_matrix, b)
    hhl_solution = HHL().solve(expanded_A, expanded_b)
    full_vector = get_full_vector(hhl_solution, 3,)
    print(full_vector)
    print('----------------------------------finish')

# %%
# M0 = priori_M(transation_matrix)
A_star = local_spai(transation_matrix, np.eye(8))
A_star[-1] = 1
expanded_A_new, expanded_b_new = Expand_A_and_b(A_star, b)
hhl_solution_new = HHL().solve(expanded_A_new, expanded_b_new)
full_vector_new = get_full_vector(hhl_solution_new, 3,)
print('directly solve the primal matrix',full_vector)
print('solve the preconditioned matrix',full_vector_new)
exact_solution = np.linalg.solve(expanded_A, expanded_b)
print(f'exact solution: {exact_solution}')


