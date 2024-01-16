from dispatch_main import *
from quantum_linear_solvers.linear_solvers import NumPyLinearSolver, HHL
from qiskit.quantum_info import Statevector
# 制备矩阵A和b
def Get_matrix_and_b(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 9001)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b(N=N, i=i)
    return transition, b

def Get_matrix_and_b_1120(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 9001)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_1120(N=N, i=i)
    return transition, b


def Get_matrix_and_b_1112(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 9001)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_1112(N=N, i=i)
    return transition, b

def Get_matrix_and_b_1119(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 9001)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_1119(N=N, i=i)
    return transition, b

def Get_matrix_and_b_1224(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 9001)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_1224(N=N, i=i)
    return transition, b


def Get_matrix_and_b_inverse(N, K, Lambda, Mu, t_mat, f_mat):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 3)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_inverse(N=N)
    return transition, b

def Get_matrix_and_b_0112(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 9001)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_0112(N=N)
    return transition, b

# 把A变成酉矩阵，b的长度也相应扩充
def Expand_A_and_b(A, b):
    A_dagger = np.conjugate(A.T)
    zero_matrix = np.zeros(A.shape)
    new_A = np.block([[zero_matrix, A], [A_dagger, zero_matrix]])
    new_b = np.block([b, np.zeros(b.shape)])
    return new_A, new_b

def row_scaling(matrix):
    # 计算原始矩阵的条件数
    original_cond = np.linalg.cond(matrix)
    # 计算每行的范数
    row_norms = np.linalg.norm(matrix, axis=1)

    # 避免除以零，将零范数替换为1（或者一个小的正数）
    row_norms[row_norms == 0] = 1

    # 初始化缩放后的矩阵
    scaled_matrix = np.copy(matrix)

    # 对每一行进行缩放，使其范数为平均范数
    mean_norm = np.mean(row_norms)

    for i in range(matrix.shape[0]):
        scaled_matrix[i, :] *= mean_norm / row_norms[i]
    
    # 得到最后一行乘的系数
    last_row_multiplier = mean_norm / row_norms[-1]

    # 计算缩放后的矩阵的条件数
    scaled_cond = np.linalg.cond(scaled_matrix)
    print('-------------------original_cond: ', original_cond)
    print('-------------------scaled_cond: ', scaled_cond)
    return scaled_matrix, last_row_multiplier


def optimize_last_row(matrix):
    """
    This function takes a matrix as input and returns a new matrix where the last row has been modified to increase sparsity.
    The function also returns a list of operations that were performed.
    
    Parameters:
    matrix (numpy array): Input matrix

    Returns:
    new_matrix (numpy array): Matrix with optimized last row
    operations (list): List of operations performed
    """
    # Get the number of rows
    num_rows = matrix.shape[0]
    
    # Copy the matrix to avoid changing the original matrix
    new_matrix = matrix.copy()
    
    # Initialize a list to store the operations
    operations = []
    
    # Loop through each row (except the last one)
    for i in range(num_rows-1):
        # Find the multiple of the ith row that when added to the last row will make the ith column in the last row zero
        if new_matrix[i, i] != 0:  # Avoid division by zero
            multiplier = -new_matrix[-1, i] / new_matrix[i, i]
            # Add this multiple of the ith row to the last row
            new_matrix[-1] += multiplier * new_matrix[i]
            # Store the operation
            operations.append((i, multiplier))
    
    return new_matrix

def get_full_vector(solution, N, inverse=False):
    if not inverse:
        start_loc = int("1"+"0"*(N+7), 2)
    else:
        start_loc = int("1"+"0"*(N+8), 2)
    # start_loc = int("1"+"0"*(N+6), 2)
    # start_loc = int("100000000", 2)
    solution_vector = Statevector(solution.state).data[start_loc:start_loc+2**(N+1)].real
    # 按比例缩放，最后加和为1
    solution_vector = solution_vector / np.sum(solution_vector)
    return np.array(solution_vector)
    # Extract vector components
def add_last_row_to_above_all(matrix, b):
    """
    This function takes a matrix and a vector as input and returns a new matrix and vector where the last row has been added to all the rows above it.
    The function also returns a list of operations that were performed.
    
    Parameters:
    matrix (numpy array): Input matrix
    b (numpy array): Input vector

    Returns:
    new_matrix (numpy array): Matrix with optimized last row
    new_b (numpy array): Vector with optimized last row
    operations (list): List of operations performed
    """
    # Get the number of rows
    num_rows = matrix.shape[0]
    
    # Copy the matrix to avoid changing the original matrix
    new_matrix = matrix.copy()
    new_b = b.copy()
    
    # Initialize a list to store the operations
    operations = []
    
    # Loop through each row (except the last one)
    for i in range(num_rows-1):
        # Add the last row to the ith row
        new_matrix[i] += new_matrix[-1]
        # Add the last element of b to the ith element of b
        new_b[i] += new_b[-1]
        # Store the operation
        operations.append(i)
    
    return new_matrix, new_b, operations


# Algorithm 1: Diagonal Scaling of A
def diagonal_scaling_1(A, epsilon=1e-10):
    """
    Apply diagonal scaling to matrix A. Adds a small positive value to diagonal elements 
    if they are non-positive to ensure they are positive before scaling.

    Parameters:
    A (numpy.ndarray): The input matrix.
    epsilon (float): A small positive value to add to non-positive diagonal elements.

    Returns:
    numpy.ndarray: The diagonally scaled matrix.
    """
    diag_A = np.diag(A)
    
    # Ensuring all diagonal elements are positive
    adjusted_diag_A = np.where(diag_A <= 0, diag_A + epsilon, diag_A)
    
    D = np.diag(1.0 / np.sqrt(adjusted_diag_A))
    A_scaled = np.dot(D, np.dot(np.tril(A), D))
    A_scaled_H = A_scaled + A_scaled.conj().T + np.eye(A.shape[0])
    return A_scaled_H, D
 
def diagonal_scaling(A):
    diag_A = np.diag(A)
    D = np.diag(1.0 / np.sqrt(diag_A))
    A_scaled = np.dot(D, np.dot(np.tril(A), D))
    A_scaled = A_scaled + A_scaled.conj().T + np.eye(A.shape[0])
    return A_scaled, D


def regularize_matrix(A, epsilon=1e-5):
    """
    Regularize a matrix by adding a small value to its diagonal.

    Parameters:
    A (numpy.ndarray): The input matrix.
    epsilon (float): A small value to add to the diagonal elements.

    Returns:
    numpy.ndarray: The regularized matrix.
    """
    return A + epsilon * np.eye(A.shape[0])


# Algorithm 2: SSAI Preconditioner
def ssai_preconditioner(A, lfil):
    n = A.shape[0]
    M = np.zeros((n, n))

    for j in range(n):
        m = np.zeros(n)
        r = np.copy(A[:, j])

        for _ in range(lfil):
        # for _ in range(2*lfil):
            i = np.argmax(np.abs(r))
            if A[i, i] == 0:
                continue  # Avoid division by zero
            delta = r[i] / A[i, i]
            m[i] += delta
            r -= delta * A[:, i]

        M[:, j] = m

    M = (M + M.T) / 2  # Symmetrizing M
    return M

def ssai_preconditioner_1(A, lfil=None):
    n = A.shape[0]  # Number of rows/columns in A
    if lfil is None:
        nnz_A = np.count_nonzero(A)  # Count of non-zero elements in A
        lfil = np.ceil(nnz_A / n).astype(int)  # Average non-zeros per row, rounded up
    # itmax = 2 * lfil  # Maximum iterations
    itmax = lfil  # Maximum iterations
    # print('-------------------lfil: ', lfil)
    M = np.zeros((n, n))  # Initializing M

    for j in range(n):
        m = np.zeros(n)  # Initialize m
        r = np.eye(n)[j]  # Unit vector e_j

        for k in range(itmax):
            i = np.argmax(np.abs(r))  # Index of maximum absolute value in r
            delta = r[i]
            m[i] += delta

            if np.count_nonzero(m) >= lfil:
                break

            r -= delta * A[:, i]  # Update the residual

        M[:, j] = m  # Update column j of M

    M = (M + M.T) / 2  # Symmetrize M

    return M

def calculate_lfil(A):
    """
    Calculate the lfil parameter for the SSAI preconditioner algorithm.

    Parameters:
    A (numpy.ndarray): The input matrix.

    Returns:
    int: The calculated lfil value.
    """
    nnz_A = np.count_nonzero(A)  # Count of non-zero elements in A
    n = A.shape[0]  # Number of rows in A
    lfil = np.ceil(nnz_A / n).astype(int)  # Average non-zeros per row, rounded up
    return lfil

def is_hermitian(matrix):
    """
    Check if a matrix is Hermitian.

    Parameters:
    matrix (numpy.ndarray): The matrix to check.

    Returns:
    bool: True if the matrix is Hermitian, False otherwise.
    """
    return np.allclose(matrix, matrix.conj().T)

def gen_matrix_and_save(N, K, Lambda, Mu, t_mat, f_mat, i=-1):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 3)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b(N=N)
    print('-------------------saving matrix: ', N)
    np.save(f'./data/TransitionMatrix_{N}.npy', transition)
    return transition, b

def gen_matrix_and_save_update(N, K, Lambda, Mu, t_mat, f_mat, i=-1, flag=False):
    two_hc = Two_State_Hypercube({'Lambda':Lambda, 'Mu': Mu})
    two_hc.Update_Parameters(N = N, K = K)
    two_hc.Random_Pref(seed = 1)
    two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
    two_hc.Random_Fraction(seed = 3)
    two_hc.Myopic_Policy(source = 't_mat')
    transition, b = two_hc.Get_matrix_and_b_0112(N=N)
    print('-------------------saving matrix: ', N)
    np.save(f'./data/TransitionMatrix_{N}_update.npy', transition)
    return transition, b

def get_full_vector(solution, N):
    """
    solution: solution of HHL
    N: number of server
    num_qubits: number of qubits
    """
    num_qubits = len(solution.state.qubits)
    start_loc = int('1' + '0' * (num_qubits-1), 2)
    # start_loc = int("1"+"0"*(N+6), 2)
    # start_loc = int("100000000", 2)
    solution_vector = Statevector(solution.state).data[start_loc:start_loc+2**(N+1)].real
    # 按比例缩放，最后加和为1
    solution_vector = solution_vector / np.sum(solution_vector)
    return np.array(solution_vector)
    # Extract vector components