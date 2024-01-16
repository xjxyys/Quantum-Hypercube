import fast_matrix_market
import os
import numpy as np
import scipy
from numpy.linalg import inv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import comb
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import random
import heapq
import functools
import math
from joblib import Parallel, delayed
import heapq

#################     Hypercube Class      #################

class Two_State_Hypercube():
    def __init__(self, data_dict = None):
        # initilize data stored in self.data_dict
        self.keys = ['N', 'K', 'Lambda', 'Mu', 'frac_j', 't_mat', 'pre_list', 'pol']
        self.rho_hyper, self.rho_approx, self.rho_simu = None, None, None # initialize the utilizations to be None 
        self.prob_dist = None
        self.data_dict = dict.fromkeys(self.keys, None) 
        if data_dict is not None:
            for k, v in data_dict.items():
                if k in self.keys:
                    self.data_dict[k] = v
        self.G = None # G for approximation
        self.Q = None # Q for approximation
        self.r = None # r for approximation
        self.P_b = None # This is to differentiate from 3-state case
        self.q_nj = None # This is the dispatch probability using Linear-alph algorithm. This is shared by both 2-state and 3-state algorithms. 

    def Update_Parameters(self, **kwargs): 
        # update any parameters passed through kwargs
        for k, v in kwargs.items(): 
            if k in self.keys:
                self.data_dict[k] = v
                if k == 'pre_list': # reset G if pre_list changes
                    self.G = None
    
    def Random_Pref(self, seed=9001):
        # random preference list
        random.seed(seed) # Set Random Seed
        N, K = self.data_dict['N'], self.data_dict['K']
        # Shuffle the IDs as each preference list
        pre_list = np.array([random.sample(list(range(N)),N) for _ in range(K)])
        self.Update_Parameters(pre_list = pre_list)
    
    def Random_Fraction(self, seed=9001):
        '''
            Obtain random frac. 
        '''
        np.random.seed(seed)
        K = self.data_dict['K']
        frac_j = np.random.random(size=K)
        frac_j /= sum(frac_j)
        self.data_dict['frac_j'] = frac_j

    def Random_Time_Mat(self, t_min = 1, t_max = 10, seed = 9001):
        np.random.seed(seed)
        N, K = self.data_dict['N'], self.data_dict['K']
        t_mat = np.random.uniform(low=t_min, high=t_max, size=(K,N))
        self.Update_Parameters(t_mat = t_mat)

    def Myopic_Policy(self, source='t_mat'):
        # Obtain the policy to dispatch the cloest available unit given the time matrix
        N, K = self.data_dict['N'], self.data_dict['K']
        if source == 't_mat':
            t_mat = self.data_dict['t_mat']
            pre_list = t_mat.argsort(axis=1)
            self.Update_Parameters(pre_list = pre_list)
        elif source == 'pre':
            pre_list = self.data_dict['pre_list']
        else:
            print('Wrong source!')
        policy = np.zeros([2**N, K],dtype=np.int64)
        for s in range(2**N):
            for j in range(K):
                pre = pre_list[j]
                for n in range(N):
                    if not s >> pre[n] & 1: # n th choice is free
                        policy[s, j] = pre[n]
                        break
        self.data_dict['pol'] = policy

    def Cal_Trans(self):
        keys = ['N', 'K', 'Lambda', 'Mu', 'pol', 'frac_j']
        N, K, Lambda, Mu, pol, frac_j = [self.data_dict.get(key) for key in keys]
        # Calculate the 
        N_state = 2**N
        A = np.zeros([N_state, N_state])
        # Calculate upward transtition
        for s in range(N_state-1): # The last state will not transition to other states by a arrival
            pol_s = pol[s]
            for j in range(K):
                dis = pol_s[j]
                A[s, s+2**dis] += Lambda * frac_j[j]
        # Calculate downward transtition
        for s in range(1,N_state): # The first state will not transition
            bin_s = bin(s)
            len_bin = len(bin_s)
            i = 0
            while bin_s[len_bin-1-i] != 'b':
                if bin_s[len_bin-1-i] == '1':
                    A[s,s-2**i] = Mu
                i += 1
        return A

    def Cal_A(self):
        # Calculate the matrix A
        matrix = self.Cal_Trans()
        matrix = matrix.T - np.diag(matrix.sum(axis=1))
        matrix[-1] = 1

        return matrix

    def Cal_b(self):
        # Calculate the b
        b = np.zeros(2**self.data_dict.get('N'))
        b[-1] = 1
        return b

    # Procedure1: Set The Initial Preconditioner
    def priori_M(self):
        # Calculate the priori initial state of M, which is denoted as M^*_0
        A =self.Cal_A()
        diagonal_elements = np.diag(A)
        diagonal_elements_reciprocal = np.reciprocal(diagonal_elements)
        M0 = np.diag(diagonal_elements_reciprocal)
        M0[-1, :] = -np.diag(M0)  
        M0[-1, -1] = 1
        return M0

#################     Functions for SPAI      #################

def cal_M(matrix, M0):
    n_most_profitable_indices = 5 # bounded from 3 to 8
    epsilon = 0.001 # 0.1 to 0.5 according to https://mediatum.ub.tum.de/doc/1107998/426923.pdf#page=64&zoom=100,117,534
    maxiter = 5 # 1 to 5

    A = scipy.sparse.csc_matrix(matrix)

    N = A.shape[0]
    M = scipy.sparse.csc_matrix(M0)
    # M = scipy.sparse.identity(N, format='csc')

    B = M * A - np.identity(M.shape[1])

    Bnormbefore = np.linalg.norm(B)
    Mnz = scipy.sparse.csr_matrix.count_nonzero(M)
    Anz = scipy.sparse.csr_matrix.count_nonzero(A)
    ratiobefore = Mnz / Anz

    #AInv = scipy.sparse.linalg.inv(A)
    np.set_printoptions(precision=10, linewidth=800)
    maxn1 = 0
    maxn2 = 0
    minn1 = maxiter*[M.shape[1]]
    minn2 = maxiter*[M.shape[1]]

    iterations = M.shape[1]*[0]
    def process(k):
        # print(k)
        iter = 0
        # For each column
        m_k = M[:,k]

        # Create e_k column
        e_k = np.matrix([0]*N).T
        e_k[k] = 1

        # Calculate J
        J = m_k.nonzero()[0] # gets row inds of nonzero

        n2 = J.size

        # Calculate A(.,J)
        A_J = A[:,J]

        # Calculate I from A(.,J)
        I = np.unique(A_J.nonzero()[0])
        n1 = I.size

        # Reduced matrix A_IJ (A hat) an n1 x n2 matrix
        A_IJ = A[np.ix_(I, J)].todense()
        
        Q, R = np.linalg.qr(A_IJ, mode="complete")
        R = R[:n2,:n2]

        # Compute mhat_k
        try:
            chat_k = np.matrix(Q.T[:,list(I).index(k)])
        except:
            chat_k = np.zeros(n2)
        mhat_k = scipy.linalg.solve_triangular(R, chat_k[0:n2])

        # Compute residual r
        rI = A_IJ * mhat_k - e_k[I]
        r = np.zeros((M.shape[0],1))
        r[I] = rI
        try:
            list(I).index(k)
        except:
            r[k] = -1
        r_norm = np.linalg.norm(r)
        
        while iter < maxiter and len(J) < np.log2(A.shape[0]):
            iter += 1
            # print("iter: ",iter)
            # Calculate L
            L = np.union1d(I,k) #np.nonzero(r)[0] # Lk ← Ik ∪ {k} ? from paper https://mediatum.ub.tum.de/doc/1107998/426923.pdf#page=64&zoom=100,117,534

            # Calculate Jtilde: All of the the new column indices of A that appear in all
            # L rows but not in J
            Jtilde = np.array([],dtype=int)
            for l in L:
                A_l = A[l,:]
                NZofA_l = np.unique(A_l.nonzero()[1])
                N_l = np.setdiff1d(NZofA_l, J)
                Jtilde = np.union1d(Jtilde,N_l)

            # Calculate the new norm of the modified residual and record the indices j
            avg_rho = 0
            j_rho_pairs = []
            for j in Jtilde:
                Ae_j = A[:,j].todense()
                Ae_jnorm = np.linalg.norm(Ae_j)
                rTAe_j = r.T * Ae_j ## if A[:,j] (row_inds[col_ptr[j]]) overlap with nonzeros of r, then mult, otherwise don't
                rho_jsquared = r_norm*r_norm - (rTAe_j * rTAe_j) / (Ae_jnorm * Ae_jnorm)
                avg_rho += rho_jsquared
                j_rho_pairs.append((rho_jsquared[0,0],j))

            avg_rho = avg_rho / len(j_rho_pairs)

            # Creates min heap to quickly find indices with lowest error.
            heap = []
            for pair in j_rho_pairs:
                    heapq.heappush(heap, (pair[0], pair[1]))

            # Select the remaining 5 indices that create the lowest residuals
            pops = 0
            Jtilde = []
            while len(heap) > 0 and pops < n_most_profitable_indices:
                pair = heapq.heappop(heap)
                if (pair[0] < avg_rho):
                    Jtilde.append(pair[1])
                    pops += 1

            # Update Q, R
            n2tilde = len(Jtilde)
            Jtilde = np.sort(Jtilde) # Needed for calculation of permutation matrices

            Itilde = np.setdiff1d(np.unique(A[:,Jtilde].nonzero()[0]), I)
            n1tilde = len(Itilde)

            AIJtilde = A[np.ix_(I, Jtilde)]
            AItildeJtilde = A[np.ix_(Itilde,Jtilde)]

            Au = Q.T * AIJtilde
            B_1 = Au[:n2,:]
            B_2 = np.vstack((Au[n2:n1,:], AItildeJtilde.todense()))

            QB, RB = np.linalg.qr(B_2, mode="complete")
            RB = RB[:n2tilde,:n2tilde]

            # Construct R and Q by stacking and matrix products
            R = np.hstack((np.vstack((R, np.zeros((n2tilde, n2)))), np.vstack((B_1, RB)))) # don't need entire R
            q = np.hstack((np.vstack((Q[:,n2:], np.zeros((n1tilde,n1-n2)))), np.vstack((np.zeros((n1,n1tilde)), np.identity(n1tilde)))))
            Q = np.hstack((np.vstack((Q[:,:n2], np.zeros((n1tilde,n2)))), q * QB))

            # New J and I
            J = np.append(J,Jtilde) # J U Jtilde
            n2 = J.size
            I = np.append(I,Itilde) # Itilde
            n1 = I.size

            try:
                chat_k = np.matrix(Q.T[:,list(I).index(k)])
            except:
                chat_k = np.zeros(n2)
            mhat_k = scipy.linalg.solve_triangular(R, chat_k[0:n2])

            rI = A[np.ix_(I,J)] * mhat_k - e_k[I]
            r[I] = rI

            try:
                list(I).index(k)
            except:
                r[k] = -1

            r_norm = np.linalg.norm(rI)

            # print("norm:", r_norm)

        if iter <= maxiter:
            iter = iter
        else:
            iter = maxiter+1

        m_k[J] = mhat_k
        # Place result column in matrix
        return m_k, iter
        
    results = Parallel(n_jobs=16)(delayed(process)(i) for i in range(0,M.shape[0]))

    for k in range(0,M.shape[0]):
        M[:,k] = results[k][0]
        iterations[k] = results[k][1]

    return M, iterations

def local_spai(matrix, M0):
    # Locally do spai   
    A_star = np.dot(cal_M(matrix, M0)[0].toarray()[:,:-1], matrix[:-1])
    A_star[-1] = 0
    A_star[-1, -1] = 1
    return A_star

#################     Functions for HHL     #################

def expand_matrix(mat):
    n = mat.shape[0]
    zeros = np.zeros_like(mat)
    block_upper = np.concatenate((zeros, mat), axis=1)
    block_lower = np.concatenate((mat.T.conj(),zeros.T.conj()), axis=1)
    block_matrix = np.concatenate((block_upper, block_lower), axis=0)
    return block_matrix

def expand_b(b):
    zeros = np.zeros_like(b)
    b = np.hstack((b, zeros))
    return b