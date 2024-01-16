# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
import random
import math
import time
import scipy
import warnings
warnings.filterwarnings("ignore")

from quantum_approach_main import *

# %%
# Test for Procedure 1
random.seed(20240113)
random_seeds = [random.randint(1, 10000) for _ in range(20)]
initial_norm_dfs = []
initial_cond_dfs = []

for i in range(10)[2:]:
    two_hc = Two_State_Hypercube({'Lambda':20*i, 'Mu': 8})
    two_hc.Update_Parameters(N = i, K = 10*i)

    norm_before_list = [] 
    norm_after_list = [] 
    cond_before_list = [] 
    cond_after_list = [] 

    for j in range(len(random_seeds)):
        two_hc.Random_Pref()
        two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
        two_hc.Random_Fraction(seed = random_seeds[j])
        two_hc.Myopic_Policy(source = 't_mat')
        A = two_hc.Cal_A()
        M0 = two_hc.priori_M()

        I = np.eye(2**i)
        B1 = A - np.identity(2**i)
        A2 = np.dot(M0, A)
        B2 = A2 - np.identity(2**i)
        norm_before_list.append(np.linalg.norm(B1))
        norm_after_list.append(np.linalg.norm(B2))
        
        A[-1] = 0
        A[-1, -1] = 1
        A2[-1] = 0
        A2[-1, -1] = 1
        cond_before_list.append(np.linalg.cond(A))
        cond_after_list.append(np.linalg.cond(A2))

    initial_norm_dfs.append(pd.DataFrame({'before': norm_before_list, 'after':norm_after_list}))
    initial_cond_dfs.append(pd.DataFrame({'before': cond_before_list, 'after':cond_after_list}))

# %%
fig, ax = plt.subplots()
ax.tick_params(axis='both', labelsize=23)
x = np.arange(2, 10)

before_mean = []
before_ci = []
after_mean = []
after_ci = []

for df in initial_norm_dfs:
    before_mean.append(df['before'].mean())
    before_ci.append(1.96 * df['before'].std() / np.sqrt(len(df['before'])))
    after_mean.append(df['after'].mean())
    after_ci.append(1.96 * df['after'].std() / np.sqrt(len(df['after'])))

ax.errorbar(x, before_mean, yerr=before_ci, label=r'$M_0=I$', marker = '.', 
            color = '#ADC8FF', capsize=3, markersize=10)
ax.errorbar(x, after_mean, yerr=after_ci, label=r'$M_0=M^*_0$', marker = '.', 
            color = '#254EDB', capsize=3, markersize=10)

ax.legend(fontsize=23, loc='upper left')

ax.set_xlabel(r'$n$', fontsize=23)
ax.set_ylabel(r'$\|M_0A - I\|_2$', fontsize=23)

fig.set_size_inches(10,7)
plt.savefig('initial_norm.png', dpi=600)
plt.show()

# %%
fig, ax = plt.subplots()
x = np.arange(2, 10)
ax.tick_params(axis='both', labelsize=23)

before_mean = []
before_ci = []
after_mean = []
after_ci = []

for df in initial_cond_dfs:
    before_mean.append(df['before'].mean())
    before_ci.append(1.96 * df['before'].std() / np.sqrt(len(df['before'])))
    after_mean.append(df['after'].mean())
    after_ci.append(1.96 * df['after'].std() / np.sqrt(len(df['after'])))

ax.errorbar(x, before_mean, yerr=before_ci, label=r'$M_0=I$', marker = '.', color = '#ADC8FF', capsize=3)
ax.errorbar(x, after_mean, yerr=after_ci, label=r'$M_0=M^*_0$', marker = '.', color = '#254EDB', capsize=3)

ax.legend(fontsize=23, loc='upper left')

ax.set_xlabel(r'$n$', fontsize=23)
ax.set_ylabel('cond' + r'$(M_0A)$', fontsize=23)
fig.set_size_inches(10,7)
plt.savefig('initial_cond.png', dpi=600)
plt.show()

# %%
# Test for Procedure 2
random.seed(20240113)
random_seeds = [random.randint(1, 10000) for _ in range(20)]
spai_sp_dfs = []
spai_cond_dfs = []

for i in range(8)[2:]:
    two_hc = Two_State_Hypercube({'Lambda':5*(i-1)*4, 'Mu': 8})
    two_hc.Update_Parameters(N = 4, K = 10)

    sp_before_list = [] 
    sp_after_list = [] 
    cond_before_list = [] 
    cond_after_list = [] 

    for j in range(len(random_seeds)):
        two_hc.Random_Pref()
        two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
        two_hc.Random_Fraction(seed = random_seeds[j])
        two_hc.Myopic_Policy(source = 't_mat')
        A = two_hc.Cal_A()
        M0 = two_hc.priori_M()

        A2 = local_spai(A, M0)
        # A[-1] = 0
        # A[-1, -1] = 1
        cond_before_list.append(np.linalg.cond(A))
        cond_after_list.append(np.linalg.cond(A2))

        a_sparse = csr_matrix(A)
        sp_before_list.append(1 - a_sparse.nnz / (a_sparse.shape[0] * a_sparse.shape[1]))
        a2_sparse = csr_matrix(A2)
        sp_after_list.append(1 - a2_sparse.nnz / (a2_sparse.shape[0] * a2_sparse.shape[1]))

    spai_sp_dfs.append(pd.DataFrame({'before': sp_before_list, 'after':sp_after_list}))
    spai_cond_dfs.append(pd.DataFrame({'before': cond_before_list, 'after':cond_after_list}))

# %%
fig, ax = plt.subplots()
x = np.arange(20, 140, 20)
ax.tick_params(axis='both', labelsize=23)

before_mean = []
before_ci = []
after_mean = []
after_ci = []

for df in spai_sp_dfs:
    before_mean.append(df['before'].mean())
    before_ci.append(1.96 * df['before'].std() / np.sqrt(len(df['before'])))
    after_mean.append(df['after'].mean())
    after_ci.append(1.96 * df['after'].std() / np.sqrt(len(df['after'])))

ax.errorbar(x, before_mean, yerr=before_ci, label=r'$A$', marker = '.', color = '#ADC8FF', capsize=3)
ax.errorbar(x, after_mean, yerr=after_ci, label=r'$A^*$', marker = '.', color = '#254EDB', capsize=3)

ax.legend(fontsize=23, loc='upper left')

ax.set_xlabel(r'$\Lambda$', fontsize=23)
ax.set_ylabel('Sparsity', fontsize=23)
fig.set_size_inches(10,7)

plt.savefig('spai_sp.png', dpi=600)
plt.show()

# %%
fig, ax = plt.subplots()
x = np.arange(20, 140, 20)
ax.tick_params(axis='both', labelsize=23)

before_mean = []
before_ci = []
after_mean = []
after_ci = []

for df in spai_cond_dfs:
    before_mean.append(df['before'].mean())
    before_ci.append(1.96 * df['before'].std() / np.sqrt(len(df['before'])))
    after_mean.append(df['after'].mean())
    after_ci.append(1.96 * df['after'].std() / np.sqrt(len(df['after'])))

ax.errorbar(x, before_mean, yerr=before_ci, label=r'$A$', marker = '.', color = '#ADC8FF', capsize=3)
ax.errorbar(x, after_mean, yerr=after_ci, label=r'$A^*$', marker = '.', color = '#254EDB', capsize=3)

ax.legend(fontsize=23, loc='upper left')

ax.set_xlabel(r'$\Lambda$', fontsize=23)
ax.set_ylabel('Condition Number', fontsize=23)
fig.set_size_inches(10,7)

plt.savefig('spai_cond.png', dpi=600)
plt.show()


