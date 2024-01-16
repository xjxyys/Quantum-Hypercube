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
from linear_solvers.hhl import HHL
warnings.filterwarnings("ignore")

from quantum_approach_main import *

# %%
from src_quantum import get_full_vector
random.seed(20240113)
random_seeds = [random.randint(1, 10000) for _ in range(20)] # 20 sets of random numbers
hhl_runtime_dfs = [] # Record 20 sets of running times with different N

for i in range(2, 6): # N = 2, 3, 4, 5
    two_hc = Two_State_Hypercube({'Lambda':20*i, 'Mu': 8})
    two_hc.Update_Parameters(N = i, K = 10*i)

    error_before_list = [] 
    error_after_list = [] 

    for j in range(len(random_seeds)):
        two_hc.Random_Pref()
        two_hc.Random_Time_Mat(t_min = 1, t_max = 10, seed = 1)
        two_hc.Random_Fraction(seed = random_seeds[j]) # Randomly generate 20 matrices
        two_hc.Myopic_Policy(source = 't_mat')
        A = two_hc.Cal_A()
        M0 = two_hc.priori_M()
        A_star = local_spai(A, M0)
        A[-1] = 0
        A[-1, -1] = 1
        A_star_expand = expand_matrix(A_star)
        A_expand = expand_matrix(A)
        b = expand_b(two_hc.Cal_b())
        # before_time records the time to solve A_expand,b
        time_1 = time.perf_counter()
        hhl_solution = HHL().solve(A_expand, b)
        time_2 = time.perf_counter()
        hhl_solution_star = HHL().solve(A_star_expand, b)
        time_3 = time.perf_counter()
        before_time = time_2 - time_1
        after_time = time_3 - time_2
        # after_time records the time to solve A_star_expand,b
        error_before_list.append(before_time)
        error_after_list.append(after_time)

    hhl_runtime_dfs.append(pd.DataFrame({'before': error_before_list, 'after':error_after_list}))

# %%
# Visualization
fig, ax = plt.subplots()
x = np.arange(2, 6)

before_mean = []
before_ci = []
after_mean = []
after_ci = []

for df in hhl_runtime_dfs:
    before_mean.append(df['before'].mean())
    before_ci.append(1.96 * df['before'].std() / np.sqrt(len(df['before'])))
    after_mean.append(df['after'].mean())
    after_ci.append(1.96 * df['after'].std() / np.sqrt(len(df['after'])))

ax.errorbar(x, before_mean, yerr=before_ci, label=r'$Ax=b$', marker = '.', color = '#ADC8FF', capsize=3)
ax.errorbar(x, after_mean, yerr=after_ci, label=r'$A^*x=b$', marker = '.', color = '#254EDB', capsize=3)

ax.legend(fontsize=20, loc='upper left')
ax.set_xlabel(r'$n$',fontsize=20)
ax.set_ylabel(r'Run Time of HHL Algorithm',fontsize=18)
ax.tick_params(axis='both', labelsize=20)

plt.xticks(np.arange(2, 6, 1))
fig.set_size_inches(10,7)
plt.savefig('hhl_time.png',dpi=600,bbox_inches='tight')
plt.show()