# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:38:30 2020

@author: Enrico Regolin

The function impute EM has been derived from Junkyu Park's implementation in:
https://joon3216.github.io/research_materials/2019/em_imputation_python.html
"""

# EM imputation
import numpy as np
from functools import reduce


#%%
#
## function for EM imputing of mising data
#original eps = 1e-08

def impute_em(X, max_iter = 200, eps = 0.1):
    '''(np.array, int, number) -> {str: np.array or int}
    
    Precondition: max_iter >= 1 and eps > 0
    
    Return the dictionary with five keys where:
    - Key 'mu' stores the mean estimate of the imputed data.
    - Key 'Sigma' stores the variance estimate of the imputed data.
    - Key 'X_imputed' stores the imputed data that is mutated from X using 
      the EM algorithm.
    - Key 'C' stores the np.array that specifies the original missing entries
      of X.
    - Key 'iteration' stores the number of iteration used to compute
      'X_imputed' based on max_iter and eps specified.
    '''
    
    nr, nc = X.shape
    C = np.isnan(X) == False
    
    # Collect M_i and O_i's
    one_to_nc = np.arange(1, nc + 1, step = 1)
    M = one_to_nc * (C == False) - 1
    O = one_to_nc * C - 1
    
    # Generate Mu_0 and Sigma_0
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows, ].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))
    
    # Start updating
    Mu_tilde, S_tilde = {}, {}
    X_tilde = X.copy()
    no_conv = True
    iteration = 0
    while no_conv and iteration < max_iter:
        for i in range(nr):
            S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
            if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
                M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                try:
                    Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.inv(S_OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                    X_tilde[i, M_i] = Mu_tilde[i]
                    S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                except:
                    print('singular matrix detected')
                    Mu_tilde[i] = Mu[np.ix_(M_i)] + S_MO @ np.linalg.pinv(S_OO) @ (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                    X_tilde[i, M_i] = Mu_tilde[i]
                    S_MM_O = S_MM - S_MO @ np.linalg.pinv(S_OO) @ S_OM
                S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
        Mu_new = np.mean(X_tilde, axis = 0)
        S_new = np.cov(X_tilde.T, bias = 1) + reduce(np.add, S_tilde.values()) / nr
        no_conv = np.linalg.norm(Mu - Mu_new) >= eps or np.linalg.norm(S - S_new, ord = 2) >= eps
        
        print(np.linalg.norm(Mu - Mu_new))
        print(np.linalg.norm(Mu - Mu_new)/np.linalg.norm(Mu))
   #     print(np.linalg.norm(S - S_new, ord = 2))

        Mu = Mu_new
        S = S_new
        
        iteration += 1
        print(round(iteration/max_iter*100,2))

    result = {
        'mu': Mu,
        'Sigma': S,
        'X_imputed': X_tilde,
        'C': C,
        'iteration': iteration
    }
    
    return result

#%%
#
#

