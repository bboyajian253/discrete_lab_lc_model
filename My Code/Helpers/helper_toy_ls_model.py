import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
from numba import njit, jit, guvectorize # NUMBA speed up quite a lot, see the functions that have the decorator just above
from scipy.optimize import brentq  # root-finding routine
import sympy as sp
from latex2sympy2 import latex2sympy
import re
min_a = -200 # can think of this as borrowing no more than min_a,000 dollars
max_a = 4000
inc_a = 1

# lambda_t_H deterministic component depends on health and age
# will need to be estimated for each composite health type, likely normalizing lambda_t_GG
# for know lets use captina's calibration she estimates two sets of coeffecients
# one for each education group college/noncollege ill take the avg for now
w_determ_cons = (0.732 + 1.29)/2 # constant
w_age = (0.074 + 0.082)/2 
w_age_2 = (-0.00095 + -0.00112)/2
w_age_3 = (0.0000024 + 0.0000032)/2
w_avg_good_health = (0.232 + 0.17)/2
w_avg_good_health_age = (0.0 + 0.0005)/2
w_good_health = (0.21 + 0.088)/2
w_good_health_age = (0.0 + 0.0015)/2 

            
a_tp1 = 300

a_grid = np.arange(min_a, max_a + inc_a, inc_a)
a_loc = np.where(a_grid == a_tp1)[0][0]


def det_lab_inc( age, health):
    """Returns the deterministic part of the wage"""
    age_comp =  w_determ_cons +  w_age * age +  w_age_2 * age * age +  w_age_3 * age * age * age
    # Ignoring average health for now, will work back in when composite types are implemented.
    health_comp =  w_good_health * health +  w_good_health_age * age * health
    comp = age_comp + health_comp
    return comp  # Returning the summed array

# Example usage
age = np.array([30, 40, 50])  # Example age array
health = np.array([0.8, 0.9, 0.7])  # Example health array

age = 30
health = 0.8



            # nu_t persistent AR(1) shock
rho_nu = 0.9472 # the autocorrelation coefficient for the earnings shock nu
sigma_eps_2 = 0.0198 # variance of innovations
sigma_nu0_2 = 0.093 # variance of initial distribution of the persistent component 

persis_lab_prod__N = 10


def rouwenhorst(N, rho, sigma, mu=0.0):
    """Rouwenhorst method to discretize AR(1) process"""

    q = (rho + 1)/2
    nu = ((N-1.0)/(1.0-rho**2))**(1/2)*sigma
    s = np.linspace(mu/(1.0-rho)-nu, mu/(1.0-rho)+nu, N) # states

    # implement recursively transition matrix
    P = np.array([[q, 1-q], [1-q, q]])

    for i in range(2,N): # add one row/column one by one
        P = (
             q*np.r_[np.c_[P, np.zeros(i)] , [np.zeros(i+1)]] + (1-q)*np.r_[np.c_[np.zeros(i), P] , 
                [np.zeros(i+1)]] + (1-q)*np.r_[[np.zeros(i+1)] ,  np.c_[P, np.zeros(i)]] + q*np.r_[[np.zeros(i+1)] ,  np.c_[np.zeros(i), P]]
                )
        
        P[1:i,:]=P[1:i,:]/2  # adjust the middle rows of the matrix

    return s, P
# states, trans = rouwenhorst(persis_lab_prod__N,rho_nu,sigma_eps_2)
# print('The states are')
# print(states)
# print()
# print('The transition probabilities for each state are ')
# print(trans)



# sp.init_printing()
#y = sp.Symbol('y')
#a function to replace greek letters and other symbols from latex into things python likes
def replace_greeks( myStr) :
    #the list of symbols and their replacements
    #use r"str_to_replace" here so py and re know to treat special characters as literals
    rep = {r"\alpha" : "a", r"\sigma" : "s"}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    newStr = pattern.sub(lambda m: rep[re.escape(m.group(0))], myStr)
    return newStr


for i in range(10,-1,-1) :
    print(i)