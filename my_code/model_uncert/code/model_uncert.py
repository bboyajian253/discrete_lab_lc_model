"""
My Model 2 - basic model no uncertainty

Contains the model equations and derivatives to be used by solver.py and others

Author: Ben Boyajian
Date: 2024-05-29 20:16:01
"""

# Import packages
# General
import time
import numpy as np
import sys
from interpolation import interp
from numba import njit, guvectorize, prange 
# My code
from pars_shocks import Pars, Shocks
import my_toolbox as tb

@njit
def leis_giv_lab(myPars: Pars, labor:float, health:float) -> float:
    """
    converts labor to leisure within period
    encodes the time endowment constraint
    """
    leisure = 1.0 - labor*myPars.phi_n - (1-health)*myPars.phi_H
    leisure = min(myPars.leis_max, leisure)
    return max(myPars.leis_min, leisure)

@njit
def lab_giv_leis(myPars: Pars, leisure:float, health:float) -> float:
    """
    converts leisure to labor within period
    encodes the time endowment constraint
    """
    labor = (1.0 - leisure - (1-health)*myPars.phi_H) / myPars.phi_n #this denom should never be zero, phi_n != 0
    labor = min(myPars.lab_max, labor)
    return max(myPars.lab_min, labor)

@njit
def leis_giv_c(myPars: Pars, c:float, wage:float) -> float:
    """
    convert consumption to leisure within period
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for leisure given current period consumption
    """

    constant = (myPars.phi_n * (1 - myPars.alpha)) / (wage * myPars.alpha) #this denom should !=0, wage is a product of exp != 0  
    leis = constant * c
    #leis = min(myPars.leis_max, leis)
    #return max(myPars.leis_min, leis)
    return leis
   
@njit
def c_giv_leis(myPars: Pars,  leis:float, wage:float) -> float:
    """
    convert leisure to consumption within period
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for consumption given current period leisure
    """
    constant = (wage * myPars.alpha) / (myPars.phi_n * (1 - myPars.alpha)) #this denom should never be zero, alpha != 1 and phi_n != 0
    return constant * leis

@njit
def util_giv_leis(myPars: Pars, c:float, leis:float) -> float:
    """
    utility function given leisure and consumption
    """
    sig = myPars.sigma_util
    alpha = myPars.alpha
    return (1/(1-sig)) * ((c**alpha) * (leis**(1-alpha))) ** (1-sig)

#derivative of utility function with respect to consumption given consumption and leisure
@njit
def util_c_giv_leis(myPars: Pars, c:float, leis:float) -> float:
    """
    derivative of utility function with respect to consumption given consumption and leisure
    """
    sig = myPars.sigma_util
    alpha = myPars.alpha
    return alpha*c**(alpha - 1)*leis**(1 - alpha)/(c**alpha*leis**(1 - alpha))**sig #this denom is 0 if c or leis is 0

@njit
def mb_lab(myPars: Pars, c:float, wage:float, labor:float, health:float) -> float:
    """
    marginal benefit of labor
    """
    leis = leis_giv_lab(myPars, labor, health)
    return wage * util_c_giv_leis(myPars, c, leis)

@njit
def mc_lab(myPars: Pars, c:float, labor:float, health:float) -> float:
    """
    marginal cost of labor
    """
    leis = leis_giv_lab(myPars, labor, health)
    return myPars.phi_n * util_leis_giv_c(myPars, leis, c)

@njit
def util_leis_giv_c(myPars: Pars, leis:float, c:float) -> float:
    """
    derivative of utility function with respect to leisure given consumption and leisure
    """
    return (1-myPars.alpha)*c**(myPars.alpha)*leis**(-myPars.alpha)/(c**myPars.alpha*leis**(1-myPars.alpha))**myPars.sigma_util

@njit
def util_c(myPars: Pars, c:float, wage:float) -> float:
    """
    deriveative of utility function with respect to consumption given consumption and health
    """
    leis = leis_giv_c(myPars, c, wage) #this can also be done explicitly in one function
    return util_c_giv_leis(myPars, c, leis)

@njit
def util_c_inv(myPars: Pars, u:float, wage:float) ->float:
    """
    given a marginal utility u and a current wage return the consumption that yields that utility
    inverse of the derivative of the utility function with respect to consumption
    """
    alpha = myPars.alpha
    sigma = myPars.sigma_util
    const =(myPars.phi_n * (1 - alpha)) / (wage* alpha) #this denom should never be zero, wage is a product of exp != 0
    inner_exponent =(alpha*(-sigma)+alpha+sigma-1)
    c = ((u*const**inner_exponent) / alpha)**(-1/sigma)
    return c

@njit
def infer_c(myPars: Pars, curr_wage:float, age: int, lab_fe_ind: int, health_ind: int, H_type_perm_ind: int, c_prime0:float, c_prime1:float ) -> float: 
    """
    infer what current consumption should be given future consumption, curr wage, and the curr state space
    calculated expectation on rhs of euler, calc the rest of the rhs, then invert util_c to get the curr c on the lhs
    """
    fut_wage0 = wage(myPars, age+1, lab_fe_ind, 0)
    fut_wage1 = wage(myPars, age+1, lab_fe_ind, 1)
    prob0 = myPars.H_trans[H_type_perm_ind, age, health_ind, 0]
    prob1 = myPars.H_trans[H_type_perm_ind, age, health_ind, 1]
    expect_util_c_prime = (prob0 * util_c(myPars, c_prime0, fut_wage0)) + (prob1 * util_c(myPars, c_prime1, fut_wage1))
    expect = expect_util_c_prime
    rhs = myPars.beta *(1 + myPars.r) * expect
    c = util_c_inv(myPars, rhs, curr_wage)
    return max(myPars.c_min, c)  

@njit
def solve_lab_a(myPars: Pars, c:float, a_prime:float,  curr_wage:float, health_ind: int) -> float:
    """
    solve for labor and assets given consumption and wage
    given current choice of c and a_prime, as well the state's wage and health 
    """
    health = myPars.H_grid[health_ind]
    leis = leis_giv_c(myPars, c, curr_wage) 
    leis = min(myPars.leis_max, leis)
    leis = max(myPars.leis_min, leis)
    lab = lab_giv_leis(myPars, leis, health)
    lab = min(myPars.lab_max, lab)
    lab = max(myPars.lab_min, lab)
    a = (c + a_prime - curr_wage*lab)/(1 + myPars.r)
    return lab, a

@njit
def invert_lab (myPars : Pars, c:float, curr_wage:float, health:float) -> float:
    """
    invert the foc to get labor given consumption and wage
    """
    rhs = (curr_wage/myPars.phi_n) * util_c(myPars, c, curr_wage)
    leis = util_leis_inv(myPars, rhs, c)
    lab = lab_giv_leis(myPars, leis, health)
    return lab

@njit
def util_leis_inv(myPars: Pars, u:float, c:float) -> float:
    """
    invert the utility function with respect to leisure
    """
    alpha = myPars.alpha
    sigma = myPars.sigma_util
    phi_n = myPars.phi_n
    phi_H = myPars.phi_H
    out_exp = 1 / (alpha*sigma - alpha - sigma)
    inside =(u * c ** (-alpha * (1-sigma)))/(1 - alpha)
    return inside ** out_exp
    
@njit
def lab_star(myPars: Pars, a_prime:float, a:float, health:float, wage:float)-> float:
    """
    return the optimal labor decision given an asset choice a_prime and a current asset level, health status, and wage
    """
    lab =  ((myPars.alpha/myPars.phi_n)*(1 - myPars.phi_H*(1-health))
            + ((myPars.alpha - 1)/wage)*((1 + myPars.r)*a - a_prime))
    lab = min(myPars.lab_max, lab)
    return max(myPars.lab_min, lab)

@njit
def c_star(myPars: Pars, a_prime:float, a:float, health:float, wage:float) -> float:
    """
    return the optimal consumption given an asset choice a_prime and a current asset level, health status, and wage
    """
    c_star = myPars.alpha*((wage/myPars.phi_n)*(1-myPars.phi_H*(1.0-health)) + (1 + myPars.r)*a - a_prime)
    return max(myPars.c_min, c_star)

@njit
def wage(myPars: Pars,  age: int, lab_fe_ind: int, h_ind: int) -> float:
    """
    calculate the wage given health, age, lab_fe, and nu i.e. the shocks
    """
    age_comp = tb.cubic(age, myPars.wage_coeff_grid[lab_fe_ind])
    health_comp = myPars.H_grid[h_ind] * myPars.wH_coeff
    my_wage = age_comp + health_comp
    my_wage = np.exp(my_wage)
    return max(myPars.wage_min, my_wage)

@njit
def gen_wages(myPars: Pars) -> np.ndarray:
    """
    generate the wage grid
    """
    wage_grid = np.zeros((myPars.lab_fe_grid_size, myPars.H_grid_size,  myPars.J))
    for j in range(myPars.J):
        for h_ind in range(myPars.H_grid_size):
            for lab_fe_ind in range(myPars.lab_fe_grid_size):
                wage_grid[lab_fe_ind, h_ind, j] = wage(myPars, j, lab_fe_ind, h_ind)
    return wage_grid

@njit
def gen_weighted_wage_hist(myPars: Pars, myShocks: Shocks) -> np.ndarray:
    """
    generate the fully weighted wage history this function weights 
    by simulation draws, H_type_perm_weights, and lab_fe_weights
    myPars.H_beg_pop_weights_by_H_type weights accounted for in myPars.H_hist 
    """
    sim_weight = 1/myPars.sim_draws
    H_hist = myShocks.H_hist
    wage_hist = np.empty(H_hist.shape)
    for lab_fe_ind in range(myPars.lab_fe_grid_size):
        for H_type_perm_ind in range(myPars.H_type_perm_grid_size):
            type_pop_weight = myPars.lab_fe_weights[lab_fe_ind] * myPars.H_type_perm_weights[H_type_perm_ind]
            for sim_ind in range(myPars.sim_draws):
                for j in range(myPars.J):
                    my_H = H_hist[lab_fe_ind, H_type_perm_ind, sim_ind, j]
                    wage_unweighted = wage(myPars, j, lab_fe_ind, my_H)
                    wage_weighted = wage_unweighted * type_pop_weight*sim_weight
                    wage_hist[lab_fe_ind, H_type_perm_ind, sim_ind, j] = wage_weighted
    return wage_hist

@njit
def gen_wage_hist(myPars: Pars, myShocks: Shocks) -> np.ndarray:
    """
    generate the wage history unweighted
    though H_hist is weighted by H_beg_pop_weights_by_H_type
    """
    wage_hist = np.empty((myPars.lab_fe_grid_size, myPars.H_type_perm_grid_size, myPars.sim_draws, myPars.J))
    for lab_fe_ind in range(myPars.lab_fe_grid_size):
        for H_type_perm_ind in range(myPars.H_type_perm_grid_size):
            for sim_ind in range(myPars.sim_draws):
                for j in range(myPars.J):
                    my_H = myShocks.H_hist[lab_fe_ind, H_type_perm_ind, sim_ind, j]
                    wage_hist[lab_fe_ind, H_type_perm_ind, sim_ind, j] = wage(myPars, j, lab_fe_ind, my_H)
    return wage_hist

# should probable jit this
@njit
def gen_weighted_sim(myPars: Pars, lc_moment_sim: np.ndarray) -> np.ndarray:
    """
    generate the weighted simulation
    weigthed by simulation draws, H_type_perm_weights, and lab_fe_weights
    """
    sim_draw_weights = np.ones(myPars.sim_draws)/myPars.sim_draws
    weighted_sim = lc_moment_sim * myPars.lab_fe_weights[:, np.newaxis, np.newaxis, np.newaxis]
    weighted_sim = weighted_sim * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis, np.newaxis]
    weighted_sim = weighted_sim * sim_draw_weights[np.newaxis, np.newaxis, :, np.newaxis]
    return weighted_sim

# should probable jit this
@njit
def wmean_non_zero(myPars: Pars, sim_with_zeros: np.ndarray) -> float:
    '''
    calculate the weighted mean of the lc simulation ignoring zeros
    '''
    non_zero_mask = (sim_with_zeros != 0)
    wnon_zero_mask = non_zero_mask * myPars.lab_fe_weights[:, np.newaxis, np.newaxis, np.newaxis] * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis, np.newaxis]
    wN = np.sum(wnon_zero_mask)

    wsim = sim_with_zeros * myPars.lab_fe_weights[:, np.newaxis, np.newaxis, np.newaxis] * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis, np.newaxis]
    wsum = np.sum(wsim*non_zero_mask)
    # wsum = np.sum(wsim[non_zero_mask])
    # for i in wsim.size:
        # if wsim.flat[i] == 0:
            # wsim.flat[i] = np.nan
    # wsum = np.nansum(wsim*non_zero_mask)


    return wsum / wN

@njit
def recover_wage(myPars: Pars, c:float, lab:float, a_prime:float, a:float) -> float: #this will divide by zero if lab = 0
    """
    recover the wage given consumption, labor, and assets
    """
    return (c + a_prime - (1 + myPars.r)*a) / lab


# put run if main funciton
if __name__ == "__main__":
    
    path = "SomePath"
    myPars = Pars(path)
    j = 5
    lab_fe_ind = 1
    health_ind = 1
    nu_ind = 1
    print(gen_weighted_wage_hist(myPars))
                    
