"""
My Model 2 - basic model no uncertainty

Contains the model equations and derivatives to be used by solver.py and others

Author: Ben Boyajian
Date: 2024-05-29 20:16:01
"""

# Import packages
import time
import numpy as np
from pars_shocks_and_wages import Pars
import my_toolbox
from numba import njit, guvectorize, prange 
from interpolation import interp
import sys

#convert labor to leisure within period
@njit
def leis_giv_lab(myPars: Pars, labor: float, health: float) -> float:
    """
    encodes the time endowment constraint
    """
    leisure = 1.0 - labor*myPars.phi_n - (1-health)*myPars.phi_H
    leisure = min(myPars.leis_max, leisure)
    return max(myPars.leis_min, leisure)

#convert leisure to labor within period
@njit
def lab_giv_leis(myPars: Pars, leisure: float, health: float) -> float:
    """
    encodes the time endowment constraint
    """
    labor = (1.0 - leisure - (1-health)*myPars.phi_H) / myPars.phi_n #this denom should never be zero, phi_n != 0
    labor = min(myPars.lab_max, labor)
    return max(myPars.lab_min, labor)

#convert labor to consumption within period
@njit
def leis_giv_c(myPars: Pars, c: float, wage: float) -> float:
    """
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for leisure given current period consumption
    """

    constant = (myPars.phi_n * (1 - myPars.alpha)) / (wage * myPars.alpha) #this denom should !=0, wage is a product of exp != 0  
    return constant * c

#converty leisure to consumption within period
@njit
def c_giv_leis(myPars: Pars,  leis: float, wage: float) -> float:
    """
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for consumption given current period leisure
    """
    constant = (wage * myPars.alpha) / (myPars.phi_n * (1 - myPars.alpha)) #this denom should never be zero, alpha != 1 and phi_n != 0
    return constant * leis

#utility function given leisure and consumption
@njit
def util_giv_leis(myPars: Pars, c: float, leis: float) -> float:
    """
    utility function
    """
    sig = myPars.sigma_util
    alpha = myPars.alpha
    return (1/(1-sig)) * ((c**alpha) * (leis**(1-alpha))) ** (1-sig)

#derivative of utility function with respect to consumption given consumption and leisure
@njit
def util_c_giv_leis(myPars: Pars, c: float, leis: float) -> float:
    """
    derivative of utility function with respect to consumption
    """
    sig = myPars.sigma_util
    alpha = myPars.alpha
    return alpha*c**(alpha - 1)*leis**(1 - alpha)/(c**alpha*leis**(1 - alpha))**sig #this denom is 0 if c or leis is 0

#deriveative of utility function with respect to consumption given consumption and health
@njit
def util_c(myPars: Pars, c: float, wage: float) -> float:
    """
    derivative of utility function with respect to consumption
    """
    leis = leis_giv_c(myPars, c, wage) #this can also be done explicitly in one function
    return util_c_giv_leis(myPars, c, leis)

#inverse of the derivative of the utility function with respect to consumption
@njit
def util_c_inv(myPars: Pars, u: float, wage: float) ->float:
    """
    given a marginal utility u and a current wage return the consumption that yields that utility
    """
    alpha = myPars.alpha
    sigma = myPars.sigma_util

    const =(myPars.phi_n * (1 - alpha)) / (wage* alpha) #this denom should never be zero, wage is a product of exp != 0
    inner_exponent =(alpha*(-sigma)+alpha+sigma-1)

    c = ((u*const**inner_exponent) / alpha)**(-1/sigma)
    return c

# infer what current consumption should be given future consumption, curr wage, and the curr state space
@njit
def infer_c(myPars: Pars, curr_wage: float, age: int, lab_fe: float, health: float, nu: float, c_prime: float ) -> float: 
    """
    calculated expectation on rhs of euler, calc the rest of the rhs, then invert util_c to get the curr c on the lhs
    """
    #try:
    fut_wage = wage(myPars, health, age+1, lab_fe, nu)    
    # except ZeroDivisionError:
    #     print("wage: Cannot divide by zero.")
    #     print("curr-wage:", curr_wage, "c_prime:", c_prime, "health: ", health, "age: ", age, "lab_fe: ", lab_fe, "nu: ", nu)
    #     sys.exit()
        
    # try:
    util = util_c(myPars, c_prime, fut_wage)
    # except ZeroDivisionError:
    #     print("util_c: Cannot divide by zero.")
    #     print("curr-wage:", curr_wage, "c_prime:", c_prime, "health: ", health, "age: ", age, "lab_fe: ", lab_fe, "nu: ", nu)
    #     sys.exit()
    
    expect = util
    rhs = myPars.beta *(1 + myPars.r) * expect
    
    #try:
    c = util_c_inv(myPars, rhs, curr_wage)
    # except ZeroDivisionError:
    #     print("util_c_inverse: Cannot divide by zero.")
    #     print("curr-wage:", curr_wage, "c_prime:", c_prime, "health: ", health, "age: ", age, "lab_fe: ", lab_fe, "nu: ", nu)
    #     sys.exit()

    #return c
    return max(myPars.c_min, c)  

# given current choice of c and a_prime, as well the state's wage and health 
@njit
def solve_lab_a(myPars: Pars, c: float, a_prime: float,  curr_wage: float, health: float) -> float:
    """
    solve for labor and assets given consumption and wage
    """
    lab = lab_giv_leis(myPars, leis_giv_c(myPars, c, curr_wage), health)
    lab = min(myPars.lab_max, lab)
    lab = max(myPars.lab_min, lab)

    a = (c + a_prime - curr_wage*lab)/(1 + myPars.r)
    a = max(myPars.a_min, a)
    a = min(myPars.a_max, a) 
    return lab, a

# return the optimal labor decision given an asset choice a_prime and a current asset level, health status, and wage
@njit
def lab_star(myPars: Pars, a_prime: float, a: float, health: float, wage: float)-> float:
    lab =  ( (myPars.alpha/myPars.phi_n)*(1 - myPars.phi_H*health)
            + ((myPars.alpha - 1)/wage)*((1 + myPars.r)*a - a_prime))
    
    return max(myPars.lab_min, lab)
#calulate deterministic part of the wage given health and age 

@njit
def det_wage(myPars: Pars, health: float, age: int) -> float:
    """
    deterministic part of the wage process
    """
    age_comp = myPars.w_age*age + myPars.w_age_2*age**2 + myPars.w_age_3*age**3
    health_comp = myPars.w_good_health*health + myPars.w_good_health_age*health*age
    return np.exp(myPars.w_determ_cons + age_comp + health_comp)

#calculate the wage given health, age, lab_fe, and nu i.e. the shocks
@njit
def wage(myPars: Pars,  age: int, lab_fe: float, health: float,  nu: float) -> float:
    """
    wage process
    """
    #det_wage = det_wage(myPars, health, age)
    det_wage = 1.0
    nu = 0.0
    return  det_wage* np.exp(lab_fe) * np.exp(nu) 

if __name__ == "__main__":
    #initialize the parameters
    myPars = Pars()

    #myWage: float = 25.0
    health: float = 1.0

    matc = np.linspace(0.0, 10.0, 10)
    c = 5.0
    mat_leis = np.linspace(myPars.leis_min, myPars.leis_max, 10)
    mat_c = np.linspace(myPars.c_min, 10 + myPars.c_min, 10)

    for health in myPars.H_grid:
        for j in range(myPars.J):
            for nu in myPars.nu_grid:
                for lab_fe in myPars.lab_FE_grid:
                   myWage = wage(myPars, j, lab_fe, health, nu)
                   print(f'The wage for state health={health}, age={j}, lab_fe ={lab_fe}, nu = {nu} is \n {myWage}')
