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
import my_toolbox as tb
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
    #return labor
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
    leis = constant * c
    #leis = min(myPars.leis_max, leis)
    #return max(myPars.leis_min, leis)
    return leis
   
    

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
@njit
def mb_lab(myPars: Pars, c: float, wage: float, labor: float, health: float) -> float:
    """
    marginal benefit of labor
    """
    leis = leis_giv_lab(myPars, labor, health)
    return wage * util_c_giv_leis(myPars, c, leis)

@njit
def mc_lab(myPars: Pars, c: float, labor: float, health: float) -> float:
    """
    marginal cost of labor
    """
    leis = leis_giv_lab(myPars, labor, health)
    return myPars.phi_n * util_leis_giv_c(myPars, leis, c)


@njit
def util_leis_giv_c(myPars: Pars, leis: float, c: float) -> float:
    """
    derivative of utility function with respect to leisure
    """
    return (1-myPars.alpha)*c**(myPars.alpha)*leis**(-myPars.alpha)/(c**myPars.alpha*leis**(1-myPars.alpha))**myPars.sigma_util

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
    #return max(myPars.c_min, c_prime)
    #try:
    #fut_wage = wage(myPars, health, age+1, lab_fe, nu)    
    fut_wage = curr_wage
    # except ZeroDivisionError:
    #     print("wage: Cannot divide by zero.")
    #     print("curr-wage:", curr_wage, "c_prime:", c_prime, "health: ", health, "age: ", age, "lab_fe: ", lab_fe, "nu: ", nu)
    #     sys.exit()
        
    # try:
    util_c_prime = util_c(myPars, c_prime, fut_wage)
    # except ZeroDivisionError:
    #     print("util_c: Cannot divide by zero.")
    #     print("curr-wage:", curr_wage, "c_prime:", c_prime, "health: ", health, "age: ", age, "lab_fe: ", lab_fe, "nu: ", nu)
    #     sys.exit()
    
    expect = util_c_prime
    rhs = myPars.beta *(1 + myPars.r) * expect
    
    #try:
    c = util_c_inv(myPars, rhs, curr_wage)
    # except ZeroDivisionError:
    #     print("util_c_inverse: Cannot divide by zero.")
    #     print("curr-wage:", curr_wage, "c_prime:", c_prime, "health: ", health, "age: ", age, "lab_fe: ", lab_fe, "nu: ", nu)
    #     sys.exit()

    #c = c_prime
    #return c
    return max(myPars.c_min, c)  

# given current choice of c and a_prime, as well the state's wage and health 
@njit
def solve_lab_a(myPars: Pars, c: float, a_prime: float,  curr_wage: float, health: float) -> float:
    """
    solve for labor and assets given consumption and wage
    """
    leis = leis_giv_c(myPars, c, curr_wage) 
    leis = min(myPars.leis_max, leis)
    leis = max(myPars.leis_min, leis)

    lab = lab_giv_leis(myPars, leis, health)
    #lab = invert_lab(myPars, c, curr_wage, health)
    lab = min(myPars.lab_max, lab)
    lab = max(myPars.lab_min, lab)

    a = (c + a_prime - curr_wage*lab)/(1 + myPars.r)
    #a = min(myPars.a_max, a)
    #a = max(myPars.a_min, a)
    return lab, a

@njit
def invert_lab (myPars : Pars, c: float, curr_wage: float, health: float) -> float:
    """
    invert the foc to get labor given consumption and wage
    """
    rhs = (curr_wage/myPars.phi_n) * util_c(myPars, c, curr_wage)
    leis = util_leis_inv(myPars, rhs, c)
    lab = lab_giv_leis(myPars, leis, health)
    return lab

@njit
def util_leis_inv(myPars: Pars, u: float, c: float) -> float:
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
    


# return the optimal labor decision given an asset choice a_prime and a current asset level, health status, and wage
@njit
def lab_star(myPars: Pars, a_prime: float, a: float, health: float, wage: float)-> float:
    lab =  ((myPars.alpha/myPars.phi_n)*(1 - myPars.phi_H*(1-health))
            + ((myPars.alpha - 1)/wage)*((1 + myPars.r)*a - a_prime))
    lab = min(myPars.lab_max, lab)
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
def wage(myPars: Pars,  age: int, lab_fe_ind: int, h_ind: float,  nu_ind: float) -> float:
    """
    wage process
    """
    return tb.cubic(age, myPars.wage_coeff_grid[lab_fe_ind])
    
    #det_wage = det_wage(myPars, health, age)
    # det_wage = 1.0
    # nu = 0.0
    # return  det_wage* np.exp(lab_fe) * np.exp(nu)
@njit
def gen_wages(myPars: Pars) -> np.ndarray:
    """
    generate the wage grid
    """
    #initialize the wage grid
    wage_grid = np.zeros((myPars.lab_FE_grid_size, myPars.H_grid_size, myPars.nu_grid_size, myPars.J))
    #loop through the wage grid
    for j in range(myPars.J):
        for h_ind in range(myPars.H_grid_size):
            for nu_ind in range(myPars.nu_grid_size):
                for lab_fe_ind in range(myPars.lab_FE_grid_size):
                    wage_grid[lab_fe_ind, h_ind, nu_ind, j] = wage(myPars, j, lab_fe_ind, h_ind, nu_ind)
    return wage_grid

@njit
def recover_wage(myPars: Pars, c: float, lab: float, a_prime: float, a: float) -> float: #this will divide by zero if lab = 0
    """
    recover the wage given consumption, labor, and assets
    """
    return (c + a_prime - (1 + myPars.r)*a) / lab

if __name__ == "__main__":
    #initialize the parameters
    path = "SomePath"
    myPars = Pars(path)

    myWage: float = 25.0
    health: float = 1.0

    matc = np.linspace(0.0, 10.0, 10)
    c = 5.0
    mat_leis = np.linspace(myPars.leis_min, myPars.leis_max, 10)
    mat_c = np.linspace(myPars.c_min, 10 + myPars.c_min, 10)
    for c in mat_c:
        #print(f'Consumption: {c}, Leisure: {leis_giv_c(myPars, c, myWage)}')
        util = util_c(myPars, c, myWage)
        rhs = myPars.beta *(1 + myPars.r) * util
        #print(f'Consumption: {c}, Utility: {util}')
        c_recovered = util_c_inv(myPars, rhs, myWage)
        print(f'Consumption: {c}, Utility: {util}, RHS: {rhs}, Recovered Consumption: {c_recovered}')



    for health in myPars.H_grid:
        for j in range(myPars.J):
            for nu in myPars.nu_grid:
                for lab_fe in myPars.lab_FE_grid:
                    for c_prime in mat_c:
                        c = infer_c(myPars, myWage, j, lab_fe, health, nu, c_prime)
                        print(f'C_prime {c_prime}, recoverred C {c}')

                    #myWage = wage(myPars, j, lab_fe, health, nu)
                    #print(f'The wage for state health={health}, age={j}, lab_fe ={lab_fe}, nu = {nu} is \n {myWage}')
                    
