"""
Created on 2024-05-13 09:49:08

@author: Ben Boyaian
"""
#import stuff
import time
import numpy as np
from pars_shocks_and_wages import Pars, Shocks
import my_toolbox
from math import exp,sqrt,log
from numba import njit, guvectorize, prange 
from interpolation import interp

#####define the pieces of the HH problem
#calculate the deterministic part of the wage
@njit
def det_wage(myPars: Pars, age, health) :
    """returns the deterministic part of the wage"""
    age_comp = myPars.w_determ_cons + myPars.w_age*age + myPars.w_age_2*age*age + myPars.w_age_3*age*age*age 
    #gonna ignore average health which is in Capatina and the iniatialization of theis program for now, will work in more health states when i have composite types working.
    health_comp = myPars.w_good_health*(1-health) + myPars.w_good_health_age*age*(1-health)
    wage = age_comp + health_comp
    #print("Comp is ", comp) 
    return wage

#calculate final wage
@njit
def wage(myPars: Pars, age, fixed_effect, health, persistent_shock ) :
    return exp(det_wage(myPars, age, health)) * exp(fixed_effect) * exp(persistent_shock)

#return leisure given labor choice and health status
@njit
def leis_giv_lab(myPars: Pars, labor, health):
    leisure = 1 - labor*myPars.phi_n -health*myPars.phi_H
    leisure = min(myPars.leis_max, leisure)
    return max(myPars.leis_min, leisure)

@njit
def lab_giv_leis(myPars: Pars, leisure, health):
    labor = (1-leisure-health*myPars.phi_H)/(myPars.phi_n)
    labor = min(myPars.lab_max, labor)
    return max(myPars.lab_min, labor)

@njit
def leis_giv_c(myPars : Pars, c, wage) :
    """
    To do this we want to leverage the static equation:
        phi_n * util_leis = wage * util_c
        manipulating this equations gives us a relatively simple equation for leisure given current period consumption
    """
    constant = (myPars.phi_n * (1 - myPars.alpha)) / (wage * myPars.alpha)
    return constant * c

### this is basically invert_n in adams code i think
@njit
def lab_giv_c(myPars : Pars, c, wage, health) :
    """
    to do this we get the leisure given the c and wage
    then convert that to labor given the calculated leisure and the given health
    """
    leis =leis_giv_c(myPars, c, wage)
    return lab_giv_leis(myPars, c, health)   
     
# the utility function given a consumption and labor choice and a current health status
# should i do anything special to handle negative values of consumption leisure and or labor?
@njit
def util(myPars: Pars, consum, labor, health) :
    leisure = leis_giv_lab(myPars, labor, health)
    util = ((consum**(myPars.alpha)*leisure**(1-myPars.alpha))**(1-myPars.sigma_util))/(1-myPars.sigma_util)
    #print("VALID CHOICES: ", consumption_choice, " and ", leisure_choice, " util is: ", util)
    return util

#derivative of the util function wrt consumption c calculated by giving leisure
@njit
def util_c_giv_leis(myPars: Pars, c, leis):
    #assign/unpack variable and param values
    #leis = myPars.leisure_giv(labor,health)
    alpha,sigma=myPars.alpha,myPars.sigma_util
    #evaluate the derivative determined analytically
    eval = alpha*c**(alpha - 1)*leis**(1 - alpha)/(c**alpha*leis**(1 - alpha))**sigma
    return eval

#derivative of the utlity function with respect to c
@njit
def util_c(myPars : Pars, c, wage) :
    leis = leis_giv_c(myPars, c, wage)
    return util_c_giv_leis(myPars, c, leis)

# inverse of util_c(c) = x
#i.e. given x (and the states wage) return the c that produces util_c(c) = x
@njit
def util_c_inv(myPars : Pars, x, wage) :
    alpha, phi_n, sigma = myPars.alpha, myPars.phi_n, myPars.sigma_util
    constant = (phi_n * (1 - alpha)) / (wage * alpha)
    ins_exponent = (-sigma * alpha) + alpha + sigma - 1
    return ((x * constant ** ins_exponent) / alpha ) ** (-1/sigma)

#should this return wages also??
@njit(parallel = True)
def mat_future_cps(myPars : Pars, mat_cp_flat_shocks) :

    #initialie a matrix to store the points of shocks to evaluate/interpolate
    interp_eval = myPars.interp_eval_points
    
    #initialize a matrix to store the interpolated c_primes and wages
    interp_c_primes = myPars.interp_c_prime_grid
   
   #loop through shock space and interpolate the c_prime for each shock combination
    for ind_shock_comb in prange(myPars.H_by_nu_size):
        H_ind, nu_ind = my_toolbox.D2toD1(ind_shock_comb, myPars.H_grid_size, myPars.nu_grid_size) #might not need depending on structure
        
        # get the value of that shock combination so that it can be interpolated
        interp_eval[0] = myPars.nu_grid[nu_ind]
        
        # interpolate the value for c_prime for that shock combination given a matrix of shock combinations 
        # and a matrix of the resulting c_primes
        # store that value in the right place in the interp_c_primes return matrix
        interp_c_primes[ind_shock_comb] = interp(myPars.nu_grid, mat_cp_flat_shocks[H_ind, :], interp_eval[0])

    return interp_c_primes
    

# take the expecteation over the possible c_primes
@njit
def expect_util_c_prime(myPars : Pars, mat_cp_flat_shocks, j, lab_fe, health, nu) :
    #state_probs = myPars.H_by_nu_flat_trans
    lab_fe_ind = np.where(myPars.lab_fe_grid == lab_fe)[0][0]
    h_ind = np.where(myPars.H_grid == health)[0][0]
    nu_ind = np.where(myPars.nu_grid == nu)[0][0]
    
    h_probs = myPars.H_trans[h_ind, :]
    nu_probs = myPars.nu_trans[nu_ind, :]

    shock_probs = np.outer(h_probs, nu_probs).flatten()
    
    #maybe there is a slick way to recover wages here... maybe i really do need matrix of them bopping around
    possible_wages = np.zeros(myPars.H_by_nu_size)
    for i in range(myPars.H_by_nu_size) :
        H_ind, nu_ind = my_toolbox.D2toD1(i, myPars.H_grid_size, myPars.nu_grid_size)
        next_age = j + 1
        possible_wages[i] = wage(myPars, next_age, lab_fe, myPars.H_grid[H_ind], myPars.nu_grid[nu_ind]) 

    #mat_wages_by_shocks = myPars.wage_grid[j+1, lab_fe_ind, :, :]
    #possible_wages = mat_wages_by_shocks.flatten()

    possible_c_primes = mat_future_cps(myPars, mat_cp_flat_shocks)


    return np.sum(shock_probs * util_c(myPars, possible_c_primes, possible_wages))


@njit
def infer_c(myPars : Pars,  curr_wage, mat_cp_flat_shocks, j, lab_fe, health, nu, c_prime_test) :
    #expect = util_c(myPars, c_prime, wage) #need to change this to actually do expectation over shocks/states 
    #expect = expect_util_c_prime(myPars, mat_cp_flat_shocks, j, lab_fe, health, nu)
    expect = util_c(myPars, c_prime_test, wage(myPars, j+1, lab_fe, health, nu))
    rhs = myPars.beta * (1 + myPars.r) * expect
    return util_c_inv(myPars, rhs, curr_wage)

    

@njit
def util_leis_giv_c(myPars: Pars, c, leis) :
    #assign/unpack variable and param values
    #leis=leisure_giv(myPars, lab, health)
    a,s=myPars.alpha,myPars.sigma_util
    #evaluate the derivative determined  analytically
    eval = c**a*(1 - a)/(leis**a*(c**a*leis**(1 - a))**s)
    return eval

@njit
def util_leis_inv_giv_c(myPars : Pars, util_leis, c) :
    alpha = myPars.alpha
    sigma = myPars.sigma_util
    inverse = ((util_leis/c**(alpha*(1 - sigma)))/(1 - alpha))**(1/(alpha*sigma - alpha - sigma)) 
    return inverse

    

#derivative of the utility function  w.r.t. labor n
#utility is negative since ther is disutility form labor
@njit
def util_lab(myPars: Pars, c, lab, health) :
    return -myPars.phi_n * util_leis_giv_c(myPars,  c, lab, health)
    

# return the orpimal consumption decision given an asset choice a_prime and a current asset level, health statues, and labor income
@njit
def c_star(myPars: Pars, a_prime, a, health, wage) :
    consum = myPars.alpha*((wage/myPars.phi_n)*(1 - myPars.phi_H*health)+(1 + myPars.r)*a - a_prime)
    return max(myPars.c_min, consum)

# return the optimal labor decision given an asset choice a_prime and a current asset level, health statues, and labor income
@njit
def n_star(myPars: Pars, a_prime, a, health, wage) :
    lab =  ( (myPars.alpha/myPars.phi_n)*(1 - myPars.phi_H*health)
            + ((myPars.alpha - 1)/wage)*((1 + myPars.r)*a - a_prime)
            )
    return max(myPars.lab_min, lab)

# return the orpimal leisure decision given an asset choice a_prime and a current asset level, health statues, and labor income
@njit
def l_star(myPars: Pars, a_prime, a, health, wage) :
    leis = leis_giv_lab(myPars, n_star(a_prime, a, health, wage), health)
    return max(myPars.leis_min, leis)




# use the budget constraint to back out current asset stock given the other partts of the BC
@njit
def infer_a(myPars :Pars, a_prime, c, labor, wage) :
    #BC is c + a_prime = labor_inc*labor +(1+myPar.r)*a_curr so
    return ( c + a_prime - wage*labor) / (1+myPars.r)
     
#solve for labor and assets given a_prime, c, wage, health
@njit
def solve_lab_a(myPars : Pars, a_prime, c, wage, health):
    """
    solve for labor given c and wage using the static condition between c and leis/labor
    then infer the current assets a that balance the budget constraint given 
    the future assets a_prime, c, labor, and labor incometo balance the BC
    """
    lab = lab_giv_c(myPars, c, wage, health)
    a = infer_a(myPars, a_prime, c, lab, wage)
    return lab, a
    




if __name__ == "__main__":

    print("Running main")
    start_time = time.time() 


 

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
