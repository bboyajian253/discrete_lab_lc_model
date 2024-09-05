"""
Created on 2024-05-13 09:49:08

@author: Ben Boyaian
"""
#import stuff
import time
import numpy as np
from pars_and_shocks import pars
import my_toolbox
from math import exp,sqrt,log
from numba import njit, guvectorize, prange # NUMBA speed up quite a lot, see the functions that have the decorator just above

#####define the pieces of the HH problem
#return leisure given labor choice and health status
@njit
def leis_giv_lab(myPars: pars, labor, health):
    leisure = 1 - labor*myPars.phi_n -health*myPars.phi_H
    leisure = min(myPars.leis_max, leisure)
    return max(myPars.leis_min, leisure)

@njit
def lab_giv_leis(myPars: pars, leisure, health):
    labor = (1-leisure-health*myPars.phi_H)/(myPars.phi_n)
    labor = min(myPars.lab_max, labor)
    return max(myPars.lab_min, labor)

@njit
def leis_giv_c(myPars : pars, c, lab_inc) :
    """
    To do this we want to leverage the static equation:
        phi_n * util_leis = lab_inc * util_c
        manipulating this equations gives us a relatively simple equation for leisure given current period consumption
    """
    constant = (myPars.phi_n * (1 - myPars.alpha)) / (lab_inc * myPars.alpha)
    return constant * c

### this is basically invert_n in adams code i think
@njit
def lab_giv_c(myPars : pars, c, lab_inc, health) :
    """
    to do this we get the leisure given the c and lab_inc
    then convert that to labor given the calculated leisure and the given health
    """
    leis =leis_giv_c(myPars, c, lab_inc)
    return lab_giv_leis(myPars, c, health) 
    

# @njit
# def leis_giv_c(myPars: pars, c, lab_inc) :
#     constant = (myPars.phi_n/lab_inc) * ((1 - myPars.alpha)/myPars.alpha)
#     return constant * c
     
# the utility function given a consumption and labor choice and a current health status
# should i do anything special to handle negative values of consumption leisure and or labor?
@njit
def util(myPars: pars, consum, labor, health) :
    leisure = leis_giv_lab(myPars, labor, health)
    util = ((consum**(myPars.alpha)*leisure**(1-myPars.alpha))**(1-myPars.sigma_util))/(1-myPars.sigma_util)
    #print("VALID CHOICES: ", consumption_choice, " and ", leisure_choice, " util is: ", util)
    return util
#derivative of the util function wrt consumption c
@njit
def util_c_giv_leis(myPars: pars, c, leis):
    #assign/unpack variable and param values
    #leis = myPars.leisure_giv(labor,health)
    alpha,sigma=myPars.alpha,myPars.sigma_util
    #evaluate the derivative determined analytically
    eval = alpha*c**(alpha - 1)*leis**(1 - alpha)/(c**alpha*leis**(1 - alpha))**sigma
    return eval

@njit
def util_c(myPars : pars, c_prime) :
    leis = leis_giv_c(c_prime)
    return util_c_giv_leis(myPars,c_prime,leis)
#derivative of the utlity function with respect to leisure

# @njit
# def util_c_inv_giv_leis(myPars : pars, util_c, leis) :
#     #returns the c that yields the marginal utility level util_c given a leisure choice
#     #leis = leisure_giv(labor, health)
#     alpha = myPars.alpha
#     sigma = myPars.sigma_util
#     inverse = (util_c/((alpha*leis**((1 - alpha)*(1 - sigma)))))**(1/(-alpha*sigma + alpha - 1))
#     return inverse

# @njit
# def inv_c_giv_leis(myPars : pars, c_prime, leis) :
#     expect = util_c_giv_leis(myPars, c_prime, leis) #this needs to change to actually take an expectation 
#     rhs = myPars.beta * (1 + myPars.r) * expect  
#     inverse = util_c_inv_giv_leis(myPars, rhs, leis)
#     return max(myPars.c_min, inverse)

# @njit
# def inv_c(myPars : pars, c_prime, lab_inc):
#     leis = leis_giv_c(myPars, c_prime, lab_inc)
#     return inv_c_giv_leis(myPars, c_prime, leis)
    
@njit
def inv_c(myPars : pars, c_prime, lab_inc ) :
    pass

@njit
def util_leis_giv_c(myPars: pars, c, leis) :
    #assign/unpack variable and param values
    #leis=leisure_giv(myPars, lab, health)
    a,s=myPars.alpha,myPars.sigma_util
    #evaluate the derivative determined  analytically
    eval = c**a*(1 - a)/(leis**a*(c**a*leis**(1 - a))**s)
    return eval

@njit
def util_leis_inv_giv_c(myPars : pars, util_leis, c) :
    alpha = myPars.alpha
    sigma = myPars.sigma_util
    inverse = ((util_leis/c**(alpha*(1 - sigma)))/(1 - alpha))**(1/(alpha*sigma - alpha - sigma)) 
    return inverse

    

#derivative of the utility function iwht respect to labor n
#utility is negative since ther is disutility form labor
@njit
def util_lab(myPars: pars, c, lab, health) :
    return -myPars.phi_n * util_leis_giv_c(myPars,  c, lab, health)
    

# return the orpimal consumption decision given an asset choice a_prime and a current asset level, health statues, and labor income
@njit
def c_star(myPars: pars, a_prime, a, health, lab_inc) :
    consum = myPars.alpha*((lab_inc/myPars.phi_n)*(1 - myPars.phi_H*health)+(1 + myPars.r)*a - a_prime)
    return max(myPars.c_min, consum)

# return the optimal labor decision given an asset choice a_prime and a current asset level, health statues, and labor income
@njit
def n_star(myPars: pars, a_prime, a, health, lab_inc) :
    lab =  ( (myPars.alpha/myPars.phi_n)*(1 - myPars.phi_H*health)
            + ((myPars.alpha - 1)/lab_inc)*((1 + myPars.r)*a - a_prime)
            )
    return max(myPars.lab_min, lab)

# return the orpimal leisure decision given an asset choice a_prime and a current asset level, health statues, and labor income
@njit
def l_star(myPars: pars, a_prime, a, health, lab_inc) :
    leis = leis_giv_lab(myPars, n_star(a_prime, a, health, lab_inc), health)
    return max(myPars.leis_min, leis)

@njit
def det_lab_inc(myPars: pars, age, health) :
    """returns the deterministic part of the wage"""
    age_comp = myPars.w_determ_cons + myPars.w_age*age + myPars.w_age_2*age*age + myPars.w_age_3*age*age*age 
    #gonna ignore average health which is in Capatina and the iniatialization of theis program for now, will work in more health states when i have composite types working.
    health_comp = myPars.w_good_health*(1-health) + myPars.w_good_health_age*age*(1-health)
    inc = age_comp + health_comp
    #print("Comp is ", comp) 
    return inc

@njit
def lab_inc(myPars: pars, age, health, persistent_shock, fixed_effect) :
    return exp(det_lab_inc(myPars, age, health)) * exp(fixed_effect) * exp(persistent_shock)

@njit
def foc_lab(myPars : pars, c, labor, health, lab_inc ) :
    lhs = lab_inc * util_c_giv_leis(myPars, c, labor, health)
    rhs = myPars.phi_n * util_leis_giv_c(myPars, c, labor, health)
    return lhs - rhs, c

@njit
def infer_a(myPars :pars, a_prime, c, labor, lab_inc) :
    #BC is c + a_prime = labor_inc*labor +(1+myPar.r)*a_curr so
    return ( c + a_prime - lab_inc*labor) / (1+myPars.r)
     

@njit
def solve_lab_a(myPars : pars, a_prime, c, lab_inc, health):
    """
    solve for labor given c and lab_inc using the static condition between c and leis/labor
    then infer the current assets a that balance the budget constraint given 
    the future assets a_prime, c, labor, and labor incometo balance the BC
    """
    lab = lab_giv_c(myPars, c,lab_inc,health)
    a = infer_a(myPars, a_prime, c, lab, lab_inc)
    return lab, a
    




if __name__ == "__main__":
    import pars_and_shocks
    print("Running main")
    start_time = time.time() 

    myPars = pars_and_shocks.pars()
    for age in prange(myPars.J+1):
        for pers_shock in myPars.nu_grid: 
            for health in reversed(prange(2)):
                for fe in myPars.lab_fe_grid:
                    my_inc = lab_inc(myPars, age, health, pers_shock, fe)
                    print('For age ', age, ' persistent shock ', pers_shock, ' health ', health, ' and fixed effect ', fe)
                    print('The labor income is: ', my_inc)
 

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
