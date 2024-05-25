"""
Created on 2024-05-19 22:39:56

@author: Ben Boyaian
"""
import my_toy_ls_model as model
import my_toolbox
from pars_and_shocks import Pars
import traceback
import csv
import time
import numpy as np
from numba import njit, prange, float64
from interpolation.splines import eval_linear
from sys import maxsize

@njit
def transform_ap_to_a(myPars : Pars, shell_a, mat_a_ap, mat_c_ap, mat_lab_ap, last_per) :
    #initialie solution shells given current assets a
    mat_ap_a, mat_c_a, mat_lab_a = np.copy(shell_a), np.copy(shell_a), np.copy(shell_a)

    evals = np.copy(myPars.a_grid)
    evals = evals.reshape(myPars.a_grid_size, 1)
    state_size_no_aj = myPars.nu_grid_size * myPars.lab_FE_grid_size * myPars.H_grid_size  
    for state in range(state_size_no_aj) :
        nu_ind, lab_fe_ind, H_ind = my_toolbox.D3toD1(state, myPars.nu_grid_size, 
                                              myPars.lab_FE_grid_size, myPars.H_grid_size)
        #convert soltuions from functions of a_prime to functions of a
        points = (mat_a_ap[:, nu_ind, lab_fe_ind, H_ind],)
        mat_c_a[:, nu_ind, lab_fe_ind, H_ind] = eval_linear(points, mat_c_ap[:, nu_ind, lab_fe_ind, H_ind], 
                                                            evals)
        mat_lab_a[:, nu_ind, lab_fe_ind, H_ind] = eval_linear(points, mat_lab_ap[:, nu_ind, lab_fe_ind, H_ind], 
                                                              evals)
        #invert a(a_prime) to a_prime(a)
        if not last_per: # default value is zero from shell_a, which is correct for last period
            mat_ap_a[:, nu_ind, lab_fe_ind, H_ind] = eval_linear(points, myPars.a_grid, evals)
    #reinsert mat_aprime_a and the corresponding c and lab solutions into a solutions matrix
    sol_a = [mat_c_a, mat_lab_a, mat_ap_a]

    return sol_a

@njit
def solve_j_indiv(myPars : Pars, a_prime, wage, mat_cp_flat_shocks, health, nu) :
    """
    do some voodoo by back substituting and inverting the focs
    """
    # Compute implied c given cc
    # try:
    c_prime = model.expect_util_c_prime(myPars, wage, mat_cp_flat_shocks, health, nu)
    c = model.infer_c(myPars, c_prime, wage)
    # except ZeroDivisionError:
    #     print("ZeroDivisionError occurred at 'infer_c' !!!")
    #     traceback.print_exc()
    #     raise
    # Solve jointly for labor and assets may need to solve for marginal tax rates here...?

    lab, a = model.solve_lab_a(myPars, a_prime, c, wage, health)

    return a, c, lab 
    

#this is for iterating over the state space quickly within a given j
#do i need c_prime to be a matrix here?
# @njit(parallel=True)

@njit(parallel=True)
def solve_per_j_iter(myPars : Pars, shell_a_prime, j, mat_c_prime, last_per ) :
    #initilaize solutions shells
    mat_a_sols, mat_c_sols, mat_lab_sols = np.copy(shell_a_prime), np.copy(shell_a_prime), np.copy(shell_a_prime)
    for state in prange(myPars.a_grid_size * myPars.nu_grid_size * myPars.lab_FE_grid_size * myPars.H_grid_size) :
        
        a_ind, nu_ind, lab_FE_ind, H_ind = my_toolbox.D4toD1(state, myPars.a_grid_size, myPars.nu_grid_size, 
                                                             myPars.lab_FE_grid_size, myPars.H_grid_size)

        #get next period assets a_prime
        a_prime = myPars.a_grid[a_ind]
        #get health
        health = myPars.H_grid[H_ind]
        #get ar1 shock
        nu = myPars.nu_grid[nu_ind]
        #get labor fixed effect
        lab_FE = myPars.lab_FE_grid[lab_FE_ind]
         
        #calculate the wage for this state...
        # it might be fater to do this first and reduce the nunmber of states and just pass labor incomes around
        # i am little bit confused about how i should map shock realizations into just labor incomes... 
        #maybe you just conver them to labor incomes and then pull the oslutions form the matrix
        wage = model.wage(myPars, j, health, nu, lab_FE)

        #If its the last period we know
        if last_per:
            #set this periods assets = to this asset point
            a = a_prime
            lab = model.n_star(myPars, 0, a, health, wage)
            # use the BC to back out consumption
            c = max(myPars.c_min, a * (1 + myPars.r) + lab * wage)
       
        else:  # but if its not the last period
            #this line here is likely to cause trouble I want the 2D matrix that remains after taking into account the current assets a and the lab fixed effect
            # the 2ds are the grids of shocks for that given state though chat gpt says this is valid python i am unable to test it yet
            mat_cp_flat_shocks = mat_c_prime[a_ind, :, lab_FE_ind, :]

            a, c, lab = solve_j_indiv(myPars, a_prime, wage, mat_cp_flat_shocks, health, nu)


        #store the state specific results
        mat_a_sols[a_ind, nu_ind, lab_FE_ind, H_ind] = a
        mat_c_sols[a_ind, nu_ind, lab_FE_ind, H_ind] = c
        mat_lab_sols[a_ind, nu_ind, lab_FE_ind, H_ind] = lab

    return mat_a_sols, mat_c_sols, mat_lab_sols


# if i try to jit this i get some typing issues initializing shell_a_prime
#@njit
def solve_per_j(myPars : Pars, j, mat_c_prime, last_per) :
    #Initialize cell for full period solutions
    # set to large negative values, why?
    
    shell_a_prime =  -maxsize * np.ones(myPars.state_space_shape_no_j) 
    shell_a = np.zeros(myPars.state_space_shape_no_j)

    #compute solution to individuals problems
    mat_a_ap, mat_c_ap, mat_lab_ap = solve_per_j_iter(myPars, shell_a_prime, j, mat_c_prime, last_per)

    #transform these from being indexed by a_prime = ap to being indexed by this periods assets a?
    mat_c, mat_lab, mat_a_prime = transform_ap_to_a(myPars, shell_a, mat_a_ap, mat_c_ap, mat_lab_ap, last_per)
  
    return [mat_c, mat_lab, mat_a_prime ]


def solve_lc(myPars : Pars):
    state_space = myPars.state_space_shape
    #open a log and log how its going
    # Initialize policy and value shells/conatiner arrays

    #use the shape of the statespace to generate the shape of the container for each choice solutions
    sol_list = ['c','n','a_prime'] # i kind of also want to store the continuation value/value functions...
    state_sols = {sol: np.empty(state_space) for sol in sol_list}
    
    mat_c = 9e9 * np.ones(myPars.state_space_shape_no_j) #this is a very big number
    for j in reversed(range(myPars.J)) :
        # # Set age-specific parameters, values
        # #set choice of c for this time/age
        mat_c_prime = mat_c
        # #check if it the last period
        last_per = (j >= myPars.J - 1)
        # #check if they are past retirement age
        # retired = (j >= par.JR)

        #solve the period choice  solutions and return a period solution list
        #store the solution for each state in the column that corresponds to the choice in the state_solutions container
        per_sol_list = solve_per_j(myPars, j, mat_c_prime, last_per)
        
        #update the maximizing choice of the key choices assets/consumptions
        for choice, sol in zip(state_sols.keys(), per_sol_list):
            state_sols[choice][:, :, :, :, j] = sol
        mat_c = per_sol_list[0]

        #print status of lifecycle solutions
        # both to the terminal and store in the status.csv file
        if myPars.print_screen >= 2 :
            print(f'solved period {j} of {myPars.J}')  
    
    return state_sols

if __name__ == "__main__":
    print("Running main")
    start_time = time.time()
    
    myPars = Pars() 
    solve_lc(myPars)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")