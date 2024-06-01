"""
solver.py

This file contains the solver module for the project.

Author: Ben Boyajian
Date: 2024-05-31 11:42:26
"""
import model_no_uncert as model
from pars_shocks_and_wages import Pars
import my_toolbox as tb

from numba import njit, prange, float64
import numpy as np
import csv
from math import inf
from typing import Tuple


#solve the whole lifecycle for the given parameters return a dictionary of solutions
def solve_lc( myPars: Pars)-> dict:
    # Start status csv
    fullpath = myPars.path + "/status.csv"
    with open(fullpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'solve_lc started'])

    # Initialize solution shells
    var_list = ['c', 'lab', 'a_prime']
    ### **NOTE:** DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN simulate.sim_lc_jit
    state_sols = {var: np.empty(myPars.state_space_shape) for var in var_list} 
    print("state_sols shape", state_sols['c'].shape)
    
    # Set initial mat_c_prime = mat_c to a large number will be replaced anyways
    mat_c = inf * np.ones(myPars.state_space_shape_no_j) #this is a very big number
    
    # Iterate over periods
    for j in reversed(range(myPars.J)): #could maybe make this inner loop a seperate function and jit and parallelize it with prange
        
        # Set age-specific parameters, values
        mat_c_prime = mat_c
        last_per = (j >= myPars.J - 1)
        # retired = (j >= par.JR)
        
        # Get period solutions
        per_sol_list = solve_per_j( myPars, j, last_per, mat_c_prime)
        
        #Store period solutions
        for var,sol in zip(state_sols.keys(), per_sol_list):
            state_sols[var][:, :, :, :, j] = sol

        # Update mat_c with the solution for consumption from this period
        mat_c = per_sol_list[0] #this means we must always return the consumption first in the solve_per_j function
        
        # Print status of life-cycle solution both to the terminal and store in the status.csv file
        fullpath = myPars.path + '/status.csv'
        with open(fullpath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'solved period {j} of {myPars.J}'])
            writer.writerow([f'consumption {j} of {myPars.J}'])
            tb.write_nd_array(writer, mat_c, 1)
            print(f'solved period {j} of {myPars.J}')
            #print(mat_c)
    
    return state_sols
    
# Solve the individual period problem given the parameters and the period sates
# this may need to be not jitted
# we must always return the consumption first in the solve_per_j function
@njit
def solve_per_j( myPars: Pars, j: int, last_per: bool, mat_c_prime: np.ndarray)-> list:
    """
    solve for c, lab, and a_prime for a given period j
    """
    #Initialie shell for period solutions and asset grid
    shell_shape = myPars.state_space_shape_no_j
    shell_a_prime =  -inf * np.ones(shell_shape)
    shell_a = np.zeros(shell_shape)

    mat_c, mat_lab, mat_a_prime = solve_per_j_iter(myPars, j, shell_a_prime, mat_c_prime, last_per)

    ## Transform variables z(a, kk) to variables z(a, k) using k(a, kk) or something like that?

    #mat_c, mat_lab, mat_a_prime = tb.create_increasing_array(myPars.state_space_shape_no_j), tb.create_increasing_array(myPars.state_space_shape_no_j), tb.create_increasing_array(myPars.state_space_shape_no_j) 
    return [mat_c, mat_lab, mat_a_prime]

# Iterate over individual states
#@njit(parallel=True)
@njit
def solve_per_j_iter(myPars: Pars, j: int, shell_a_prime: np.ndarray, mat_c_prime: np.ndarray, last_per: bool)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over individual states.
    """
    # Initialize solution matrices
    mat_c_ap, mat_lab_ap, mat_a_ap = np.copy(shell_a_prime), np.copy(shell_a_prime), np.copy(shell_a_prime)
    
    # Iterate over states
    for state in range(myPars.state_space_no_j_size): #can be parellized with prange messes up order of any printings in the loop
        
        # Get state indices and values
        a_prime_ind, lab_FE_ind, H_ind, nu_ind = tb.D4toD1(state, myPars.a_grid_size, myPars.lab_FE_grid_size, myPars.H_grid_size, myPars.nu_grid_size)
        a_prime, lab_FE, H, nu = myPars.a_grid[a_prime_ind], myPars.lab_FE_grid[lab_FE_ind], myPars.H_grid[H_ind], myPars.nu_grid[nu_ind]
        ind_tuple = (a_prime_ind, lab_FE_ind, H_ind, nu_ind)

        # Get current wage ***AND FUTURE WAGE IF WAGE VARIES?***
        curr_wage = model.wage(myPars, j, lab_FE, H, nu)
        
        # Get  period solutions
        if last_per: # Consume everything and work as little as possible though i could put model.lab_star here with a_prime = 0
            lab, a = myPars.lab_min, a_prime
            c = a * (1 + myPars.r) + lab * curr_wage
            #if c < myPars.c_min:
            print("State:", state, "Consumption:", c, "Assets:", a, "Curr wage", curr_wage, "Lab:", lab)
            c =  max(myPars.c_min, c)
            print("State:", state, "Consumption:", c, "Assets:", a, "Curr wage", curr_wage, "Lab:", lab)
        else:
            c_prime = mat_c_prime[ind_tuple]
            c, lab, a = solve_j_indiv(myPars, a_prime, curr_wage, j, lab_FE, H, nu, c_prime)
            print("Indices:", ind_tuple, "State:", state) 
            print("C_prime:", c_prime, "C:", c, "Assets:", a, "Curr wage", curr_wage, "Lab:", lab)
        # Store state specific solutions
        mat_c_ap[ind_tuple], mat_lab_ap[ind_tuple], mat_a_ap[ind_tuple] = c, lab, a
        
        # # Print status of individual state solutions
        # print('state:', state, 'a_prime_ind:', a_prime_ind, 'lab_FE_ind:', lab_FE_ind, 'H_ind:', H_ind, 'nu_ind:', nu_ind)
        # print('a_prime:', a_prime, 'lab_FE:', lab_FE, 'H:', H, 'nu:', nu, 'j:', j, 'c_prime:', c_prime)
        # print('c:', c, 'lab:', lab, 'a:', a)
        
    
    return mat_c_ap, mat_lab_ap, mat_a_ap

@njit
def solve_j_indiv( myPars: Pars, a_prime: float, curr_wage: float, j: int, lab_fe: float, H: float, nu: float, c_prime: float)-> Tuple[float, float, float]:
    #c, lab, a = 1,2,3

    # Compute implied c given cc = c_prime
    # dVV_dkk = (1 + r) * model.du_dc(cc, par)
    # c = model.invert_c(dVV_dkk, par)
    c = model.infer_c(myPars, curr_wage, j, lab_fe, H, nu, c_prime)

    lab, a = model.solve_lab_a(myPars, c, a_prime, curr_wage, H)

    return c, lab, a

@njit
def trans_ap_to_a( myPars: Pars) :
    pass

if __name__ == "__main__":
    path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output"
    myPars = Pars(path, J=3, a_grid_size=50, a_min= -15.0, a_max = 15.0, lab_FE_grid=np.array([0.0]), H_grid=np.array([0.0]), nu_grid_size=1)
    solve_lc(myPars)