"""
solver.py

This file contains the solver module for the project.

Author: Ben Boyajian
Date: 2024-05-31 11:42:26
"""
# Import packages
# General
from numba import njit, prange, float64
import numpy as np
import csv
from math import inf
from typing import Tuple
from interpolation.splines import eval_linear
# My code
import model
from pars_shocks import Pars
import my_toolbox as tb

def solve_lc(myPars: Pars, path: str = None )-> dict:
    """
    solve the life-cycle problem given the parameters
    return a dictionary of solutions
    """
    # Start status csv
    if path is None:
        path = myPars.path + 'output/'
    fullpath = path + "status.csv"
    with open(fullpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'solve_lc started'])

    # Initialize solution shells
    var_list = ['c', 'lab', 'a_prime'] ### **NOTE:** DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN simulate.sim_lc_numba
    state_sols = {var: np.empty(myPars.state_space_shape) for var in var_list} 
    if myPars.print_screen >= 3:
        print("state_sols shape", state_sols['c'].shape)
    
    mat_c = inf * np.ones(myPars.state_space_shape_no_j) #this is a very big number
    mat_VF = -inf * np.ones(myPars.state_space_shape_no_j) #this is a very small number
    for j in reversed(range(myPars.J)): #could maybe make this inner loop a seperate function and jit and parallelize it with prange
        mat_c_prime = mat_c
        # mat_lab_prime = mat_lab
        # mat_VF_prime = mat_VF
        last_per = (j >= myPars.J - 1) # might change with retirement and death
        per_sol_list = solve_per_j(myPars, j, last_per, mat_c_prime)
        # per_sol_list = solve_per_j(myPars, j, last_per, mat_c_prime, mat_lab_prime, mat_VF_prime)
        for var,sol in zip(state_sols.keys(), per_sol_list):
            state_sols[var][:, :, :, :, j] = sol
        mat_c = per_sol_list[0] #this means we must always return the consumption first in the solve_per_j function
        # mat_lab = per_sol_list[1]
        # mat_VF = per_sol_list[3]
        
        if myPars.print_screen >= 3:
            print(f'solved period {j} of {myPars.J}')
        if path is None:
            path = myPars.path + 'output/'
        fullpath = path + '/status.csv'
        with open(fullpath, mode='a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerow([f'solved period {j} of {myPars.J}'])
            
            if myPars.print_screen >= 2:
                for state in range(np.prod(myPars.state_space_shape_no_j)):
                    a_ind, lab_fe_ind, H_ind, H_type_perm_ind = tb.D4toD1(state, myPars.a_grid_size, myPars.lab_fe_grid_size, myPars.H_grid_size, myPars.H_type_perm_grid_size)
                    ind_tuple = (a_ind, lab_fe_ind, H_ind, H_type_perm_ind, j) # also incorporate j in the tuple
                    # Create row elements without using f-strings
                    state_row = ['state:', state, 
                                'a:', myPars.a_grid[a_ind], 
                                'lab_fe:', myPars.lab_fe_grid[lab_fe_ind], 
                                'H:', myPars.H_grid[H_ind], 
                                'H_type_perm:', myPars.H_type_perm_grid[H_type_perm_ind], 
                                'j:', j]
                    writer.writerow(state_row)
                    solution_row = ['c:', state_sols["c"][ind_tuple], 'lab:', state_sols["lab"][ind_tuple], 'a_prime:', state_sols["a_prime"][ind_tuple]]
                    writer.writerow(solution_row)

    return state_sols
    
@njit
def solve_per_j( myPars: Pars, j: int, last_per: bool, mat_c_prime: np.ndarray)-> list:
# def solve_per_j( myPars: Pars, j: int, last_per: bool, mat_c_prime: np.ndarray, mat_lab_prime: np.ndarray, mat_VF_prime: np.ndarray)-> list:
    """
    solve for c, lab, and a_prime for a given period j
    Solve the individual period problem given the parameters and the period sates
    we must always return the consumption first in the solve_per_j function
    """
    #Initialie shell for period solutions and asset grid
    shell_shape = myPars.state_space_shape_no_j
    shell_a_prime =  -inf * np.ones(shell_shape)
    shell_a = np.zeros(shell_shape)

    mat_c_ap, mat_lab_ap, mat_a_prime_ap = solve_per_j_iter(myPars, j, shell_a_prime, mat_c_prime, last_per)
    # mat_c_ap, mat_lab_ap, mat_a_prime_ap, mat_VF_ap = solve_per_j_iter(myPars, j, shell_a_prime, mat_c_prime, mat_lab_prime, mat_VF_prime, last_per)
    mat_c, mat_lab, mat_a_prime = transform_ap_to_a(myPars, shell_a, mat_c_ap, mat_lab_ap, mat_a_prime_ap, last_per)
    # mat_c, mat_lab, mat_a_prime, mat_VF = transform_ap_to_a(myPars, shell_a, mat_c_ap, mat_lab_ap, mat_a_prime_ap, mat_VF_ap, last_per)
    return [mat_c, mat_lab, mat_a_prime]
    # return [mat_c, mat_lab, mat_a_prime, mat_VF]
#@njit(parallel=True)
@njit
def solve_per_j_iter(myPars: Pars, j: int, shell_a_prime: np.ndarray, mat_c_prime: np.ndarray, last_per: bool)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# def solve_per_j_iter(myPars: Pars, j: int, shell_a_prime: np.ndarray, mat_c_prime: np.ndarray, mat_lab_prime: np.ndarray, mat_VF_prime: np.ndarray, last_per: bool
# )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over all individual states within a period j and solve the state specific problem
    """
    mat_c_ap, mat_lab_ap, mat_a_ap = np.copy(shell_a_prime), np.copy(shell_a_prime), np.copy(shell_a_prime)
    # mat_c_ap, mat_lab_ap, mat_a_ap, mat_VF_ap = np.copy(shell_a_prime), np.copy(shell_a_prime), np.copy(shell_a_prime), np.copy(shell_a_prime)
    
    for state in prange(myPars.state_space_no_j_size): #can be parellized with prange messes up order of any printings in the loop
        # Get state indices and values
        a_prime_ind, lab_fe_ind, H_ind, H_type_perm_ind = tb.D4toD1(state, myPars.a_grid_size, myPars.lab_fe_grid_size, myPars.H_grid_size, myPars.H_type_perm_grid_size)
        a_prime, lab_fe, H, H_type_perm= myPars.a_grid[a_prime_ind], myPars.lab_fe_grid[lab_fe_ind], myPars.H_grid[H_ind], myPars.H_type_perm_grid[H_type_perm_ind]
        ind_tuple = (a_prime_ind, lab_fe_ind, H_ind, H_type_perm_ind)
        curr_wage = model.wage(myPars, j, lab_fe_ind, H_ind) 

        # Solve state solutions
        if last_per: 
            lab = model.lab_star(myPars, 0, a_prime, H, curr_wage)
            # lab = 0
            a = a_prime
            c = a * (1 + myPars.r) + lab * curr_wage
            #c = a*(1+myPars.r) 
            c =  max(myPars.c_min, c)
            # VF = model.util(myPars, c, 1)

        else:
            c, lab, a = solve_j_indiv(myPars, a_prime, curr_wage, j, a_prime_ind, lab_fe_ind, H_ind, H_type_perm_ind, mat_c_prime)
            # c, lab, a, VF = solve_j_indiv(myPars, a_prime, j, a_prime_ind, lab_fe_ind, H_ind, H_type_perm_ind, mat_c_prime, mat_lab_prime, mat_VF_prime)

        # Store the state solutions 
        mat_c_ap[ind_tuple], mat_lab_ap[ind_tuple], mat_a_ap[ind_tuple] = c, lab, a 
        # mat_c_ap[ind_tuple], mat_lab_ap[ind_tuple], mat_a_ap[ind_tuple], mat_VF_ap[ind_tuple] = c, lab, a, VF 
        
    return mat_c_ap, mat_lab_ap, mat_a_ap
    # return mat_c_ap, mat_lab_ap, mat_a_ap, mat_VF_ap

@njit
def solve_j_indiv( myPars: Pars, a_prime:float, curr_wage:float, j: int, a_prime_ind:int, lab_fe_ind: int, H_ind: int, 
                  H_type_perm_ind: int, mat_c_prime:np.ndarray)-> Tuple[float, float, float]:
# def solve_j_indiv( myPars: Pars, a_prime:float, j: int, a_prime_ind:int, lab_fe_ind: int, H_ind: int, H_type_perm_ind: int,
#                   mat_c_prime:np.ndarray, mat_lab_prime:np.ndarray, mat_VF_prime:np.ndarray)-> Tuple[float, float, float, float]:
    """
    Solve the individual period problem for a given state
    returns c, lab, a, VF
    """
    BAD, GOOD = 0, 1
    c_prime0 = mat_c_prime[a_prime_ind, lab_fe_ind, BAD, H_type_perm_ind]
    c_prime1 = mat_c_prime[a_prime_ind, lab_fe_ind, GOOD, H_type_perm_ind]
    # lab_prime0 = mat_lab_prime[a_prime_ind, lab_fe_ind, BAD, H_type_perm_ind]
    # lab_prime1 = mat_lab_prime[a_prime_ind, lab_fe_ind, GOOD, H_type_perm_ind]
    # euler_rhs = model.euler_rhs(myPars, c_prime0, c_prime1, lab_prime0, lab_prime1, H_type_perm_ind, j, H_ind)

    # VF_prime0 = mat_VF_prime[a_prime_ind, lab_fe_ind, BAD, H_type_perm_ind]
    # VF_prime1 = mat_VF_prime[a_prime_ind, lab_fe_ind, GOOD, H_type_perm_ind]
    # expect_VF = model.expect_VF(myPars, VF_prime0, VF_prime1, H_type_perm_ind, j, H_ind)

    # c, lab, a, VF = model.make_choices(myPars, euler_rhs, expect_VF, a_prime, curr_wage, H_ind)
    # return c, lab, a, VF

    c = model.infer_c(myPars, curr_wage, j, lab_fe_ind, H_ind, H_type_perm_ind, c_prime0, c_prime1)
    lab, a = model.solve_lab_a(myPars, c, a_prime, curr_wage, H_ind)
    return c, lab, a

@njit
def transform_ap_to_a(myPars : Pars, shell_a, mat_c_ap, mat_lab_ap, mat_a_ap, last_per) -> list:
    """
    Transform the a_prime solutions to a solutions via interpolation returns a list of the a indexed solutions
    """
  
    mat_ap_a, mat_c_a, mat_lab_a = np.copy(shell_a), np.copy(shell_a), np.copy(shell_a)
    # mat_VF_a = np.copy(shell_a)
    evals = np.copy(myPars.a_grid)
    evals = evals.reshape(myPars.a_grid_size, 1)
    state_size_no_aj =  myPars.lab_fe_grid_size * myPars.H_grid_size * myPars.H_type_perm_grid_size 
    
    for state in range(state_size_no_aj) :
        lab_fe_ind, H_ind, H_type_perm_ind = tb.D3toD1(state, myPars.lab_fe_grid_size, myPars.H_grid_size, myPars.H_type_perm_grid_size)
        points = (mat_a_ap[:, lab_fe_ind, H_ind, H_type_perm_ind],)
        mat_c_a[:, lab_fe_ind, H_ind, H_type_perm_ind] = eval_linear(points, mat_c_ap[:, lab_fe_ind, H_ind, H_type_perm_ind], evals)
        mat_lab_a[:, lab_fe_ind, H_ind, H_type_perm_ind] = eval_linear(points, mat_lab_ap[:, lab_fe_ind, H_ind, H_type_perm_ind], evals)
        # mat_lab_a[:, lab_fe_ind, H_ind, H_type_perm_ind] = mat_lab_ap[:, lab_fe_ind, H_ind, H_type_perm_ind] # this is the raw labor from the state solutions should be 0 or 1 (so no interpolation)
        # mat_VF_a[:, lab_fe_ind, H_ind, H_type_perm_ind] = eval_linear(points, mat_VF_ap[:, lab_fe_ind, H_ind, H_type_perm_ind], evals)

        if not last_per: # default value is zero from shell_a, which is correct for last period
            mat_ap_a[:, lab_fe_ind, H_ind, H_type_perm_ind] = eval_linear(points, myPars.a_grid, evals)

    sol_a = [mat_c_a, mat_lab_a, mat_ap_a]
    # sol_a = [mat_c_a, mat_lab_a, mat_ap_a, mat_VF_a]

    return sol_a


if __name__ == "__main__":
    pass