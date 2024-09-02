"""
simulate.py

Desc:
Takes X, returns Y
Simulates stuff etc.

Created on 2024-05-21 21:47:16

@author: Ben Boyaian
"""

#import stuff
# General
import numpy as np
from numba import njit, prange
from interpolation.splines import eval_linear 
from interpolation import interp # i use this for interpolation instead of eval_linear
from typing import List, Dict

# My code
from pars_shocks import Pars, Shocks
import model_uncert as model

#@njit(parallel=True) # to paralleliize swap this decorator for the one below
@njit
def sim_lc_numba(myPars : Pars, myShocks: Shocks, sim_vals_list: List[np.ndarray], state_sols_list: List[np.ndarray]) -> List[np.ndarray]:
    [sim_c, sim_lab, sim_a, sim_wage, sim_lab_income]  = sim_vals_list
    [c_lc, lab_lc, a_prime_lc] = state_sols_list

    # simulate-forward life-cycle outcomes
    # loop over all the dimensions of the simulation space 
    for j in prange(myPars.J):
        for lab_fe_ind in prange(myPars.lab_FE_grid_size):
            # for starting_h_ind in prange(myPars.H_grid_size):        
            for H_type_perm_ind in prange(myPars.H_type_perm_grid_size):
                for sim_ind in prange(myPars.sim_draws):

                    # get the value of a from the previous period
                    a = sim_a[lab_fe_ind, H_type_perm_ind, sim_ind, j]
                    evals = a # for clarity, this is the value of a at which we are evaluating the state solutions

                    # get this period's health state from the health history
                    curr_h_ind = myShocks.H_hist[lab_fe_ind, H_type_perm_ind, sim_ind, j]
                    # interp the value of c, labor, and a_prime from the state solutions
                    c = interp(myPars.a_grid, c_lc[:, lab_fe_ind, curr_h_ind, H_type_perm_ind, j], evals)
                    lab = interp(myPars.a_grid, lab_lc[:, lab_fe_ind, curr_h_ind, H_type_perm_ind, j], evals)
                    a_prime = interp(myPars.a_grid, a_prime_lc[:, lab_fe_ind, curr_h_ind, H_type_perm_ind, j], evals)
                    # wage = myPars.wage_grid[lab_fe_ind, h_ind, nu_ind, j] #not doing it this way since pulling from memory is likely slower than computation
                    # wage = model.recover_wage(myPars, c, lab, a_prime, a) # has divide by zero issues
                    wage = model.wage(myPars, j, lab_fe_ind, curr_h_ind)
                    lab_income = wage * lab # will  need  adjustment for taxes, etc. eventually, may need a function in model.py like recover_wage

                    # store the values of c, labor, and a_prime in the simulation arrays
                    sim_c[lab_fe_ind, H_type_perm_ind, sim_ind, j] = c
                    sim_lab[lab_fe_ind, H_type_perm_ind, sim_ind, j] = lab
                    sim_a[lab_fe_ind, H_type_perm_ind, sim_ind, j + 1] = a_prime
                    sim_wage[lab_fe_ind, H_type_perm_ind, sim_ind, j] = wage
                    sim_lab_income[lab_fe_ind, H_type_perm_ind, sim_ind, j] = lab_income

    # infer wage and earnings outcomes given labor, constuption and assets 
    return [sim_c, sim_lab, sim_a, sim_wage, sim_lab_income]


def sim_lc(myPars : Pars, myShocks : Shocks, state_sols: Dict[str, np.ndarray])-> Dict[str, np.ndarray]:
    """
    simulate life-cycle profiles given state solutions (and shock processes if they exist)
    """

    # initialize shells for life-cycle solutions
    vlist = ['c', 'lab', 'a', 'wage', 'lab_income'] # could add interesting moments:, 'wage', 'leisure', 'health', 'income'
    # **NOTE** DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN sim_lc_jit

    #create dictionary where each v in varlist is a key and the value is a np array of -9999s with shape par2.shapesim
    sim = {v: -9999 * np.ones(myPars.state_space_shape_sims) for v in vlist}
    # start everyone with zero assets
    sim['a'][ :, :, :, 0] = 0.0  

    # simulate life-cycle outcomes
    # get the initial values
    sim_vals_list = list(sim.values())
    state_sols_list = list(state_sols.values())
    sim_list = sim_lc_numba(myPars, myShocks, sim_vals_list, state_sols_list)

    # store simulation results in dictionary with matching keys and return
    sim = {v: s for v, s in zip(sim.keys(), sim_list)}
    return sim