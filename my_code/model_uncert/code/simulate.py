"""
simulate.py

Desc:
Simulates forward the life-cycle profiles of consumption, labor, assets, wage, and labor earnings given state solutions and shock processes

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
    [sim_c, sim_lab, sim_a, sim_wage, sim_lab_earnings]  = sim_vals_list
    [c_lc, lab_lc, a_prime_lc] = state_sols_list

    for j in prange(myPars.J):
        for lab_fe_ind in prange(myPars.lab_fe_grid_size):
            for H_type_perm_ind in prange(myPars.H_type_perm_grid_size):
                for sim_ind in prange(myPars.sim_draws):
                    a = sim_a[lab_fe_ind, H_type_perm_ind, sim_ind, j] # get the a from the previous period
                    evals = a # for clarity, this is the value of a at which we are evaluating the state solutions
                    curr_h_ind = myShocks.H_hist[lab_fe_ind, H_type_perm_ind, sim_ind, j] # get this per health state from history

                    # interp the value of c, labor, and a_prime from the state solutions
                    c = interp(myPars.a_grid, c_lc[:, lab_fe_ind, curr_h_ind, H_type_perm_ind, j], evals)
                    lab = interp(myPars.a_grid, lab_lc[:, lab_fe_ind, curr_h_ind, H_type_perm_ind, j], evals)
                    a_prime = interp(myPars.a_grid, a_prime_lc[:, lab_fe_ind, curr_h_ind, H_type_perm_ind, j], evals)
                    wage = model.wage(myPars, j, lab_fe_ind, curr_h_ind)
                    lab_earnings = wage * lab * 100 # 100 is a scaling factor since labor is between 0 and 1 
                    # will  need  adjustment for taxes, etc. eventually, may need a function in model.py like recover_wage
                    # store the values of c, labor, and a_prime in the simulation arrays
                    sim_c[lab_fe_ind, H_type_perm_ind, sim_ind, j] = c
                    sim_lab[lab_fe_ind, H_type_perm_ind, sim_ind, j] = lab
                    sim_a[lab_fe_ind, H_type_perm_ind, sim_ind, j + 1] = a_prime
                    sim_wage[lab_fe_ind, H_type_perm_ind, sim_ind, j] = wage
                    sim_lab_earnings[lab_fe_ind, H_type_perm_ind, sim_ind, j] = lab_earnings

    return [sim_c, sim_lab, sim_a, sim_wage, sim_lab_earnings]


def sim_lc(myPars : Pars, myShocks : Shocks, state_sols: Dict[str, np.ndarray])-> Dict[str, np.ndarray]:
    """
    simulate life-cycle profiles given state solutions (and shock processes if they exist)
    """
    vlist = ['c', 'lab', 'a', 'wage', 'lab_earnings'] # **NOTE** DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN sim_lc_numba
    
    #dict where each v in vlist is a key that stores an np.ndarray of -9999s with shape myPars.state_space_shape_sims 
    sim = {v: -9999 * np.ones(myPars.state_space_shape_sims) for v in vlist}
    # get the initial values
    sim['a'][ :, :, :, 0] = 0.0 # start everyone with 0 assets 
    sim_vals_list = list(sim.values())
    state_sols_list = list(state_sols.values())
    sim_list = sim_lc_numba(myPars, myShocks, sim_vals_list, state_sols_list)

    # store simulation results in dictionary with matching keys and return
    sim = {v: s for v, s in zip(sim.keys(), sim_list)}
    return sim