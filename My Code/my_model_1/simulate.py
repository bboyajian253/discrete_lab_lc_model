"""
simulate.py

Desc:
Takes X, returs Y
Simulates dtuf etc.

Created on 2024-05-21 21:47:16

@author: Ben Boyaian
"""

#import stuff
from pars_shocks_and_wages import Pars, Shocks
import my_toy_ls_model as model
from interpolation.splines import eval_linear
import numpy as np
from numba import njit, prange
from interpolation import interp

@njit(parallel=True)
def sim_lc_numba(myPars : Pars, sim_vals_list, state_sols_list):
    [sim_a, sim_lab, sim_c]  = sim_vals_list
    [c_lc, lab_lc, a_prime_lc] = state_sols_list

    # simulate-forward life-cycle outcomes 
    for j in prange(myPars.J):
        for lab_fe_ind in prange(myPars.lab_FE_grid_size):
            for h_ind in prange(myPars.H_grid_size):        
                for nu_ind in prange(myPars.nu_grid_size):
                    for sim_ind in prange(myPars.sim_draws):
                        a = sim_a[lab_fe_ind, h_ind, nu_ind, sim_ind, j]
                        #wage = model.
                        #evals = np.array([a])
                        evals = a

                        c = interp(myPars.a_grid, c_lc[:, lab_fe_ind, h_ind, nu_ind, j], evals)
                        lab = interp(myPars.a_grid, lab_lc[:, lab_fe_ind, h_ind, nu_ind, j], evals)
                        a_prime = interp(myPars.a_grid, a_prime_lc[:, lab_fe_ind, h_ind, nu_ind, j], evals)
                        
                        sim_c[lab_fe_ind, h_ind, nu_ind, sim_ind, j] = c
                        sim_lab[lab_fe_ind, h_ind, nu_ind, sim_ind, j] = lab
                        sim_a[lab_fe_ind, h_ind, nu_ind, sim_ind, j + 1] = a_prime


    return [sim_a, sim_lab, sim_c]


def sim_lc(myPars : Pars, myShocks : Shocks, state_sols):
    """
    simulate life-cycle profiles given state solutions (and shock processes if they exist)
    """

    # initialize shells for life-cycle solutions
    #this is the list of results/moments we want to simulate
    vlist = ['a', 'lab ', 'c'] # could add interesting moments:, 'wage', 'leisure', 'health', 'income'
    # **NOTE** DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN sim_lc_jit

    #create dictionary where each v in varlist is a key and the value is a np array of -9999s with shape par2.shapesim
    # par2.shapesim = (par.J, par.Na, par.Nsim)
    sim = {v: -9999 * np.ones(myPars.state_space_shape_sims) for v in vlist}
    # start everyone with zero assets
    sim['a'][ :, :, :, :, 0] = 0.0  

    # simulate life-cycle outcomes
    # get the initial values of the simulations as a list
    sim_vals_list = list(sim.values())
    # get the values of the state solutions as a list
    state_sols_list = list(state_sols.values())

    # call the jit-ted function sim_lc_jit to simulate the life-cycle outcomes
    #given the shell of simulation values and the state solutions as well as the grid_intrp_sim = UCGrid((par.gridk[0], par.gridk[-1], par.Nk)) and parameters
    #  I need one of these in my Pars class: par2.grid_intrp_sim,
    sim_list = sim_lc_numba(myPars, sim_vals_list, state_sols_list)

    # store simulation results in dictionary with matching keys
    # and return it below
    sim = {v: s for v, s in zip(sim.keys(), sim_list)}


    return sim