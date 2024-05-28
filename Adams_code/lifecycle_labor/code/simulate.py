# lifecycle_labor: simulate
# simulate life-cycle outcomes given state-specific solutions

import model
import numpy as np
from numba import njit
from interpolation.splines import eval_linear  # see, https://github.com/EconForge/interpolation.py

import toolbox


@njit
def sim_lc_jit(sim_list, state_solns, grid_intrp_sim, r, w, par):
    """
    jit-ted portion of sim_lc: simulate life-cycle profiles given state solutions and shock processes
    """

    # unpack sim_list and state_solns variables
    [sime_tru, simeph_tru, simn_tru, simk, simh, simn, simc, simheff, simy, simtax, simtaxe, simtaxc] = sim_list
    [c_lc, n_lc, kk_lc] = state_solns

    # simulate-forward life-cycle outcomes for (k, h, c)
    for j in range(par.J):
        for na in range(par.Na):
            for nsim in range(par.Nsim):
                k = simk[na, nsim, j]
                evals = np.array([k])
                simh[na, nsim, j] = model.human_capital(j, na, par)
                c = eval_linear(grid_intrp_sim, c_lc[na, :, j], evals)
                n = eval_linear(grid_intrp_sim, n_lc[na, :, j], evals)
                kk = eval_linear(grid_intrp_sim, kk_lc[na, :, j], evals)
                simc[na, nsim, j] = c
                simn[na, nsim, j] = n
                simk[na, nsim, j + 1] = kk

    # infer life-cycle outcomes for (e, n, eph, heff, income, taxes, budget)
    simc[:, :, -1], simn[:, :, -1] = 0.0, 0.0
    sime_tru = model.earn(w * simh, simn, par)
    simn_tru = simn
    simeph_tru = sime_tru / np.where(simn_tru > 0, simn_tru, 1e-5)
    simheff = sime_tru / w

    for j in range(par.J):
        for na in range(par.Na):
            for nsim in range(par.Nsim):
                simy[na, nsim, j] = model.income(r * simk[na, nsim, j], sime_tru[na, nsim, j])

    simtaxe, simtaxc = sime_tru - model.aftertaxearn(sime_tru, par), par.τc * simc
    simtax = simtaxe + simtaxc
    simy[:, :, -1] = 0.0

    return [sime_tru, simeph_tru, simn_tru, simk, simh, simn, simc, simheff, simy, simtax, simtaxe, simtaxc]


def sim_lc(state_solns, r, w, par, par2, shock):
    """
    simulate life-cycle profiles given state solutions (and shock processes if they exist)
    """

    # initialize shells for life-cycle solutions
    #this is the list of results/moments we want to simulate
    vlist = ['e_tru', 'eph_tru', 'n_tru', 'k', 'h','n', 'c', 'heff', 'y', 'tax', 'taxe', 'taxc']
    # **NOTE** DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN sim_lc_jit

    #create dictionary where each v in varlist is a key and the value is a np array of -9999s with shape par2.shapesim
    sim = {v: -9999 * np.ones(par2.shapesim) for v in vlist}
    # start everyone with zero assets
    sim['k'][:, :, 0] = 0.0  

    # simulate life-cycle outcomes
    # get the initial values of the simulations as a list
    list1 = list(sim.values())
    # get the values of the state solutions as a list
    list2 = list(state_solns.values())

    # call the jit-ted function sim_lc_jit to simulate the life-cycle outcomes
    #given the shell of simulation values and the state solutions as well as the grid_intrp_sim = UCGrid((par.gridk[0], par.gridk[-1], par.Nk)) and parameters
    sim_list = sim_lc_jit(list1, list2, par2.grid_intrp_sim, r, w, par)

    # store simulation results in dictionary with matching keys
    # and return it below
    sim = {v: s for v, s in zip(sim.keys(), sim_list)}

    # add measurement error
    sim['n'] = np.exp(shock.me_n) * sim['n_tru']
    sim['e'] = np.exp(shock.me_e) * sim['e_tru']
    sim['eph'] = sim['e'] / sim['n']

    return sim
