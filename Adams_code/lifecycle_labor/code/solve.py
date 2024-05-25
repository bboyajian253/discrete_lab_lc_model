# lifecycle_labor: solve
# contains solves model for all possible states

import csv
import model
import numpy as np
from numba import njit, prange
from interpolation.splines import eval_linear  # see, https://github.com/EconForge/interpolation.py
from interpolation import interp
import toolbox


@njit
def transform_kk_to_k(shell_k, mk_kk, mc_kk, mn_kk, last_prd, par):
    """
    Transform variables z(a, kk) to variables z(a, k) using solution k(a, kk).
    """
    # Initialize solutions given (a, k)
    mkk_k, mc_k, mn_k = np.copy(shell_k), np.copy(shell_k), np.copy(shell_k)

    # Want variables as a function of k, where k from gridk
    evals = np.copy(par.gridk)
    evals = evals.reshape(par.Nk, 1)

    for na in range(par.Na):

        # Convert solutions from functions of kk, z(kk), to functions of k, z(k)
        points = (mk_kk[na, :],)
        mc_k[na, :] = eval_linear(points, mc_kk[na, :], evals)
        mn_k[na, :] = eval_linear(points, mn_kk[na, :], evals)

        # Invert k(kk) to kk(k)
        if not last_prd:  # default value is zero from shell_k, which is correct for last period
            mkk_k[na, :] = eval_linear(points, par.gridk, evals)

    # Re-insert mkk to soln_k and return
    soln_k = [mc_k, mn_k, mkk_k]

    return soln_k


@njit
def solve_j_indiv(kk, w, h, r, cc, nlo, nhi, par):
    """
    Solve for consumption c, labor n, income y, capital k
    ...given savings kk, next period consumption cc, current hc h.
    """

    # Compute implied c given cc
    dVV_dkk = (1 + r) * model.du_dc(cc, par)
    c = model.invert_c(dVV_dkk, par)

    # Solve jointly for labor n, income y, capital k, and marginal tax rate
    n, k = model.solve_n_k_yt(kk, c, w * h, r, nlo, nhi, par)

    return k, c, n


@njit
def solve_j_iterate(shell_kk, j, r, w, mcc, maxnj, last_prd, par):
    """
    Iterate over individual states.
    """
    # Initialize solution shells
    mk_kk, mc_kk, mn_kk = np.copy(shell_kk), np.copy(shell_kk), np.copy(shell_kk)

    # Iterate over states
    for s in range(par.Na * par.Nk):

        # Infer states from gridpoints
        # transform the state space index s from one dimension to 2
        na, nkk = toolbox.D2toD1(s, par.Na, par.Nk)
        #get where human capital > minc  given age and ability  
        h = model.human_capital(j, na, par)
        #get next period k_prime 
        kk = par.gridk[nkk]
        nlo = par.minn

        # If last period, answer is trivial
        k, n = kk, nlo
        c = max(par.minc, k * (1 + r))

        # If not last period, solve for (c, n, k)
        if not last_prd:
            k, c, n = solve_j_indiv(kk, w, h, r, mcc[na, nkk], nlo, maxnj, par)

        # Store individual results
        mk_kk[na, nkk] = k
        mc_kk[na, nkk] = c
        mn_kk[na, nkk] = n

    return mk_kk, mc_kk, mn_kk


def solve_j(j, r, w, mcc, retired, last_prd, par, par2):
    """
    Solve for c, n, kk in period j.
    """
    # Initialize shells for full period solution
    # set values of next period k_prime to large negative vals
    shell_kk = -999.999e3 * np.ones(par2.shape_kk)
    # set a bunch of zeros for k
    shell_k = np.zeros(par2.shape_k)

    # Set min/ max for labor supply and investment
    maxnj = par.maxn * (1 - retired)

    # Compute solution to individual problems
    mk_kk, mc_kk, mn_kk = solve_j_iterate(shell_kk, j, r, w, mcc, maxnj, last_prd, par)

    # Transform variables z(a, kk) to variables z(a, k) using k(a, kk)
    mc, mn, mkk = transform_kk_to_k(shell_k, mk_kk, mc_kk, mn_kk, last_prd, par)

    return [mc, mn, mkk]


def solve_lc(r, w, par, par2):

    # Print status of life-cycle solution
    fullpath = par2.path + 'status.csv'
    with open(fullpath, 'w', newline='') as f:
        pen = csv.writer(f)
        pen.writerow([f'solve_lc started'])

    # Initialize policy and value shells/conatiner arrays
    vlist = ['c', 'n', 'kk']
    #use the shape of the statespace to generate the shape of the container for each choice solutions
    state_solns = {v: np.empty(par2.shapestatesolns) for v in vlist}
    ###Note: DO NOT CHANGE ORDER OF vlist W/O CHANGING ORDER IN par.sim_lc_jit

    # Set values for period J+1 which imply zero (no benefit to saving in last period)
    # define container for max c which is very large in the last period
    mc = 9e9 * np.ones(par2.shape_j) #this is a very big number

    # Solve life-cycle via backward induction
    #start with the last period
    for j in reversed(range(par.J)):

        # Set age-specific parameters, values
        #set choice of c for this time/age
        mcc = mc
        #check if it the last period
        last_prd = (j >= par.J - 1)
        #check if they are past retirement age
        retired = (j >= par.JR)

        # Solve period j policies
        # solve for this periods policy functions given 
        #the period j, interest rate r, wage w, if they retired, if its the last period and the parameters
        # returns maxmizing c, n, and k_prime  
        prd_soln_list = solve_j(j, r, w, mcc, retired, last_prd, par, par2)

        # 
        for v, s in zip(state_solns.keys(), prd_soln_list):
            state_solns[v][:, :, j] = s
        # grab the first argument returned by solve_j which is the maximizing consump.
        mc = prd_soln_list[0]

        # Print status of life-cycle solution
        # both to the terminal and store in the status.csv file
        fullpath = par2.path + 'status.csv'
        with open(fullpath, 'a', newline='') as f:
            pen = csv.writer(f)
            pen.writerow([f'solved period {j} of {par.J}'])
            print(f'solved period {j} of {par.J}')

    return state_solns
