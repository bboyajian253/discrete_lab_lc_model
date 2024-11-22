"""
pars_shocks_and_wages.py

Created on 2024-05-18 01:09:42

by @author Ben Boyajian

"""
#file to store parameter and shockvalues for the model

import my_toolbox as tb
import numpy as np
from math import exp,sqrt,log,prod
from numba import njit, guvectorize, float64, int64, prange, types
from numba.core.types import unicode_type, UniTuple
from numba.experimental import jitclass
import time
import copy

#a big list of parameter values for the model
pars_spec = [   ('lab_fe_grid', float64[:]), # a list of values for that fixed effect
                ('lab_fe_grid_size', int64), # the size of the list of values for that fixed effect
                ('lab_fe_weights', float64[:]), # the weights for the fixed effect
                ('lab_fe_tauch_mu', float64), # the mean of the fixed effect distribution
                ('lab_fe_tauch_sigma', float64), # the standard deviation of the fixed effect distribution
                ('beta', float64), # discount factor 
                ('alpha', float64), # cobb douglass returns to consumption
                ('sigma_util', float64), # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs
                ('phi_n', float64), # 40 hours + 5 commuting is the time cost to a discrete work/not work decision
                ('phi_H', float64), # cost to being in bad health, for now pretend there are only two states
                ('r', float64) , # interest rate on assets
                ('a_min', float64), # smallest possible asset value = the borrowing constraint
                ('a_max', float64), # largest possible asset value
                ('a_grid_growth', float64), # spacing/curvature/growth of asset grid parameter
                ('a_grid', float64[:]), # stores the asset grid
                ('a_grid_size', int64), # total number of points on the asset grid
                ('H_type_perm_grid', float64[:]), #grid to hold the premanent health types
                ('H_type_perm_grid_size', int64), #size of the permanent health type grid
                ('H_type_perm_weights', float64[:]), #weights for the permanent health type grid
                ('H_beg_pop_weights_by_H_type', float64[:, :]), #weights for the permanent health type grid
                ('H_grid', float64[:]), # stores the health grid
                ('H_grid_size', int64), # total number of points on the health grid
                ('H_trans_uncond', float64[:, :, :]), #matrix of unconditional health transition probabilities
                ('H_trans', float64[:, :, :, :]), #matrix of health transition probabilities
                ('delta_pi_BB', float64), #diff in prob of bad health persistence between the two types
                ('delta_pi_GG', float64), #diff in prob of good health persistence between the two types
                ('epsilon_gg', float64), # adjustment to the probability of good health persistence
                ('epsilon_bb', float64), # adjustment to the probability of bad health persistence
                ('state_space_shape', UniTuple(int64, 5)), #the shape/dimensions of the full state space with time/age J
                ('state_space_shape_no_j', UniTuple(int64, 4)),
                ('state_space_no_j_size', int64), #size of the state space with out time/age J
                ('state_space_shape_sims', UniTuple(int64, 4)), #the shape/dimensions of the period state space for simulations
                ('lab_min', float64 ), #minimum possible choice for labor, cannot pick less than 0 hours
                ('lab_max', float64), # max possible for labor choice
                ('c_min', float64 ), #minimum possible choice for consumption, cannot pick less than 0
                ('leis_min', float64), #min possible choice for leisure
                ('leis_max', float64), #max possible choice for leisure 
                ('sim_draws', int64),       # number of simulation draws
                ('J', int64),                 # number of time periods -1 (period 0 is first
                ('print_screen', int64),  #indicator for what type of printing to do... may drop
                ('interp_eval_points', float64[:]),
                ('sim_interp_grid_spec', types.Tuple((float64, float64, int64))),
                ('start_age', int64), #age to start the model at
                ('end_age', int64), #age to end the model at
                ('age_grid', int64[:]), #grid of ages
                ('path', unicode_type), #path to save results to
                ('wage_coeff_grid', float64[:,:]), #grid to hold the wage coefficients
                ('wH_coeff', float64), #wage coefficient for health status 
                ('wage_min', float64), #minimum wage
                ('max_iters', int64), #maximum number of iterations for root finder
                ('max_calib_iters', int64), #maximum number of iterations for calibration
        ]

@jitclass(pars_spec)
class Pars() :      
    def __init__(self, path,    
            wage_coeff_grid = np.array([[10.0,0.0,0.0,0.0], [20.0,0.5,-0.01,0.0], [30.0,1.0,-0.02,0.0], [40.0,1.5,-0.03,0.0]]),
            wage_min = 0.0001, #minimum wage

            #a discrete list of productivities to use for testing
            lab_fe_grid = np.array([1.0, 2.0, 3.0]),
            lab_fe_weights = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
            lab_fe_tauch_mu = 0.0,
            lab_fe_tauch_sigma = 1.0,
            # utility parameters
            beta = 0.95, # discount factor
            alpha = 0.70, #.5, # cobb douglass returns to consumption
            sigma_util = 3, # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs

            # time costs and health costs
            phi_n = 1.125, # 40 hours + 5 commuting is the time cost to a discrete work/not work decision was originally 1.125
            phi_H = 0.0, # time cost to being in bad health, for now pretend there are only two states was originally 0.10

            # interest rate and maybe taxes later
            r = 0.02, # interest rate on assets

            a_min = -50, # can think of this as borrowing no more than a_min*1000 dollars
            a_max = 250, # max of the asset grid
            a_grid_growth = 0.0, #detrmines growth rate and thus curvature of asset grid at 0 just doe slinear space
            a_grid_size = 300, #set up for gride with curvature

            H_type_perm_grid = np.array([0.0,1.0]), #grid to hold the premanent health types
            H_type_perm_weights = np.array([0.5,0.5]), #weights for the permanent health type grid
            H_beg_pop_weights_by_H_type = np.array([[0.5, 0.5], [0.5, 0.5]]), #weights for the permanent health type grid
            H_grid = np.array([0.0,1.0]),
            H_trans_uncond = np.repeat(np.array([[0.9, 0.1], [0.7, 0.3]])[np.newaxis, :, :], 51, axis=0).reshape(51,2,2),
            H_trans = np.repeat(np.array([[[0.9, 0.1], [0.7, 0.3]],[[0.4, 0.6], [0.2, 0.8]]])[:, np.newaxis, :,:], 51, axis=0).reshape(2,51,2,2),
            delta_pi_BB = 0.0, #diff in prob of bad health persistence between the two types
            delta_pi_GG = 0.0, #diff in prob of good health persistence between the two types
            epsilon_gg = 0.0, # adjustment to the probability of good health persistence
            epsilon_bb = 0.0, # adjustment to the probability of bad health persistence

            lab_min = 0.00,
            lab_max = 1.0,
            c_min = 0.0001,
            leis_min = 0.0,
            leis_max = 1.0,    

            J = 50,             # number of time periods -1 (period 0 is first)
            start_age = 25,     #age to start the model at
            sim_draws = 1000,   # number of simulation draws
            print_screen = 2,
            max_iters = 100,
            max_calib_iters = 10,
        ):
        
        self.wage_coeff_grid = wage_coeff_grid
        self.wH_coeff = 0.25
        self.wage_min = wage_min  

        # gamma fixed productiviy drawn at birth
        self.lab_fe_grid = self.wage_coeff_grid[:,0]
        self.lab_fe_weights = lab_fe_weights
        self.lab_fe_grid_size = len(self.lab_fe_grid)
        self.lab_fe_tauch_mu = lab_fe_tauch_mu
        self.lab_fe_tauch_sigma = lab_fe_tauch_sigma

        ###iniatlize utlity parameters###
        self.alpha,self.sigma_util = alpha,sigma_util
        self.phi_n,self.phi_H = phi_n,phi_H

        ###interest rate and maybe taxes later
        self.r = r
        self.beta = 1/(1 + r)

        ###intialie grid(s)
        ###define asset grid###
        self.a_min,self.a_max = a_min,a_max
        self.a_grid_size = a_grid_size
        self.a_grid_growth = a_grid_growth
        self.a_grid = tb.gen_grid(a_grid_size, a_min, a_max, a_grid_growth)
        
        self.H_type_perm_grid = H_type_perm_grid
        self.H_type_perm_grid_size = len(H_type_perm_grid)
        self.H_type_perm_weights = H_type_perm_weights
        self.H_beg_pop_weights_by_H_type = H_beg_pop_weights_by_H_type
        self.H_grid  = H_grid
        self.H_grid_size = len(H_grid)

        self.J = J

        self.H_trans_uncond = H_trans_uncond
        self.delta_pi_BB = delta_pi_BB
        self.delta_pi_GG = delta_pi_GG
        self.epsilon_gg = epsilon_gg
        self.epsilon_bb = epsilon_bb

        if delta_pi_BB > 0.0 or delta_pi_GG > 0.0:
            self.H_trans = gen_MH_trans(H_trans_uncond, self.H_type_perm_grid_size, self.J, self.H_grid_size, self.delta_pi_BB, self.delta_pi_GG)
        else: 
            self.H_trans = H_trans

        self.interp_eval_points = np.zeros(1)

        self.c_min = c_min

        self.leis_min,self.leis_max = leis_min, leis_max

        self.lab_min = lab_min
        self.lab_max = leis_max / self.phi_n

        self.start_age = start_age
        self.end_age = start_age + J + 1
        self.age_grid = np.arange(self.start_age, self.end_age, 1)

        self.sim_draws = sim_draws
        self.print_screen = print_screen

        self.state_space_shape = (self.a_grid_size, self.lab_fe_grid_size, self.H_grid_size, self.H_type_perm_grid_size, self.J) 
        self.state_space_shape_no_j = (self.a_grid_size, self.lab_fe_grid_size, self.H_grid_size, self.H_type_perm_grid_size)
        self.state_space_shape_sims = (self.lab_fe_grid_size, self.H_type_perm_grid_size, self.sim_draws, self.J + 1)
        self.state_space_no_j_size = self.a_grid_size * self.lab_fe_grid_size * self.H_grid_size * self.H_type_perm_grid_size
        # self.tuple_sim_ndim = (0,1,2,3)
        # self.len_state_space_shape_sims = len(self.state_space_shape_sims)

        self.sim_interp_grid_spec = (self.a_min, self.a_max, self.a_grid_size)

        self.path = path
        self.max_iters = max_iters
        self.max_calib_iters = max_calib_iters

    def set_w1(self, w1: float):
        # self.wage_coeff_grid[:,1] = w1
        for i in range(self.lab_fe_grid_size):
            self.wage_coeff_grid[i,1] = w1
    
    def set_w2(self, w2: float):
        # self.wage_coeff_grid[:,2] = w2
        for i in range(self.lab_fe_grid_size):
            self.wage_coeff_grid[i,2] = w2

    def update_H_trans(self)-> np.ndarray:
        self.H_trans = gen_MH_trans(self.H_trans_uncond, self.H_type_perm_grid_size, self.J, self.H_grid_size, self.delta_pi_BB, self.delta_pi_GG)
        return self.H_trans

    def set_delta_pi_BB(self, delta_pi_BB: float):
        self.delta_pi_BB = delta_pi_BB
        self.update_H_trans()
    
    def set_delta_pi_GG(self, delta_pi_GG: float):
        self.delta_pi_GG = delta_pi_GG
        self.update_H_trans()
    
    def set_H_trans_uncond(self, H_trans_uncond: np.ndarray):
        self.H_trans_uncond = H_trans_uncond
        self.update_H_trans()


@njit
def gen_MH_trans(MH_trans_uncond:np.ndarray, H_type_perm_grid_size:int, J:int, H_grid_size: int, delta_pi_BB:float, delta_pi_GG:float) -> np.ndarray:
    """
    Calculate full health transition matrix from reshaped matrix with shape (J, H_grid_size, H_grid_size)
    """
    ret_mat = np.zeros((H_type_perm_grid_size, J, H_grid_size, H_grid_size))
    mat_BB = MH_trans_uncond[:, 0, 0]
    mat_GG = MH_trans_uncond[:, 1, 1]

    mat_BB_low_typ = mat_BB * (1 + delta_pi_BB) # *xb_j
    mat_BB_high_typ = mat_BB * (1 - delta_pi_BB) # *xb_j
    mat_GG_low_typ = mat_GG * (1 - delta_pi_GG) # *xg_j
    mat_GG_high_typ = mat_GG * (1 + delta_pi_GG) # *xg_j
    # mat_BB_low_typ = mat_BB + delta_pi_BB # *xb_j
    # mat_BB_high_typ = mat_BB - delta_pi_BB # *xb_j
    # mat_GG_low_typ = mat_GG - delta_pi_GG # *xg_j
    # mat_GG_high_typ = mat_GG + delta_pi_GG # *xg_j

    # low type
    ret_mat[0, :, 0, 0] = mat_BB_low_typ
    ret_mat[0, :, 0, 1] = 1 - mat_BB_low_typ 
    ret_mat[0, :, 1, 0] = 1 - mat_GG_low_typ 
    ret_mat[0, :, 1, 1] = mat_GG_low_typ

    # high type
    ret_mat[1, :, 0, 0] = mat_BB_high_typ
    ret_mat[1, :, 0, 1] = 1 - mat_BB_high_typ
    ret_mat[1, :, 1, 0] = 1 - mat_GG_high_typ
    ret_mat[1, :, 1, 1] = mat_GG_high_typ

    return ret_mat

@njit
def gen_default_wage_coeffs(lab_fe_grid: np.ndarray, num_wage_terms = 4)-> np.ndarray:
    """
    Generate default wage coefficients
    the constant wage term comes from lab_fe_grid the rest are set to dummies
    w_coeff_grid[0, :] = [lab_fe_grid[0], 0.0, 0.0, 0.0] so that the first lab_fe type has no wage growth
    """
    num_lab_fe = lab_fe_grid.shape[0] # ensures numba compatibility to use shape instead of len 
    w_coeff_grid = np.zeros((num_lab_fe, num_wage_terms)) 
    for lab_fe_index in range(num_lab_fe):
        w_coeff_grid[lab_fe_index, :] = [lab_fe_grid[lab_fe_index], 1.0, -0.02, 0.0] 
    return w_coeff_grid

# def copy_pars_instance(myPars: Pars) -> Pars:

shock_spec = [
        ('myPars', Pars.class_type.instance_type),
        ('H_shocks', float64[:, :, :, :]),
        ('H_hist', int64[:, :, :, :]),
        ]

#i feel like drawing shocks using numpy instead of scipy should be jit-able
@jitclass(shock_spec)
class Shocks: 
    def __init__(
        self,
        myPars : Pars
                ):
        self.myPars = myPars
        np.random.seed(1111)
        draws = np.random.uniform(0,1, tb.tuple_product(myPars.state_space_shape_sims))
        reshaped_draws = draws.reshape(myPars.state_space_shape_sims)
        self.H_shocks = reshaped_draws
        self.H_hist = gen_H_hist(myPars, self.H_shocks)

@njit
def gen_H_hist(myPars: Pars, H_shocks: np.ndarray) -> np.ndarray:
    BAD = 0
    GOOD = 1
    hist = np.zeros(myPars.state_space_shape_sims, dtype=np.int64)
    for j in range(myPars.J+1):
        if j > 0:
            hist_j = hist[:, :, :, j-1]
            xgg_j_old = gen_x_gg_j_old(myPars, hist_j)
        for lab_fe_ind in range(myPars.lab_fe_grid_size):
            # for start_H_ind in range(myPars.H_grid_size):
            for H_type_perm_ind in range(myPars.H_type_perm_grid_size):
                for sim_ind in range(myPars.sim_draws):
                    if j > 0:
                        shock = H_shocks[lab_fe_ind, H_type_perm_ind, sim_ind, j-1]       
                        prev_health_state_ind = hist[lab_fe_ind, H_type_perm_ind, sim_ind, j-1]
                        adj = prev_health_state_ind * myPars.epsilon_gg + (1 - prev_health_state_ind) * myPars.epsilon_bb # since prev_health_state_ind is 0 or 1
                        health_recovery_prob = myPars.H_trans[H_type_perm_ind, j-1, prev_health_state_ind, GOOD] + adj
                        # leave this here in case get dpi stuff to work later
                        # if prev_health_state_ind == GOOD:
                        #     health_recovery_prob = myPars.H_trans[H_type_perm_ind, j-1, prev_health_state_ind, GOOD]
                        #     # health_recovery_prob = myPars.H_trans[H_type_perm_ind, j-1, prev_health_state_ind, GOOD] * xgg_j_old 
                        # else:
                        #     health_recovery_prob = myPars.H_trans[H_type_perm_ind, j-1, prev_health_state_ind, GOOD]
                        if shock <= health_recovery_prob:
                                hist[lab_fe_ind, H_type_perm_ind, sim_ind, j] = GOOD
                    else:
                        if sim_ind / myPars.sim_draws < myPars.H_beg_pop_weights_by_H_type[H_type_perm_ind, GOOD]:
                            hist[lab_fe_ind, H_type_perm_ind, sim_ind, j] = GOOD
    return hist

@njit
def gen_x_gg_j_old(myPars: Pars, H_hist_j: np.ndarray)-> float:
    BAD, LOW = 0, 0
    GOOD, HIGH = 1, 1
    omega_g = gen_omega_j_old(myPars, H_hist_j, GOOD)
    delta = myPars.delta_pi_GG

    denom = omega_g*(1+delta) + (1-omega_g)*(1-delta)
    return 1/denom

@njit
def gen_x_bg_j_old(myPars: Pars, H_hist_j: np.ndarray, j: int)-> float:
    BAD, LOW = 0, 0
    GOOD, HIGH = 1, 1
    omega_b = gen_omega_j_old(myPars, H_hist_j, BAD)
    H_trans_uncond = myPars.H_trans_uncond
    H_trans = myPars.H_trans
    pi_bg = H_trans_uncond[j, BAD, GOOD]
    pi_bgH = H_trans[HIGH, j, BAD, GOOD]
    pi_bgL = H_trans[LOW, j, BAD, GOOD]
    denom = omega_b * pi_bgH + (1 - omega_b) * pi_bgL
    return pi_bg / denom
    

@njit
def gen_x_j_old(myPars: Pars, H_hist_j: np.ndarray, state: int)-> float:
    BAD = 0
    GOOD = 1
    omega = gen_omega_j_old(myPars, H_hist_j, state)
    if state == BAD:
        delta = myPars.delta_pi_BB
        denom = omega*(1-delta) + (1-omega)*(1+delta)
        return 1/denom
    elif state == GOOD:
        delta = myPars.delta_pi_GG 
        denom = omega*(1+delta) + (1-omega)*(1-delta)
        return 1/denom

    print("state must be 0 or 1")
    return -999

    
@njit
def gen_omega_j_old(myPars:Pars, H_hist_j:np.ndarray, state:int) -> float:
    BAD = 0
    GOOD = 1
    LOW = 0
    HIGH = 1

    # # Get the dimensions
    # n_lab_fe_weights = myPars.lab_fe_weights.shape[0]
    # n_H_type_perm_weights = myPars.H_type_perm_weights.shape[0]
    # n_sim_draws = myPars.sim_draws
    # weights = np.ones((n_lab_fe_weights, n_H_type_perm_weights, n_sim_draws))
    # # Manually construct weights to avoid broadcasting
    # for i in range(n_lab_fe_weights):
    #     for j in range(n_H_type_perm_weights):
    #         weights[i, j, :] = myPars.lab_fe_weights[i] * myPars.H_type_perm_weights[j]

    if state == GOOD:
        # w_gMH_hist_j = H_hist_j * weights
        w_gMH_hist_j = H_hist_j 
        num_gMH = np.sum(w_gMH_hist_j)
        num_gMH_htype = np.sum(w_gMH_hist_j[:, HIGH, :])
        return num_gMH_htype / num_gMH if num_gMH > 0 else 0.0
    elif state == BAD:
        # w_bMH_hist_j = (1 - H_hist_j) * weights
        w_bMH_hist_j = (1 - H_hist_j) 
        num_bMH = np.sum(w_bMH_hist_j)
        num_bMH_htype = np.sum(w_bMH_hist_j[:, HIGH, :])
        return num_bMH_htype / num_bMH if num_bMH > 0 else 0.0
    
    print("State must be 0 (BAD) or 1 (GOOD)")
    return -999

if __name__ == "__main__":
    print("Running pars_shocks.py")
    start_time = time.time()
    # import os 
    # path = os.path.realpath(__file__ + "/../../")

    path = "../cont_lab_lc_model/"
    input_path = path + "/input/50p_age_moms/"
    trans_path = input_path + "MH_trans_uncond_age.csv"
    
    myPars = Pars(path, J = 51, sim_draws = 1000)
    H_trans = myPars.H_trans
    print(H_trans.shape)
    print(H_trans[:,0,:,:])
    myShocks = Shocks(myPars)
    H_hist = myShocks.H_hist
    print(H_hist.shape)
    print(H_hist[:,:,0,0])

    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")