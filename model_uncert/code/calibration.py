"""
calibration.py

This file contains the calibration code for the model.

Author: Ben
Date: 2024-06-17 15:01:32
"""

# Import packages
# General
import numpy as np
from numpy import log 
import csv
from typing import Tuple, List, Dict
import time
from numba import njit
import pandas as pd

# My code
import my_toolbox as tb
import solver
import model
import pars_shocks as ps
from pars_shocks import Pars, Shocks
import simulate
import plot_lc as plot_lc
import io_manager as io

def calib_w0_mu(myPars: Pars, myShocks: Shocks, main_path: str, tol:float, target:float, w0_mu_min:float, w0_mu_max:float
                  )-> Tuple[np.ndarray, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    calibrates the wage fixed effect weights to match the target mean wage
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    target: the target mean wage
    returns a tuple with the calibrated weights, the mean_wage, the state solutions and the simulations
    """
    # myShocks = Shocks(myPars)
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_mean_calib_params.csv")
    mean_wage = -999.999
    sate_sols = {}
    sim_lc = {}

    # print(f"lab_fe_collapse_weight_wages = {lab_fe_collapse_weight_wages}")
    # print(f"myPars.wH_coeff = {myPars.wH_coeff}")
    
    # # define the lambda function to find the zero of
    get_w0_mean_diff = lambda new_mu: w0_mu_mom_giv_mu(myPars, myShocks, new_mu) - target
    calibrated_mu = tb.bisection_search(get_w0_mean_diff, w0_mu_min, w0_mu_max, tol, myPars.max_iters, myPars.print_screen)
    calibrated_weights = tb.Taucheniid_numba(myPars.lab_fe_tauch_sigma, myPars.lab_fe_grid_size, mean = calibrated_mu, state_grid = myPars.lab_fe_grid)[0] 

    #update the labor fixed effect weights
    myPars.lab_fe_weights = calibrated_weights
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    mean_wage = w0_mu_moment(myPars, myShocks)

    if myPars.print_screen >= 1:
        print(f"w0_mean calibration exited: mean wage = {mean_wage}, target mean wage = {target}")
        print(f"calibrated_mu = {calibrated_mu}")
        print(f"Calibrated weights = {calibrated_weights}")

    return calibrated_weights, mean_wage, state_sols, sim_lc

@njit
def w0_mu_mom_giv_mu(myPars:Pars, myShocks:Shocks, new_mu:float)-> float:
    myPars.lab_fe_tauch_mu = new_mu 
    new_weights, tauch_state_grid = tb.Taucheniid_numba(myPars.lab_fe_tauch_sigma, myPars.lab_fe_grid_size, mean = new_mu, state_grid = myPars.lab_fe_grid)
    myPars.lab_fe_weights = new_weights
    return w0_mu_moment(myPars, myShocks)

@njit
def w0_mu_moment(myPars: Pars, myShocks: Shocks)-> float:
    " get the weighted mean of wages for period 0"
    mean_first_per_wage = w0_moments(myPars, myShocks)[0]
    return mean_first_per_wage

def calib_w0_sigma(myPars: Pars, myShocks: Shocks, main_path: str, tol:float, target:float, w0_sigma_min:float, w0_sigma_max:float
                )-> Tuple[np.ndarray, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    calibrates the wage fixed effect weights to match the target standard deviation of wages
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    sd_target: the target standard deviation of wages
    returns a tuple with the calibrated weights, the standard deviation of wages, the state solutions and the simulations
    """
    # io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_sd_calib_params.csv")
    sd_wage = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w0_sd_diff = lambda new_sigma: w0_sigma_mom_giv_sigma(myPars, myShocks, new_sigma) - target
    calibrated_sigma = tb.bisection_search(get_w0_sd_diff, w0_sigma_min, w0_sigma_max, tol, myPars.max_iters, myPars.print_screen)
    calibrated_weights = tb.Taucheniid_numba(calibrated_sigma, myPars.lab_fe_grid_size, mean = myPars.lab_fe_tauch_mu, state_grid = myPars.lab_fe_grid)[0]

    #update the labor fixed effect weights
    myPars.lab_fe_weights = calibrated_weights
    # shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    sd_wage = w0_sigma_moment(myPars, myShocks)

    if myPars.print_screen >= 1:
        print(f"w0_sd calibration exited: sd wage = {sd_wage}, target sd wage = {target}")
        print(f"calibrated_sigma = {myPars.lab_fe_tauch_sigma}")
        print(f"Calibrated weights = {myPars.lab_fe_weights}")

    return calibrated_weights, sd_wage, state_sols, sim_lc

@njit
def w0_sigma_mom_giv_sigma(myPars: Pars, myShocks: Shocks, new_sigma:float)-> float:
    myPars.lab_fe_tauch_sigma = new_sigma
    new_weights, tauch_state_grid = tb.Taucheniid_numba(new_sigma, myPars.lab_fe_grid_size, mean = myPars.lab_fe_tauch_mu, state_grid = myPars.lab_fe_grid)
    myPars.lab_fe_weights = new_weights
    return w0_sigma_moment(myPars, myShocks)

@njit
def w0_sigma_moment(myPars: Pars, myShocks: Shocks)-> float:
    sd_first_per_wage = w0_moments(myPars, myShocks)[1]
    return sd_first_per_wage

@njit
def w0_moments(myPars: Pars, myShocks: Shocks)-> Tuple[float, float]:
    " get the weighted mean and standard deviation of wages for period 0"
    # myShocks = Shocks(myPars)
    first_per_wages = np.log(model.gen_wage_hist(myPars, myShocks)[:,:,:,0])
    # first_per_weighted_wages = model.gen_weighted_wage_hist(myPars, myShocks)[:,:,:,0] 
    first_per_weighted_wages = model.gen_wlog_wage_hist(myPars, myShocks)[:,:,:,0]
    mean_first_per_wage = np.sum(first_per_weighted_wages)

    # Calculate the weighted variance
    deviations =  first_per_wages - mean_first_per_wage
    squared_deviations = deviations ** 2
    # Apply the weights
    weighted_squared_deviations = squared_deviations * (1.0 / myPars.sim_draws)
    weighted_squared_deviations = weighted_squared_deviations * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis]
    weighted_squared_variance = np.sum(weighted_squared_deviations * myPars.lab_fe_weights[:, np.newaxis, np.newaxis]) 
    # Calculate the weighted standard deviation
    sd_first_per_wage = np.sqrt(weighted_squared_variance)
    
    return mean_first_per_wage, sd_first_per_wage

def get_w0_mean_targ(myPars: Pars, target_folder_path: str)-> float:
    """ get the target mean wage for period 0 from myPars.path + '/input/wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_moments_path = target_folder_path + '/wage_moments.csv'
    # data_mom_col_ind = 1 # wage
    data_mom_col_ind = 3 # log wage
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return mean_wage_by_age[0]

def get_w0_sd_targ(myPars: Pars, target_folder_path: str)-> float:
    """ get the target standard deviation of wages for period 0 from myPars.path + '/input/wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_moments_path = target_folder_path + '/wage_moments.csv'
    # data_mom_col_ind = 2 # wage
    data_mom_col_ind = 4 # log wage
    sd_wage_col= tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return sd_wage_col[0]

def calib_w1(myPars: Pars, myShocks: Shocks, main_path: str, tol:float, target:float, w1_min:float, w1_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    calibrates the wage growth coefficient to match the target wage growth
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    tol: the tolerance for the calibration
    target: the target wage growth
    w1_min: the minimum value of the wage growth coefficient
    w1_max: the maximum value of the wage growth coefficient
    returns a tuple with the calibrated wage growth coefficient, the wage growth, the state solutions and the simulated labor choices
    """
    # io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w1_calib_params.csv")
    w1_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w1_diff = lambda new_coeff: w1_moment_giv_w1(myPars, myShocks, new_coeff) - target
    calibrated_w1 = tb.bisection_search(get_w1_diff, w1_min, w1_max, tol, myPars.max_iters, myPars.print_screen)
    # update the wage coeff grid
    for i in range(myPars.lab_fe_grid_size):
        myPars.wage_coeff_grid[i, 1] = calibrated_w1

    # solve, simulate model for the calibrated w1
    w1_moment = w1_moment_giv_w1(myPars, myShocks, calibrated_w1) 
    # io.print_params_to_csv(myPars, path = main_path, file_name = "w1_calib_params.csv")
    # shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    if myPars.print_screen > 0:
        print(f"Calibration exited: w1 = {calibrated_w1}, wage growth = {w1_moment}, target wage growth = {target}") 

    return calibrated_w1, w1_moment, state_sols, sim_lc

@njit
def w1_moment_giv_w1(myPars: Pars, myShocks:Shocks, new_coeff:float)-> float:
    """ updates the wage_coeff_grid and returns the new wage growth """
    # for i in range(myPars.lab_fe_grid_size): 
    #     myPars.wage_coeff_grid[i, 1] = new_coeff
    myPars.set_w1(new_coeff)
    return w1_moment(myPars, myShocks)

@njit
def w1_moment(myPars: Pars, myShocks:Shocks)-> float:
    """ calculates the wage growth given the model parameters """
    # myShocks = Shocks(myPars)
    # wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    wage_sims = model.gen_wlog_wage_hist(myPars, myShocks)
    mean_wage_by_age = tb.sum_last_axis_numba(wage_sims)
    wage_diff = np.max(mean_wage_by_age) - mean_wage_by_age[0]
    return wage_diff

def get_w1_targ(myPars: Pars, target_folder_path: str)-> float:
    """ gets the target wage growth until age '60' from myPars.path + '/input/wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_moments_path = target_folder_path + '/wage_moments.csv'
    data_mom_col_ind = 3 # log wage
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    # want to get wages before age 60
    age_60_ind = 60 - myPars.start_age
    my_max = np.max(mean_wage_by_age[:age_60_ind])
    return my_max - mean_wage_by_age[0]

def calib_w2(myPars: Pars, myShocks: Shocks, main_path: str, tol:float, target:float, w2_min:float, w2_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    calibrates the quadratic wage coefficient to match the target wage decay
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    tol: the tolerance for the calibration
    target: the target wage decay
    w2_min: the minimum value of the wage decay coefficient
    w2_max: the maximum value of the wage decay coefficient
    returns a tuple with the calibrated wage decay coefficient, the wage decay, the state solutions and the simulations
    """
    # io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w2_calib_params.csv")
    w2_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w2_diff = lambda new_coeff: w2_moment_giv_w2(myPars, myShocks, new_coeff) - target
    calibrated_w2 = tb.bisection_search(get_w2_diff, w2_min, w2_max, tol, myPars.max_iters, myPars.print_screen)
    # update the wage coeff grid
    for i in range(myPars.lab_fe_grid_size):
        myPars.wage_coeff_grid[i, 2] = calibrated_w2
    
    # solve, simulate model for the calibrated w2
    w2_moment = w2_moment_giv_w2(myPars, myShocks, calibrated_w2)
    # io.print_params_to_csv(myPars, path = main_path, file_name = "w2_calib_params.csv")
    # shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: w2 = {calibrated_w2}, wage growth = {w2_moment}, target wage growth = {target}")

    return calibrated_w2, w2_moment, state_sols, sim_lc

@njit
def w2_moment_giv_w2(myPars: Pars, myShocks:Shocks, new_coeff:float)-> float:
    """ updates the wage_coeff_grid skipping the first row coefficient and returns the new wage decay """
    myPars.set_w2(new_coeff)
    return w2_moment(myPars, myShocks)

@njit
def w2_moment(myPars: Pars, myShocks: Shocks)-> float:
    """ calculates the wage decay given the model parameters """
    # myShocks = Shocks(myPars)
    # wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    wage_sims = model.gen_wlog_wage_hist(myPars, myShocks)
    mean_wage_by_age = tb.sum_last_axis_numba(wage_sims)
    wage_diff = np.max(mean_wage_by_age) - mean_wage_by_age[myPars.J-1] # or is it J-1
    # wage_diff = log(np.max(mean_wage_by_age)) - log(mean_wage_by_age[myPars.J-1])
    return wage_diff

def get_w2_targ(myPars: Pars, target_folder_path: str)-> float:
    """ gets the target wage decay starting at age 60 'wage_moments.csv' """
    data_moments_path =   target_folder_path + '/wage_moments.csv'
    # data_mom_col_ind = 1
    data_mom_col_ind = 3 # log wage
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    age_60_ind = 60 - myPars.start_age
    my_max = np.max(mean_wage_by_age[:age_60_ind])
    return my_max - mean_wage_by_age[-1]

def calib_wH(myPars: Pars, myShocks:Shocks, main_path: str, tol:float, target:float, wH_min:float, wH_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """ 
    calibrate the wage coeffecient on the health state to match the target wage premium
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    tol: the tolerance for the calibration
    target: the target wage premium
    wH_min: the minimum value of the wage premium coefficient
    wH_max: the maximum value of the wage premium coefficient
    returns a tuple with the calibrated wage premium coefficient, the wage premium, the state solutions and the simulations
    """
    # io.print_params_to_csv(myPars, path = main_path, file_name = "pre_wH_calib_params.csv")
    wH_moment = -999.999
    sate_sols = {}
    sim_lc = {}

    # define the lambda function to find the zero of
    get_wH_diff = lambda new_coeff: wH_moment_giv_wH(myPars, myShocks, new_coeff) - target
    calibrated_wH = tb.bisection_search(get_wH_diff, wH_min, wH_max, tol, myPars.max_iters, myPars.print_screen)
    myPars.wH_coeff = calibrated_wH

    # solve, simulate model for the calibrated wH
    wH_moment = wH_moment_giv_wH(myPars, myShocks, calibrated_wH)
    # io.print_params_to_csv(myPars, path = main_path, file_name = "wH_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: wH = {calibrated_wH}, wage premium = {wH_moment}, target wage premium = {target}")

    return calibrated_wH, wH_moment, state_sols, sim_lc

@njit
def wH_moment_giv_wH(myPars: Pars, myShocks: Shocks, new_coeff:float)-> float:
    myPars.wH_coeff = new_coeff
    return wH_moment(myPars, myShocks)

# we probably have a way to jit this at this point
@njit
def wH_moment(myPars: Pars, myShocks: Shocks)-> float:
    """returns the wage premium given the model parameters"""
    # myShocks = Shocks(myPars)
    wage_sims = np.log(model.gen_wage_hist(myPars, myShocks)[:,:,:,:myPars.J])
    H_hist = myShocks.H_hist[:,:,:,:myPars.J]
    healthy_wage_sims = wage_sims * (H_hist==1)
    unhealthy_wage_sims = wage_sims * (H_hist==0)

    if np.count_nonzero(healthy_wage_sims) == 0 or np.count_nonzero(unhealthy_wage_sims) == 0:
        return -999.999

    healthy_mean_wage = model.wmean_non_zero(myPars, healthy_wage_sims)
    unhealthy_mean_wage = model.wmean_non_zero(myPars, unhealthy_wage_sims)
    wage_prem = healthy_mean_wage - unhealthy_mean_wage

    return wage_prem

def get_wH_targ(myPars: Pars, target_folder_path: str)-> float:
    """ get the target wage premium from myPars.path + '/input/MH_wage_moments.csv' """
    data_moments_path = target_folder_path + '/MH_wage_moments.csv'
    # data_mom_col_ind = 0 # mean diff in log wage
    data_mom_col_ind = 1 # coeff on log wage
    mean_wage_diff = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return mean_wage_diff[0]

def calib_alpha(myPars: Pars, myShocks:Shocks, main_path: str, lab_tol:float, mean_lab_targ:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray], Shocks]:
    """
    calibrates the alpha parameter of the model to match the target mean labor worked.
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    lab_tol: the tolerance for the calibration
    mean_lab_targ: the target mean labor worked
    returns a tuple with the calibrated alpha, the mean labor worked, the state solutions and the simulated labor choices
    """
    # io.print_params_to_csv(myPars, path = main_path, file_name = "pre_alpha_calib_params.csv")
    mean_lab = -999.999
    state_sols = {}
    sim_lc = {}
    alpha_min = 0.00001 #alpha of 0 is not economically meaningful
    alpha_max = 1.0
    
    # define the lambda function to find the zero of     
    get_mean_lab_diff = lambda new_alpha: alpha_moment_giv_alpha(myPars, myShocks, new_alpha, main_path)[0] - mean_lab_targ 
    calib_alpha = tb.bisection_search(get_mean_lab_diff, alpha_min, alpha_max, lab_tol, myPars.max_iters, myPars.print_screen) 
    myPars.alpha = calib_alpha # myPars is mutable this also happens inside solve_mean_lab_giv_alpha but i think its more readable here
    
    # solve, simulate and plot model for the calibrated alpha
    mean_lab, state_sols, sim_lc, shocks = alpha_moment_giv_alpha(myPars, myShocks, calib_alpha, main_path)
    # io.print_params_to_csv(myPars, path = main_path, file_name = "alpha_calib_params.csv")
    if myPars.print_screen >= 1:
        print(f"Calibration exited: alpha = {calib_alpha}, mean labor worked = {mean_lab}, target mean labor worked = {mean_lab_targ}")
    
    return calib_alpha, mean_lab, state_sols, sim_lc, myShocks
    

def alpha_moment_giv_alpha(myPars : Pars, myShocks: Shocks, new_alpha:float, main_path : str = None
                           ) ->Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray], Shocks]:
    '''
        solves the model for a given alpha and returns the alpha moment: the mean labor worked, the model solutions and simulations
    ''' 
    myPars.alpha = new_alpha
    # shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    lab_sim_lc = sim_lc['lab']
    mean_lab = alpha_moment_giv_lab_sim(myPars, lab_sim_lc) 
    return mean_lab, state_sols, sim_lc, myShocks


def alpha_moment_giv_lab_sim(myPars: Pars, lab_sim_lc: np.ndarray)-> float:
    """
    calculates the mean labor worked given the simulations
    takes the following arguments:
    myPars: the parameters of the model
    sims: the simulations of the model
    returns the mean labor worked
    """
    labor_sims = lab_sim_lc[:, :, :, :myPars.J]
    weighted_labor_sims = model.gen_weighted_sim(myPars, labor_sims) 
    mean_lab_by_age = tb.sum_last_axis_numba(weighted_labor_sims)
    mean_lab = np.mean(mean_lab_by_age)
    return mean_lab

def alpha_moment_giv_sims(myPars: Pars, sim_lc: Dict[str, np.ndarray])-> float:
    return alpha_moment_giv_lab_sim(myPars, sim_lc['lab'])

def get_alpha_targ(myPars: Pars, target_folder_path: str) -> float:
    """
    reads alpha target moment from myPars.path + '/input/labor_moments.csv'
    """
    data_moments_path = target_folder_path + '/alpha_mom_targ.csv'
    # data_mom_col_ind = 1
    data_mom_col_ind = 0
    mean_labor_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return np.mean(mean_labor_by_age)


def calib_phi_H(myPars: Pars, main_path: str, tol:float, target:float, phi_H_min:float, phi_H_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    '''
    calibrates the phi_H parameter of the model to match the target difference in mean log hours worked between the bad MH and good MH states
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    tol: the tolerance for the calibration
    target: the target difference in mean log hours worked between the bad MH and good MH states
    phi_H_min: the minimum value of the phi_H parameter
    phi_H_max: the maximum value of the phi_H parameter
    returns a tuple with the calibrated phi_H, the difference in mean log hours worked between the bad MH and good MH states, the state solutions and the simulations
    '''
    # io.print_params_to_csv(myPars, path = main_path, file_name = "pre_phi_H_calib_params.csv")
    phi_H_moment = -999.999
    state_sols = {}
    sim_lc = {}

    # define the lambda function to find the zero of
    get_phi_H_diff = lambda new_phi_H: phi_H_moment_giv_phi_H(myPars, new_phi_H)[0] - target
    # print("Entering bisection search")
    calibrated_phi_H = tb.bisection_search(get_phi_H_diff, phi_H_min, phi_H_max, tol, myPars.max_iters, myPars.print_screen)
    myPars.phi_H = calibrated_phi_H

    # solve, simulate model for the calibrated phi_H
    phi_H_moment, state_sols, sim_lc = phi_H_moment_giv_phi_H(myPars, calibrated_phi_H)
    # io.print_params_to_csv(myPars, path = main_path, file_name = "phi_H_calib_params.csv")
    # if myPars.print_screen >= 1:
    #     print(f"Calibration exited: phi_H = {calibrated_phi_H}, difference in mean log hours worked = {phi_H_moment}, target difference in mean log hours worked = {target}")

    return calibrated_phi_H, phi_H_moment, state_sols, sim_lc

def phi_H_moment_giv_phi_H(myPars: Pars, new_phi_H:float)-> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    '''
    solves the model for a given phi_H and returns the phi_H moment, the mean phi_H, the state solutions and the simulations
    '''
    myPars.phi_H = new_phi_H
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    lab_sim_lc = sim_lc['lab']
    return phi_H_moment(myPars, lab_sim_lc), state_sols, sim_lc

# def phi_H_moment(myPars: Pars, sims: Dict[str, np.ndarray], shocks: Shocks)-> float:

@njit
def phi_H_moment(myPars: Pars, lab_sims: np.ndarray)-> float:
    '''
    returns the difference in mean log hours worked between the bad MH and good MH states 
    '''
    # pre_retirement_age = 55
    # pre_retirement_age_ind = np.where(myPars.age_grid == pre_retirement_age)[0][0]
    # lab_sims = lab_sims[:, :, :, :pre_retirement_age_ind]
    # H_hist: np.ndarray = shocks.H_hist[:, :, :, :pre_retirement_age_ind]
    shocks = Shocks(myPars)
    lab_sims = lab_sims[:, :, :, :myPars.J]
    H_hist: np.ndarray = shocks.H_hist[:, :, :, :myPars.J]

    # # take the log where hours are positive
    # for i in range(lab_sims.size):
    #     if lab_sims.flat[i] <= 0:
    #         lab_sims.flat[i] = 1e-10
    # log_lab_sims = np.log(lab_sims)

    # bad_MH_log_lab_sims = log_lab_sims * (H_hist == 0)
    # good_MH_log_lab_sims = log_lab_sims * (H_hist == 1)
    bad_MH_sims = lab_sims * (H_hist == 0)
    good_MH_sims = lab_sims * (H_hist == 1)

    # bad_MH_log_lab_mean = model.wmean_non_zero(myPars, bad_MH_log_lab_sims)
    # good_MH_log_lab_mean = model.wmean_non_zero(myPars, good_MH_log_lab_sims)
    # diff = good_MH_log_lab_mean - bad_MH_log_lab_mean   
    bad_MH_lab_mean = model.wmean_non_zero(myPars, bad_MH_sims)
    good_MH_lab_mean = model.wmean_non_zero(myPars, good_MH_sims)
    diff = good_MH_lab_mean - bad_MH_lab_mean

    return diff
    
def get_phi_H_targ(myPars: Pars, target_folder_path: str)-> float:
    '''
    reads phi_H target moment from target_folderZJpath+ '/MH_hours_moments.csv'
    '''
    # data_moments_path = myPars.path + '/input/MH_hours_moments.csv'
    # data_moments_path = target_folder_path + '/MH_hours_moments.csv'
    data_moments_path = target_folder_path + '/MH_hours_moments.csv'
    data_mom_col_ind = 0
    # mean_log_lab_diff = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    mean_lab_diff = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    # return mean_log_lab_diff[0]
    return mean_lab_diff[0]/100


def calib_epsilon_gg(myPars: Pars, myShocks:Shocks, main_path:str, tol:float, target:float, eps_gg_min:float, eps_gg_max:float)-> Tuple[float, float, Dict[str, np.array], Dict[str, np.array]]:
    eps_gg_moment = -999
    state_sols = {}
    sim_lc = {}

    # define the objective function
    get_eps_diff = lambda new_eps: eps_gg_mom_giv_eps(myPars,  new_eps) - target
    calibrated_eps = tb.bisection_search(get_eps_diff, eps_gg_min, eps_gg_max, tol, myPars.max_iters, myPars.print_screen)
    # myPars.epsilon_gg = calibrated_eps
    myPars.set_eps_gg(calibrated_eps)

    # solve, simulate model for the calibrated epsilon_gg
    eps_moment = eps_gg_mom_giv_eps(myPars, calibrated_eps)
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    if myPars.print_screen > 0:
        print(f"calibrated epsilon_gg = {calibrated_eps}, epsilon_gg moment = {eps_moment}, target = {target}")

    return calibrated_eps, eps_moment, state_sols, sim_lc

@njit
def eps_gg_mom_giv_eps(myPars: Pars, new_eps: float)->float:
    # myPars.epsilon_gg = new_eps
    myPars.set_eps_gg(new_eps)
    myShocks = Shocks(myPars)
    return eps_gg_moment(myPars, myShocks)
@njit
def eps_gg_moment(myPars:Pars, myShocks:Shocks)->float:
    # myShocks = Shocks(myPars)
    H_hist = myShocks.H_hist
    weighted_H_hist = model.gen_weighted_sim(myPars, H_hist) 
    # the line below may prove difficult to jit
    # good_MH_age_sim = np.sum(weighted_H_hist, tuple(range(H_hist.ndim - 1)))
    good_MH_age_sim = tb.sum_last_axis_numba(weighted_H_hist)
    bad_MH_age_sim = 1 - good_MH_age_sim
    return np.mean(bad_MH_age_sim)

def get_eps_gg_targ(myPars: Pars, target_folder_path)->float:
    file_path = target_folder_path + "mean_bad_MH_by_age.csv"
    bad_MH_age_data = pd.read_csv(file_path)
    # rename the columns
    bad_MH_age_data.columns = ['age', 'mean_badMH']
    bad_MH_age_data = bad_MH_age_data['mean_badMH'].to_numpy()
    return np.mean(bad_MH_age_data)

def calib_epsilon_bb(myPars: Pars, main_path:str, tol:float, target:float, eps_bb_min:float, eps_bb_max:float)-> Tuple[float, float, Dict[str, np.array], Dict[str, np.array]]:
    eps_bb_mom = -999.999
    state_sols = {}
    sim_lc = {}

    # define the objective function
    get_eps_diff = lambda new_eps: eps_bb_mom_giv_eps(myPars, new_eps) - target
    calibrated_eps = tb.bisection_search(get_eps_diff, eps_bb_min, eps_bb_max, tol, myPars.max_iters, myPars.print_screen)
    # myPars.epsilon_bb = calibrated_eps
    myPars.set_eps_bb(calibrated_eps)

    # solve the model
    eps_moment = eps_bb_mom_giv_eps(myPars, calibrated_eps)
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    if myPars.print_screen > 0:
        print(f"calibrated_eps = {calibrated_eps}, eps_moment = {eps_moment}, target = {target}")

    return calibrated_eps, eps_moment, state_sols, sim_lc

def get_eps_bb_targ(myPars: Pars, target_folder_path: str)-> float:
    '''
    reads delta_pi_GG target moment from target_folder_path + '/autocorr_matrix.csv'
    '''
    data_moms_path = target_folder_path + '/autocorr_matrix.csv'
    autocorr_mat_pd = pd.read_csv(data_moms_path)
    MH_autocorr = autocorr_mat_pd['MH']
    return MH_autocorr.iloc[1]

@njit
def eps_bb_mom_giv_eps(myPars: Pars, new_eps: float)->float:
    # myPars.epsilon_bb = new_eps
    myPars.set_eps_bb(new_eps)
    myShocks = Shocks(myPars)
    return eps_bb_moment(myPars, myShocks)

@njit
def eps_bb_moment(myPars: Pars, myShocks:Shocks)-> float:
    '''
    returns the autocorrelation of health with its nth lag
    '''
    n = 1
    H_hist = myShocks.H_hist[:, :, :, :myPars.J]
    H_hist_ac = tb.lagged_corr_jit(H_hist, max_lag=10)
    return H_hist_ac[n]

@njit
def calib_all_eps_numba(myPars:Pars, main_path:str,
                eps_bb_tol:float, eps_gg_tol:float, eps_bb_target:float, eps_gg_target:float,
                eps_bb_min:float = -0.3, eps_bb_max:float = 0.3, eps_gg_min:float = -0.3, eps_gg_max:float = 0.3,
                eps_bb_max_iters:int = 30, eps_gg_max_iters:int = 30, 
                )-> Tuple[float, float, float, float, Dict[str, np.array], Dict[str, np.array]]:

    eps_bb_err = lambda new_eps_bb: eps_bb_mom_giv_eps(myPars, new_eps_bb) - eps_bb_target
    eps_gg_err = lambda new_eps_gg: eps_gg_mom_giv_eps(myPars, new_eps_gg) - eps_gg_target

    # calibrate epsilon_bb
    bb_min = eps_bb_min
    bb_max = eps_bb_max
    bb_iters = 0
    bb_err = 1 + eps_bb_tol
    gg_err = 1 + eps_gg_tol
    while (abs(bb_err) > eps_bb_tol or abs(gg_err) > eps_gg_tol) and bb_iters < eps_bb_max_iters:
        # calibrate epsilon_gg 
        gg_min = eps_gg_min
        gg_max = eps_gg_max
        gg_iters = 0
        gg_err = 1 + eps_gg_tol
        while abs(gg_err) > eps_gg_tol and gg_iters < eps_gg_max_iters:
            gg_mid_pt = (gg_min + gg_max)/2
            gg_err = eps_gg_err(gg_mid_pt)
            if gg_err < 0:
                gg_max = gg_mid_pt
            else:
                gg_min = gg_mid_pt
            gg_iters += 1
            # gg_err = eps_gg_err(myPars.epsilon_gg)
        # calibrate epsilon_bb
        bb_mid_pt = (bb_min + bb_max)/2
        bb_err = eps_bb_err(bb_mid_pt)
        if bb_err > 0:
            bb_max = bb_mid_pt
        else:
            bb_min = bb_mid_pt
        bb_iters += 1
        # bb_err = eps_bb_err(myPars.epsilon_bb)
        gg_err = eps_gg_err(myPars.epsilon_gg)

    calib_eps_bb = myPars.epsilon_bb
    calib_eps_gg = myPars.epsilon_gg
    myShocks = Shocks(myPars)

    return myPars, myShocks, calib_eps_bb, calib_eps_gg

def calib_all_eps(myPars:Pars, main_path:str,
                eps_bb_tol:float, eps_gg_tol:float, eps_bb_target:float, eps_gg_target:float,
                eps_bb_min:float = -0.3, eps_bb_max:float = 0.3, eps_gg_min:float = -0.3, eps_gg_max:float = 0.3,
                eps_bb_max_iters:int = 30, eps_gg_max_iters:int = 30, 
                )-> Tuple[float, float, float, float, Dict[str, np.array], Dict[str, np.array]]:

    myPars, myShocks, calib_eps_bb, calib_eps_gg = calib_all_eps_numba(myPars, main_path,
                                                                       eps_bb_tol, eps_gg_tol, eps_bb_target, eps_gg_target,
                                                                       eps_bb_min, eps_bb_max, eps_gg_min, eps_gg_max,
                                                                       eps_bb_max_iters, eps_gg_max_iters)
    eps_gg_mom = eps_gg_moment(myPars, myShocks)
    eps_bb_mom = eps_bb_moment(myPars, myShocks)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, myShocks, state_sols)
    return calib_eps_bb, calib_eps_gg, eps_bb_mom, eps_gg_mom, state_sols, sim_lc

def get_all_targets(myPars: Pars, target_folder_path: str = None)-> Dict[str, float]:
    # Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    gets all the targets from the input folder, assumes the target .csv files are in the folder located at target_folder_path
    returns alpha_targ, w0_mean_targ, w0_sd_targ, w1_targ, w2_targ, wH_targ
    """
    if target_folder_path is None:
        target_folder_path = myPars.path + '/input/'
    alpha_targ = get_alpha_targ(myPars, target_folder_path)
    w0_mean_targ = get_w0_mean_targ(myPars, target_folder_path)
    w0_sd_targ = get_w0_sd_targ(myPars, target_folder_path)
    w1_targ = get_w1_targ(myPars, target_folder_path)
    w2_targ = get_w2_targ(myPars, target_folder_path)
    wH_targ = get_wH_targ(myPars, target_folder_path)
    phi_H_targ = get_phi_H_targ(myPars, target_folder_path)
    eps_gg_targ = get_eps_gg_targ(myPars, target_folder_path)
    eps_bb_targ = get_eps_bb_targ(myPars, target_folder_path)
    targ_dict = {'alpha': alpha_targ, 'w0_mu': w0_mean_targ, 'w0_sigma': w0_sd_targ, 'w1': w1_targ, 'w2': w2_targ, 'wH': wH_targ, 
                 'phi_H': phi_H_targ, 'eps_gg': eps_gg_targ, 'eps_bb': eps_bb_targ}
    return targ_dict

def calib_all(myPars: Pars, myShocks: Shocks, modify_shocks: bool = True, 
        do_wH_calib: bool = True, do_dpi_calib: bool = True, do_phi_H_calib: bool = False, 
        do_eps_gg_calib: bool = True, do_eps_bb_calib: bool = False,  

        w0_mu_min:float = 0.0, w0_mu_max:float = 30.0, w0_sigma_min:float = 0.001, w0_sigma_max = 1.0, 
        w1_min:float = 0.0, w1_max:float = 10.0, w2_min = -1.0, w2_max = 0.0, wH_min = -5.0, wH_max = 5.0, 
        dpi_BB_min:float = -0.0, dpi_BB_max:float = 2.0, dpi_GG_min:float = 0.0, dpi_GG_max:float = 1.0,
        phi_H_min: float = 0.0, phi_H_max: float = 20.0, eps_gg_min: float = -0.3, eps_gg_max: float = 0.3,
        eps_bb_min: float = -0.3, eps_bb_max: float = 0.3,

        alpha_tol:float = 0.001, w0_mu_tol:float = 0.001, w0_sigma_tol:float = 0.001, w1_tol:float = 0.001, w2_tol:float = 0.001, wH_tol:float = 0.001, 
        dpi_BB_tol:float = 0.001, dpi_GG_tol:float = 0.001, phi_H_tol:float = 0.001, eps_gg_tol:float = 0.001, eps_bb_tol:float = 0.001,

        calib_path: str = None,

        **targ_args: Dict[str, float] 
        )-> (Tuple[Pars, Shocks, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]):
    """
    calibrates all the parameters of the model
    takes arguments that represent the targets and tolerances for the calibration
    returns a tuple with the calibrated parameters, the state solutions and the simulations
    """
    targ_defaults ={
        'alpha': 0.40, 'w0_mu': 20.0, 'w0_sigma': 3.0,
        'w1': 0.2, 'w2': 0.2,'wH': 0.2,
        'phi_H': 0.1, 'dpi_BB': 0.5, 'dpi_GG': 0.3,
        'eps_gg': 0.4, 'eps_bb': 0.5
    }
    # update the defaults with the arguments
    targ_dict = {**targ_defaults, **targ_args}
    # unpack the dictionary
    alpha_mom_targ:float = targ_dict['alpha']
    w0_mu_mom_targ:float = targ_dict['w0_mu']
    w0_sigma_mom_targ:float = targ_dict['w0_sigma']
    w1_mom_targ:float = targ_dict['w1']
    w2_mom_targ:float = targ_dict['w2']
    wH_mom_targ:float = targ_dict['wH']
    phi_H_mom_targ:float = targ_dict['phi_H']
    dpi_BB_mom_targ:float = targ_dict['dpi_BB']
    dpi_GG_mom_targ:float = targ_dict['dpi_GG']
    eps_gg_mom_targ:float = targ_dict['eps_gg']
    eps_bb_mom_targ:float = targ_dict['eps_bb']

    # set up return arrays
    state_sols = {}
    sims = {}
    my_eps_bb_mom = -999.999
    my_eps_gg_mom = -999.999
    my_w0_mu_mom = -999.999
    my_w0_sigma_mom = -999.999
    my_w1_mom = -999.999
    my_w2_mom = -999.999
    my_wH_mom = -999.999
    my_alpha_mom = -999.999
    # could do this with a dictionary, where i update the model_moms as i calibrate each parameter
    # moms_dict = {'alpha': my_alpha_mom, 'w0_mu': my_w0_mu_mom, 'w0_sigma': my_w0_sigma_mom, 
    #              'w1': my_w1_mom, 'w2': my_w2_mom, 'wH': my_wH_mom, 'eps_gg': my_eps_gg_mom, 'eps_bb': my_eps_bb_mom}

    for i in range(myPars.max_calib_iters):
        #print()
        print(f"***** Calibration iteration {i} *****")
        # calibrate epsilon_bb (eps_bb)
        if do_eps_bb_calib and do_eps_gg_calib:
            my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
            my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
            # moms_dict['eps_bb'] = eps_bb_moment(myPars, myShocks)
            # moms_dict['eps_gg'] = eps_gg_moment(myPars, myShocks)
            if np.abs(my_eps_bb_mom - eps_bb_mom_targ) > eps_bb_tol or np.abs(my_eps_gg_mom - eps_gg_mom_targ) > eps_gg_tol:
                print("Calibrating epsilon_bb and epsilon_gg")
                calib_eps_bb, calib_eps_gg, my_eps_bb_mom, my_eps_gg_mom, state_sols, sims = calib_all_eps(myPars, calib_path, 
                                                                                                            eps_bb_tol, eps_gg_tol, eps_bb_mom_targ, eps_gg_mom_targ,
                                                                                                            eps_bb_min, eps_bb_max, eps_gg_min, eps_gg_max)
                myShocks = Shocks(myPars)
                # print(f"eps_bb error: {np.abs(my_eps_bb_mom - eps_bb_mom_targ)}")
                # print(f"eps_gg error: {np.abs(my_eps_gg_mom - eps_gg_mom_targ)}")
                # exit()
        else:
            if do_eps_bb_calib:
                my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
                if np.abs(my_eps_bb_mom - eps_bb_mom_targ) > eps_bb_tol:
                    print("Calibrating epsilon_bb")
                    calib_eps_bb, my_eps_bb_mom, state_sols, sims = calib_epsilon_bb(myPars, calib_path, eps_bb_tol, eps_bb_mom_targ, eps_bb_min, eps_bb_max)
                    myShocks = Shocks(myPars)
                    # print(f"eps_bb error: {np.abs(my_eps_bb_mom - eps_bb_mom_targ)}")
            # calibrate epsilon_gg (eps_gg)
            elif do_eps_gg_calib:
                my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
                if ((not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol)
                    and np.abs(my_eps_gg_mom - eps_gg_mom_targ) > eps_gg_tol)):
                    print("Calibrating epsilon_gg")
                    calib_eps_gg, my_eps_gg_mom, state_sols, sims = calib_epsilon_gg(myPars, myShocks, calib_path, eps_gg_tol, eps_gg_mom_targ, eps_gg_min, eps_gg_max)
                    myShocks = Shocks(myPars)
                    my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
                    # print(f"eps_bb error: {np.abs(my_eps_bb_mom - eps_bb_mom_targ)}")

        # calibrate w0_mu and w0_sigma 
        my_w0_mu_mom = w0_mu_moment(myPars, myShocks)
        if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
            and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
            and not (np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol)):
            print("Calibrating w0_mu")
            w0_weights, my_w0_mu_mom, state_sols, sims = calib_w0_mu(myPars, myShocks, calib_path, w0_mu_tol, w0_mu_mom_targ, w0_mu_min, w0_mu_max)
            my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
            my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
        my_w0_sigma_mom = w0_sigma_moment(myPars, myShocks) 
        if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
            and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
            and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and not(np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol)):
            print("Calibrating w0_sigma")
            w0_weights, my_w0_sigma_mom, state_sols, sims = calib_w0_sigma(myPars, myShocks, calib_path, w0_sigma_tol, w0_sigma_mom_targ, w0_sigma_min, w0_sigma_max)
            my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
            my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
            my_w0_mu_mom = w0_mu_moment(myPars, myShocks)

        # rest of calibration starts 
        if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
            and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
            and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol):
            print("Calibrating w1")
            w1_calib, my_w1_mom, state_sols, sims = calib_w1(myPars, myShocks, calib_path, w1_tol, w1_mom_targ, w1_min, w1_max)
            my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
            my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
            my_w0_mu_mom = w0_mu_moment(myPars, myShocks)
            my_w0_sigma_mom = w0_sigma_moment(myPars, myShocks)
            if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
                and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
                and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                and np.abs(my_w1_mom - w1_mom_targ) < w1_tol):
                print("Calibrating w2")
                w2_calib, my_w2_mom, state_sols, sims = calib_w2(myPars, myShocks, calib_path, w2_tol, w2_mom_targ, w2_min, w2_max)
                my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
                my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
                my_w0_mu_mom = w0_mu_moment(myPars, myShocks)
                my_w0_sigma_mom = w0_sigma_moment(myPars, myShocks)
                my_w1_mom = w1_moment(myPars, myShocks)
                if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
                    and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
                    and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                    and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol):
                    if do_wH_calib:
                        print("Calibrating wH")
                        wH_calib, my_wH_mom, state_sols, sims = calib_wH(myPars, myShocks, calib_path, wH_tol, wH_mom_targ, wH_min, wH_max)                        
                    my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
                    my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
                    my_w0_mu_mom = w0_mu_moment(myPars, myShocks)
                    my_w0_sigma_mom = w0_sigma_moment(myPars, myShocks)
                    my_w1_mom = w1_moment(myPars, myShocks)
                    my_w2_mom = w2_moment(myPars, myShocks)
                    if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
                        and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
                        and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                        and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol 
                        and (not do_wH_calib or np.abs(my_wH_mom - wH_mom_targ) < wH_tol)):
                        print("Calibrating alpha")
                        alpha_calib, my_alpha_mom, state_sols, sims, shocks = calib_alpha(myPars, myShocks, calib_path, alpha_tol, alpha_mom_targ)
                        my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
                        my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
                        my_w0_mu_mom = w0_mu_moment(myPars, myShocks)
                        my_w0_sigma_mom = w0_sigma_moment(myPars, myShocks)
                        my_w1_mom = w1_moment(myPars, myShocks)
                        my_w2_mom = w2_moment(myPars, myShocks)
                        my_wH_mom = wH_moment(myPars, myShocks)
                        if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
                            and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
                            and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                            and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol 
                            and (not do_wH_calib or np.abs(my_wH_mom - wH_mom_targ) < wH_tol) and np.abs(my_alpha_mom - alpha_mom_targ) < alpha_tol):
                            if do_phi_H_calib:
                                print("Calibrating phi_H")
                                phi_H_calib, my_phi_H_mom, state_sols, sims = calib_phi_H(myPars, calib_path, phi_H_tol, phi_H_mom_targ, phi_H_min, phi_H_max)
                            else:
                                lab_sims = sims['lab']
                                my_phi_H_mom = phi_H_moment(myPars, lab_sims)
                            my_eps_bb_mom = eps_bb_moment(myPars, myShocks)
                            my_eps_gg_mom = eps_gg_moment(myPars, myShocks)
                            my_w0_mu_mom = w0_mu_moment(myPars, myShocks)
                            my_w0_sigma_mom = w0_sigma_moment(myPars, myShocks)
                            my_w1_mom = w1_moment(myPars, myShocks)
                            my_w2_mom = w2_moment(myPars, myShocks)
                            my_wH_mom = wH_moment(myPars, myShocks)
                            my_alpha_mom = alpha_moment_giv_sims(myPars, sims)
                        if ((not do_eps_gg_calib or (np.abs(my_eps_gg_mom - eps_gg_mom_targ) < eps_gg_tol))
                            and (not do_eps_bb_calib or (np.abs(my_eps_bb_mom - eps_bb_mom_targ) < eps_bb_tol))
                            and np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                            and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol 
                            and (not do_wH_calib or np.abs(my_wH_mom - wH_mom_targ) < wH_tol) and np.abs(my_alpha_mom - alpha_mom_targ) < alpha_tol
                            and (not do_phi_H_calib or np.abs(my_phi_H_mom - phi_H_mom_targ) < phi_H_tol)):
                            print(f"Calibration converged after {i+1} iterations")
                            if not do_wH_calib:
                                print("********** wH calibration was skipped **********")
                            if not do_phi_H_calib:
                                print("********** phi_H calibration was skipped **********")
                            if not do_eps_gg_calib:
                                print("********** epsilon_gg calibration was skipped ********")
                            else:
                                print(f"epsilon_gg = {myPars.epsilon_gg}, epsilon_gg mom = {my_eps_gg_mom}, epsilon_gg mom targ = {eps_gg_mom_targ}")
                            if not do_eps_bb_calib:
                                print("********** epsilon_bb calibration was skipped ********")
                            else:
                                print(f"epsilon_bb = {myPars.epsilon_bb}, epsilon_bb mom = {my_eps_bb_mom}, epsilon_bb mom targ = {eps_bb_mom_targ}")
                            print(f"w0_weights = {myPars.lab_fe_weights}, w0_mean = {my_w0_mu_mom}, w0_mean_targ = {w0_mu_mom_targ}") 
                            print(f"w0_sd = {my_w0_sigma_mom}, w0_sd_targ = {w0_sigma_mom_targ}")
                            print(f"w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_mom}, w1 mom targ = {w1_mom_targ}")
                            print(f"w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_mom}, w2 mom targ = {w2_mom_targ}")
                            print(f"wH = {myPars.wH_coeff}, wH moment = {my_wH_mom}, wH mom targ = {wH_mom_targ}")
                            print(f"alpha = {myPars.alpha}, alpha moment = {my_alpha_mom}, alpha mom targ = {alpha_mom_targ}")
                            print(f"phi_H = {myPars.phi_H}, phi_H moment = {my_phi_H_mom}, phi_H mom targ = {phi_H_mom_targ}")
                            moms_dict = {'alpha': my_alpha_mom, 'w0_mu': my_w0_mu_mom, 'w0_sigma': my_w0_sigma_mom, 
                                            'w1': my_w1_mom, 'w2': my_w2_mom, 'wH': my_wH_mom, 'phi_H': my_phi_H_mom,
                                            'eps_gg': my_eps_gg_mom, 'eps_bb': my_eps_bb_mom}

                            return myPars, myShocks, state_sols, sims, moms_dict

    # calibration does not converge
    print(f"Calibration did not converge after {myPars.max_calib_iters} iterations")
    if not do_wH_calib:
        print("********** wH calibration was skipped **********")
    if not do_phi_H_calib:
        print("********** phi_H calibration was skipped **********")
    if not do_eps_gg_calib:
        print("********** epsilon_gg calibration was skipped ********")
    else:
        print(f"epsilon_gg = {myPars.epsilon_gg}, epsilon_gg mom = {my_eps_gg_mom}, epsilon_gg mom targ = {eps_gg_mom_targ}")
    if not do_eps_bb_calib:
        print("********** epsilon_bb calibration was skipped ********")
    else:
        print(f"epsilon_bb = {myPars.epsilon_bb}, epsilon_bb mom = {my_eps_bb_mom}, epsilon_bb mom targ = {eps_bb_mom_targ}")
    print(f"w0_weights = {myPars.lab_fe_weights}, w0_mean = {my_w0_mu_mom}, w0_mean_targ = {w0_mu_mom_targ}") 
    print(f"w0_sd = {my_w0_sigma_mom}, w0_sd_targ = {w0_sigma_mom_targ}")
    print(f"w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_mom}, w1 mom targ = {w1_mom_targ}")
    print(f"w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_mom}, w2 mom targ = {w2_mom_targ}")
    print(f"wH = {myPars.wH_coeff}, wH moment = {my_wH_mom}, wH mom targ = {wH_mom_targ}")
    print(f"alpha = {myPars.alpha}, alpha moment = {my_alpha_mom}, alpha mom targ = {alpha_mom_targ}")
    print(f"phi_H = {myPars.phi_H}, phi_H moment = {my_phi_H_mom}, phi_H mom targ = {phi_H_mom_targ}")
    moms_dict = {'alpha': my_alpha_mom, 'w0_mu': my_w0_mu_mom, 'w0_sigma': my_w0_sigma_mom, 
                    'w1': my_w1_mom, 'w2': my_w2_mom, 'wH': my_wH_mom, 'phi_H': my_phi_H_mom,
                    'eps_gg': my_eps_gg_mom, 'eps_bb': my_eps_bb_mom}

    return myPars, myShocks, state_sols, sims, moms_dict

if __name__ == "__main__":
    start_time = time.perf_counter()