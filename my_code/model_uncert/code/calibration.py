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
import os
import subprocess

# My code
import my_toolbox as tb
import solver
import model_uncert as model
import pars_shocks as ps
from pars_shocks import Pars, Shocks
import simulate
import plot_lc as plot_lc
import io_manager as io

def calib_w0_mu(myPars: Pars, main_path: str, tol:float, target:float, w0_mu_min:float, w0_mu_max:float
                  )-> Tuple[np.ndarray, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    calibrates the wage fixed effect weights to match the target mean wage
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    target: the target mean wage
    returns a tuple with the calibrated weights, the mean_wage, the state solutions and the simulations
    """
    myShocks = Shocks(myPars)
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_mean_calib_params.csv")
    mean_wage = -999.999
    sate_sols = {}
    sim_lc = {}

    # print(f"lab_fe_collapse_weight_wages = {lab_fe_collapse_weight_wages}")
    # print(f"myPars.wH_coeff = {myPars.wH_coeff}")
    
    # # define the lambda function to find the zero of
    get_w0_mean_diff = lambda new_mu: w0_mu_mom_giv_mu(myPars, new_mu) - target
    calibrated_mu = tb.bisection_search(get_w0_mean_diff, w0_mu_min, w0_mu_max, tol, myPars.max_iters, myPars.print_screen)
    my_lab_fe_grid = np.exp(myPars.lab_fe_grid)
    calibrated_weights = tb.Taucheniid(myPars.lab_fe_tauch_sigma, myPars.lab_fe_grid_size, mean = calibrated_mu, state_grid = my_lab_fe_grid)[0] 

    #update the labor fixed effect weights
    myPars.lab_fe_weights = calibrated_weights
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    mean_wage = w0_mu_moment(myPars)

    if myPars.print_screen >= 1:
        print(f"w0_mean calibration exited: mean wage = {mean_wage}, target mean wage = {target}")
        print(f"Calibrated weights = {calibrated_weights}")

    return calibrated_weights, mean_wage, state_sols, sim_lc

def w0_mu_mom_giv_mu(myPars: Pars, new_mu:float)-> float:
    myPars.lab_fe_tauch_mu = new_mu 
    my_lab_fe_grid = np.exp(myPars.lab_fe_grid)
    new_weights, tauch_state_grid = tb.Taucheniid(myPars.lab_fe_tauch_sigma, myPars.lab_fe_grid_size, mean = new_mu, state_grid = my_lab_fe_grid)
    myPars.lab_fe_weights = new_weights
    return w0_mu_moment(myPars)

@njit
def w0_mu_moment(myPars: Pars)-> float:
    " get the weighted mean of wages for period 0"
    myShocks = Shocks(myPars)
    first_per_weighted_wages = model.gen_weighted_wage_hist(myPars, myShocks)[:,:,:,0] 
    mean_first_per_wage = np.sum(first_per_weighted_wages)
    return mean_first_per_wage

def calib_w0_sigma(myPars: Pars, main_path: str, tol:float, target:float, w0_sigma_min:float, w0_sigma_max:float
                )-> Tuple[np.ndarray, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    calibrates the wage fixed effect weights to match the target standard deviation of wages
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    sd_target: the target standard deviation of wages
    returns a tuple with the calibrated weights, the standard deviation of wages, the state solutions and the simulations
    """
    myShocks = Shocks(myPars)
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_sd_calib_params.csv")
    sd_wage = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w0_sd_diff = lambda new_sigma: w0_sigma_mom_giv_sigma(myPars, new_sigma) - target
    # calibrated_sigma = tb.bisection_search(get_w0_sd_diff, w0_sigma_min, w0_sigma_max, tol, myPars.max_iters, myPars.print_screen)
    calibrated_sigma = tb.bisection_search(get_w0_sd_diff, w0_sigma_min, w0_sigma_max, tol, myPars.max_iters, myPars.print_screen)
    my_lab_fe_grid = np.exp(myPars.lab_fe_grid)
    calibrated_weights = tb.Taucheniid(calibrated_sigma, myPars.lab_fe_grid_size, mean = myPars.lab_fe_tauch_mu, state_grid = my_lab_fe_grid)[0]

    #update the labor fixed effect weights
    myPars.lab_fe_weights = calibrated_weights
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    sd_wage = w0_sigma_moment(myPars)

    if myPars.print_screen >= 1:
        print(f"w0_sd calibration exited: sd wage = {sd_wage}, target sd wage = {target}")
        print(f"Calibrated weights = {calibrated_weights}")

    return calibrated_weights, sd_wage, state_sols, sim_lc

def w0_sigma_mom_giv_sigma(myPars: Pars, new_sigma:float)-> float:
    myPars.lab_fe_tauch_sigma = new_sigma
    my_lab_fe_grid = np.exp(myPars.lab_fe_grid)
    new_weights, tauch_state_grid = tb.Taucheniid(new_sigma, myPars.lab_fe_grid_size, mean = myPars.lab_fe_tauch_mu, state_grid = my_lab_fe_grid)
    myPars.lab_fe_weights = new_weights
    return w0_sigma_moment(myPars)

@njit
def w0_sigma_moment(myPars: Pars)-> float:
    # " get the weighted standard deviation of wages for period 0"
    # myShocks = Shocks(myPars)
    # first_per_wages = model.gen_wage_hist(myPars, myShocks)[:,:,:,0]
    # first_per_weighted_wages = model.gen_weighted_wage_hist(myPars, myShocks)[:,:,:,0] 
    # mean_first_per_wage = np.sum(first_per_weighted_wages)

    # # Calculate the weighted variance
    # deviations =  first_per_wages - mean_first_per_wage
    # squared_deviations = deviations ** 2
    # # Apply the weights
    # weighted_squared_deviations = squared_deviations * (1.0 / myPars.sim_draws)
    # weighted_squared_deviations = weighted_squared_deviations * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis]
    # weighted_squared_variance = np.sum(weighted_squared_deviations * myPars.lab_fe_weights[:, np.newaxis, np.newaxis])
    # # Calculate the weighted standard deviation
    # sd_first_per_wage = np.sqrt(weighted_squared_variance)
    sd_first_per_wage = w0_moments(myPars)[1]
    # print(f"sd_first_per_wage = {sd_first_per_wage}")
    return sd_first_per_wage


def calib_w0(myPars: Pars, main_path: str, mean_target:float, sd_target:float):
    """
    calibrates the wage fixed effect weights to match the target mean and standard deviation of wages
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    mean_target: the target mean wage for period 0
    sd_target: the target standard deviation of wages for period 0
    returns a tuple with the calibrated weights, the mean_wage, the sd_wage, state_solutions and sim_lc
    """
    myShocks = Shocks(myPars)
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_calib_params.csv")
    mean_wage = -999.999
    sd_wage = -999.999
    sate_sols = {}
    sim_lc = {}

    first_per_wages = model.gen_wage_hist(myPars, myShocks)[:,:,:,0]
    weighted_first_per_wages = first_per_wages * (1/myPars.sim_draws)
    weighted_first_per_wages = weighted_first_per_wages * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis]
    lab_fe_collapse_weight_wages = np.sum(first_per_wages, axis = tuple(range(1,first_per_wages.ndim)))
    # print(f"lab_fe_collapse_weight_wages = {lab_fe_collapse_weight_wages}")
    # print(f"myPars.wH_coeff = {myPars.wH_coeff}")

    my_weights = tb.optimize_weights(lab_fe_collapse_weight_wages, mean_target, sd_target, myPars.lab_fe_grid_size, myPars.print_screen)

    #update the labor fixed effect weights
    myPars.lab_fe_weights = my_weights
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    mean_wage, sd_wage = w0_moments(myPars)

    if myPars.print_screen >= 1:
        print(f"Calibration exited: mean wage = {mean_wage}, target mean wage = {mean_target}, sd wage = {sd_wage}, target sd wage = {sd_target}")
        print(f"Calibrated weights = {my_weights}")

    return my_weights, mean_wage, sd_wage, state_sols, sim_lc 


@njit
def w0_moments(myPars: Pars)-> Tuple[float, float]:
    " get the weighted mean and standard deviation of wages for period 0"
    myShocks = Shocks(myPars)
    first_per_wages = model.gen_wage_hist(myPars, myShocks)[:,:,:,0]
    first_per_weighted_wages = model.gen_weighted_wage_hist(myPars, myShocks)[:,:,:,0] 
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
    data_mom_col_ind = 1
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return mean_wage_by_age[0]

def get_w0_sd_targ(myPars: Pars, target_folder_path: str)-> float:
    """ get the target standard deviation of wages for period 0 from myPars.path + '/input/wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_moments_path = target_folder_path + '/wage_moments.csv'
    data_mom_col_ind = 2
    sd_wage_col= tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return sd_wage_col[0]

def calib_w1(myPars: Pars, main_path: str, tol:float, target:float, w1_min:float, w1_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w1_calib_params.csv")
    w1_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w1_diff = lambda new_coeff: w1_moment_giv_w1(myPars, new_coeff) - target
    calibrated_w1 = tb.bisection_search(get_w1_diff, w1_min, w1_max, tol, myPars.max_iters, myPars.print_screen)
    # update the wage coeff grid
    for i in range(myPars.lab_fe_grid_size):
        myPars.wage_coeff_grid[i, 1] = calibrated_w1

    # solve, simulate model for the calibrated w1
    w1_moment = w1_moment_giv_w1(myPars,  calibrated_w1) 
    io.print_params_to_csv(myPars, path = main_path, file_name = "w1_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: w1 = {calibrated_w1}, wage growth = {w1_moment}, target wage growth = {target}") 

    return calibrated_w1, w1_moment, state_sols, sim_lc

@njit
def w1_moment_giv_w1(myPars: Pars, new_coeff:float)-> float:
    """ updates the wage_coeff_grid and returns the new wage growth """
    # for i in range(myPars.lab_fe_grid_size): 
    #     myPars.wage_coeff_grid[i, 1] = new_coeff
    myPars.set_w1(new_coeff)
    return w1_moment(myPars)

@njit
def w1_moment(myPars: Pars)-> float:
    """ calculates the wage growth given the model parameters """
    myShocks = Shocks(myPars)
    wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    # mean_axis = myPars.tuple_sim_ndim[:-1]
    # mean_axis = tuple(range(wage_sims.ndim - 1))
    # mean_axis = tb.range_tuple_numba(wage_sims.ndim - 1)
    # mean_axis = tuple(range(myPars.len_state_space_shape_sims - 1))
    # mean_wage_by_age = np.sum(wage_sims, axis = mean_axis)
    # mean_wage_by_age =tb.sum_axis_numba(wage_sims, mean_axis)
    mean_wage_by_age = tb.sum_last_axis_numba(wage_sims)

    wage_diff = log(np.max(mean_wage_by_age)) - log(mean_wage_by_age[0])
    # mean_axis = tuple(range(myPars.len_state_space_shape_sims - 1))
    # wage_diff = w1_moment_numba(myPars, mean_axis)
    return wage_diff

@njit
def w1_moment_numba(myPars: Pars, sum_axis: Tuple[int])-> float:
    """ calculates the wage growth given the model parameters """
    myShocks = Shocks(myPars)
    wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    mean_wage_by_age = np.sum(wage_sims, axis = sum_axis)
    wage_diff = log(np.max(mean_wage_by_age)) - log(mean_wage_by_age[0])
    return wage_diff 

def get_w1_targ(myPars: Pars, target_folder_path: str)-> float:
    """ gets the target wage growth until age '60' from myPars.path + '/input/wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_moments_path = target_folder_path + '/wage_moments.csv'
    data_mom_col_ind = 1
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    # want to get wages before age 60
    age_60_ind = 60 - myPars.start_age
    my_max = np.max(mean_wage_by_age[:age_60_ind])
    return log(my_max)- log(mean_wage_by_age[0])

def calib_w2(myPars: Pars, main_path: str, tol:float, target:float, w2_min:float, w2_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w2_calib_params.csv")
    w2_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w2_diff = lambda new_coeff: w2_moment_giv_w2(myPars, new_coeff) - target
    calibrated_w2 = tb.bisection_search(get_w2_diff, w2_min, w2_max, tol, myPars.max_iters, myPars.print_screen)
    # update the wage coeff grid
    for i in range(myPars.lab_fe_grid_size):
        myPars.wage_coeff_grid[i, 2] = calibrated_w2
    
    # solve, simulate model for the calibrated w2
    w2_moment = w2_moment_giv_w2(myPars, calibrated_w2)
    io.print_params_to_csv(myPars, path = main_path, file_name = "w2_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: w2 = {calibrated_w2}, wage growth = {w2_moment}, target wage growth = {target}")

    return calibrated_w2, w2_moment, state_sols, sim_lc

@njit
def w2_moment_giv_w2(myPars: Pars, new_coeff:float)-> float:
    """ updates the wage_coeff_grid skipping the first row coefficient and returns the new wage decay """
    # for i in range(1, myPars.lab_fe_grid_size):
    # for i in range(myPars.lab_fe_grid_size):
    #     myPars.wage_coeff_grid[i, 2] = new_coeff
    myPars.set_w2(new_coeff)
    return w2_moment(myPars)

@njit
def w2_moment(myPars: Pars)-> float:
    """ calculates the wage decay given the model parameters """
    myShocks = Shocks(myPars)
    wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    mean_wage_by_age = tb.sum_last_axis_numba(wage_sims)
    # mean_wage_by_age = np.sum(wage_sims, axis = tuple(range(wage_sims.ndim - 1)))
    # mean_wage = np.sum(wage_sims, axis=tuple(range(wage_sims.ndim - 1)))
    # if np.all(mean_wage > 0):
    #     wage_diff = log(np.max(mean_wage)) - log(mean_wage[myPars.J-1])
    # else:
    #     print("Invalid values in mean_wage for log operation")
    #     print(f"mean_wage = {mean_wage}")
    #     # return
    wage_diff = log(np.max(mean_wage_by_age)) - log(mean_wage_by_age[myPars.J-1])
    return wage_diff

def get_w2_targ(myPars: Pars, target_folder_path: str)-> float:
    """ gets the target wage decay starting at age 60 from myPars.path + '/input/wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_moments_path =   target_folder_path + '/wage_moments.csv'
    data_mom_col_ind = 1
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    age_60_ind = 60 - myPars.start_age
    my_max = np.max(mean_wage_by_age[:age_60_ind])
    return log(my_max) - log(mean_wage_by_age[-1])

def calib_wH(myPars: Pars, myShocks: Shocks, main_path: str, tol:float, target:float, wH_min:float, wH_max:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_wH_calib_params.csv")
    wH_moment = -999.999
    sate_sols = {}
    sim_lc = {}

    # define the lambda function to find the zero of
    get_wH_diff = lambda new_coeff: wH_moment_giv_wH(myPars, myShocks,  new_coeff) - target
    calibrated_wH = tb.bisection_search(get_wH_diff, wH_min, wH_max, tol, myPars.max_iters, myPars.print_screen)
    myPars.wH_coeff = calibrated_wH

    # solve, simulate model for the calibrated wH
    wH_moment = wH_moment_giv_wH(myPars, myShocks,  calibrated_wH)
    io.print_params_to_csv(myPars, path = main_path, file_name = "wH_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: wH = {calibrated_wH}, wage premium = {wH_moment}, target wage premium = {target}")

    return calibrated_wH, wH_moment, state_sols, sim_lc

# @njit
def wH_moment_giv_wH(myPars: Pars, myShocks: Shocks, new_coeff:float)-> float:
    myPars.wH_coeff = new_coeff
    return wH_moment(myPars, myShocks)

# @njit
def wH_moment(myPars: Pars, myShocks: Shocks)-> float:
    """returns the wage premium given the model parameters"""
    wage_sims = model.gen_wage_hist(myPars, myShocks) 
    H_hist = myShocks.H_hist
    # healthy wage is the wage when the health state is 1
    healthy_wage_sims = wage_sims * H_hist[:, :, :, :myPars.J]
    mean_healthy_wage = np.mean(healthy_wage_sims[healthy_wage_sims > 0])
    # mean_healthy_wage = tb.mean_nonzero_numba(healthy_wage_sims)
    unhealthy_wage_sims = wage_sims * (1 - H_hist[:, :, :, :myPars.J])
    mean_unhealthy_wage = np.mean(unhealthy_wage_sims[unhealthy_wage_sims > 0])
    # mean_unhealthy_wage = tb.mean_nonzero_numba(unhealthy_wage_sims)

    wage_prem = log(mean_healthy_wage) - log(mean_unhealthy_wage)
    return wage_prem

def get_wH_targ(myPars: Pars, target_folder_path: str)-> float:
    """ get the target wage premium from myPars.path + '/input/MH_wage_moments.csv' """
    # data_moments_path = myPars.path + '/input/MH_wage_moments.csv'
    data_moments_path = target_folder_path + '/MH_wage_moments.csv'
    data_mom_col_ind = 0
    mean_wage_diff = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return mean_wage_diff[0]

def calib_alpha(myPars: Pars, main_path: str, lab_tol:float, mean_lab_targ:float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray], Shocks]:
    """
    calibrates the alpha parameter of the model to match the target mean labor worked.
    takes the following arguments:
    myPars: the parameters of the model
    main_path: the path to the main directory
    lab_tol: the tolerance for the calibration
    mean_lab_targ: the target mean labor worked
    returns a tuple with the calibrated alpha, the mean labor worked, the state solutions and the simulated labor choices
    """
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_alpha_calib_params.csv")
    mean_lab = -999.999
    state_sols = {}
    sim_lc = {}
    alpha_min = 0.00001 #alpha of 0 is not economically meaningful
    alpha_max = 1.0
    
    # define the lambda function to find the zero of     
    get_mean_lab_diff = lambda new_alpha: alpha_moment_giv_alpha(myPars, new_alpha, main_path)[0] - mean_lab_targ 
    calib_alpha = tb.bisection_search(get_mean_lab_diff, alpha_min, alpha_max, lab_tol, myPars.max_iters, myPars.print_screen) 
    myPars.alpha = calib_alpha # myPars is mutable this also happens inside solve_mean_lab_giv_alpha but i think its more readable here
    
    # solve, simulate and plot model for the calibrated alpha
    mean_lab, state_sols, sim_lc, shocks = alpha_moment_giv_alpha(myPars, calib_alpha, main_path)
    io.print_params_to_csv(myPars, path = main_path, file_name = "alpha_calib_params.csv")
    if myPars.print_screen >= 1:
        print(f"Calibration exited: alpha = {calib_alpha}, mean labor worked = {mean_lab}, target mean labor worked = {mean_lab_targ}")
    
    return calib_alpha, mean_lab, state_sols, sim_lc, shocks
    

def alpha_moment_giv_alpha(myPars : Pars,  new_alpha:float, main_path : str = None) ->Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray], Shocks]:
    '''
        solves the model for a given alpha and returns the alpha moment: the mean labor worked, the model solutions and simulations
    ''' 
    myPars.alpha = new_alpha
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    mean_lab = alpha_moment_giv_sims(myPars, sim_lc) 
    return mean_lab, state_sols, sim_lc, shocks

def alpha_moment_giv_sims(myPars: Pars, sims: Dict[str, np.ndarray])-> float:
    """
    calculates the mean labor worked given the simulations
    takes the following arguments:
    myPars: the parameters of the model
    sims: the simulations of the model
    returns the mean labor worked
    """
    labor_sims = sims['lab'][:, :, :, :myPars.J]
    weighted_labor_sims = model.gen_weighted_sim(myPars, labor_sims) 
    mean_lab_by_age = np.sum(weighted_labor_sims, axis = tuple(range(weighted_labor_sims.ndim-1)))
    mean_lab = np.mean(mean_lab_by_age)
    return mean_lab

def get_alpha_targ(myPars: Pars, target_folder_path: str) -> float:
    """
    reads alpha target moment from myPars.path + '/input/labor_moments.csv'
    """
    # data_moments_path = myPars.path + '/input/labor_moments.csv'
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
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_phi_H_calib_params.csv")
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
    io.print_params_to_csv(myPars, path = main_path, file_name = "phi_H_calib_params.csv")
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
    return phi_H_moment(myPars, lab_sim_lc, shocks), state_sols, sim_lc

# def phi_H_moment(myPars: Pars, sims: Dict[str, np.ndarray], shocks: Shocks)-> float:

@njit
def phi_H_moment(myPars: Pars, lab_sims: np.ndarray, shocks: Shocks)-> float:
    '''
    returns the difference in mean log hours worked between the bad MH and good MH states 
    '''
    # pre_retirement_age = 55
    # pre_retirement_age_ind = np.where(myPars.age_grid == pre_retirement_age)[0][0]
    # lab_sims = lab_sims[:, :, :, :pre_retirement_age_ind]
    # H_hist: np.ndarray = shocks.H_hist[:, :, :, :pre_retirement_age_ind]
    lab_sims = lab_sims[:, :, :, :myPars.J]
    H_hist: np.ndarray = shocks.H_hist[:, :, :, :myPars.J]

    # take the log where hours are positive
    for i in range(lab_sims.size):
        if lab_sims.flat[i] <= 0:
            lab_sims.flat[i] = 1e-10
    log_lab_sims = np.log(lab_sims)

    bad_MH_log_lab_sims = log_lab_sims * (H_hist == 0)
    good_MH_log_lab_sims = log_lab_sims * (H_hist == 1)

    bad_MH_log_lab_mean = model.wmean_non_zero(myPars, bad_MH_log_lab_sims)
    good_MH_log_lab_mean = model.wmean_non_zero(myPars, good_MH_log_lab_sims)
    # bad_MH_log_lab_mean = np.mean(bad_MH_log_lab_sims, where = bad_MH_log_lab_sims != 0)
    # good_MH_log_lab_mean = np.mean(good_MH_log_lab_sims, where = good_MH_log_lab_sims != 0)
    diff = good_MH_log_lab_mean - bad_MH_log_lab_mean   

    # print(f"good_MH_log_lab_mean: {good_MH_log_lab_mean} - bad_MH_log_lab_mean: {bad_MH_log_lab_mean} = diff: {diff}")

    return diff
    
def get_phi_H_targ(myPars: Pars, target_folder_path: str)-> float:
    '''
    reads phi_H target moment from myPars.path + '/input/MH_hours_moments.csv'
    '''
    # data_moments_path = myPars.path + '/input/MH_labor_moments.csv'
    data_moments_path = target_folder_path + '/MH_hours_moments.csv'
    data_mom_col_ind = 0
    mean_log_lab_diff = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return mean_log_lab_diff[0]

def get_all_targets(myPars: Pars, target_folder_path: str = None)-> Tuple[float, float, float, float, float, float]:
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
    return alpha_targ, w0_mean_targ, w0_sd_targ, w1_targ, w2_targ, wH_targ, phi_H_targ

def calib_all(myPars: Pars, myShocks: Shocks, do_wH_calib: bool = True, do_phi_H_calib: bool = True,  
        alpha_mom_targ:float = 0.40, w0_mu_mom_targ:float = 20.0, w0_sigma_mom_targ:float = 3.0, w1_mom_targ:float = 0.2, w2_mom_targ:float = 0.2, 
        wH_mom_targ:float = 0.2, phi_H_mom_targ:float = 0.02, 
        w0_mu_min:float = 0.0, w0_mu_max:float = 100.0, w0_sigma_min:float = 0.001, w0_sigma_max = 50.0, 
        w1_min:float = 0.0, w1_max:float = 10.0, w2_min = -1.0, w2_max = 0.0, wH_min = -5.0, wH_max = 5.0, 
        phi_H_min:float = 0.00001, phi_H_max:float = 0.25,
        alpha_tol:float = 0.001, w0_mu_tol:float = 0.001, w0_sigma_tol:float = 0.0001, w1_tol:float = 0.001, w2_tol:float = 0.001, 
        wH_tol:float = 0.001, phi_H_tol:float = 1e-2,
        calib_path: str = None)-> (
        Tuple[float, np.ndarray, float, float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]):
    """
    calibrates all the parameters of the model
    takes arguments that represent the targets and tolerances for the calibration
    returns a tuple with the calibrated parameters, the state solutions and the simulations
    """
    # set up return arrays
    state_sols = {}
    sims = {}
    my_alpha_mom = -999.999
    my_w0_mu_mom = -999.999
    my_w0_sigma_mom = -999.999
    my_w1_mom = -999.999
    my_w2_mom = -999.999
    my_wH_mom = -999.999

    for i in range(myPars.max_calib_iters):
        print(f"***** Calibration iteration {i} *****")
        # print("Calibrating w0_mu")
        w0_weights, my_w0_mu_mom, state_sols, sims = calib_w0_mu(myPars, calib_path, w0_mu_tol, w0_mu_mom_targ, w0_mu_min, w0_mu_max)
        # print(f"""Calibrated w0 weights = {w0_weights}, w0 mean = {my_w0_mu_mom}, w0 mean targ = {w0_mu_mom_targ}""")
        if (np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol):
            # print("Calibrating w0_sigma")
            w0_weights, my_w0_sigma_mom, state_sols, sims = calib_w0_sigma(myPars, calib_path, w0_sigma_tol, w0_sigma_mom_targ, w0_sigma_min, w0_sigma_max)
            my_w0_mu_mom = w0_mu_moment(myPars)
            # print(f"""Calibrated w0 weights = {w0_weights}, w0 mean = {my_w0_mu_mom}, w0 mean targ = {w0_mu_mom_targ},
            #                                             w0_sigma = {my_w0_sigma_mom}, w0_sigma targ = {w0_sigma_mom_targ}""")
            if (np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol):
                # print("Calibrating w1")
                w1_calib, my_w1_mom, state_sols, sims = calib_w1(myPars, calib_path, w1_tol, w1_mom_targ, w1_min, w1_max)
                my_w0_mu_mom = w0_mu_moment(myPars)
                my_w0_sigma_mom = w0_sigma_moment(myPars)
                # print(f"""Calibrated w0 weights = {w0_weights}, w0 mean = {my_w0_mu_mom}, w0 mean targ = {w0_mu_mom_targ},
                #                                         w0_sigma = {my_w0_sigma_mom}, w0_sigma targ = {w0_sigma_mom_targ},
                #                                         w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_mom}, w1 mom targ = {w1_mom_targ}""")
                if (np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                    and np.abs(my_w1_mom - w1_mom_targ) < w1_tol):
                    # print("Calibrating w2")
                    w2_calib, my_w2_mom, state_sols, sims = calib_w2(myPars, calib_path, w2_tol, w2_mom_targ, w2_min, w2_max)
                    my_w0_mu_mom = w0_mu_moment(myPars)
                    my_w0_sigma_mom = w0_sigma_moment(myPars)
                    my_w1_mom = w1_moment(myPars)
                    if (np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                        and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol):
                        if do_wH_calib:
                            # print("Calibrating wH")
                            wH_calib, my_wH_mom, state_sols, sims = calib_wH(myPars, myShocks, calib_path, wH_tol, wH_mom_targ, wH_min, wH_max)                        
                        my_w0_mu_mom = w0_mu_moment(myPars)
                        my_w0_sigma_mom = w0_sigma_moment(myPars)
                        my_w1_mom = w1_moment(myPars)
                        my_w2_mom = w2_moment(myPars)
                        if (np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                            and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol
                            and (not do_wH_calib or np.abs(my_wH_mom - wH_mom_targ) < wH_tol)):
                            # calibrating phi_H
                            if do_phi_H_calib:
                                # print("Calirating phi_H")
                                phi_H_calib, my_phi_H_mom, state_sols, sims = calib_phi_H(myPars, calib_path, phi_H_tol, phi_H_mom_targ, phi_H_min, phi_H_max)
                            else:
                                shocks = Shocks(myPars) # this is inefficient we could calculate the shocks at an earlier calib step maybe
                                my_phi_H_mom = phi_H_moment(myPars, sims["lab"], shocks)
                            my_w0_mu_mom = w0_mu_moment(myPars)
                            my_w0_sigma_mom = w0_sigma_moment(myPars)
                            my_w1_mom = w1_moment(myPars)
                            my_w2_mom = w2_moment(myPars)
                            my_wH_mom = wH_moment(myPars, myShocks)
                            # # print("Calibrating alpha")
                            # alpha_calib, my_alpha_mom, state_sols, sims, shocks = calib_alpha(myPars, calib_path, alpha_tol, alpha_mom_targ)
                            # my_w0_mu_mom = w0_mu_moment(myPars)
                            # my_w0_sigma_mom = w0_sigma_moment(myPars)
                            # my_w1_mom = w1_moment(myPars)
                            # my_w2_mom = w2_moment(myPars)
                            # my_wH_mom = wH_moment(myPars, myShocks)
                            if(np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                                and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol 
                                and (not do_wH_calib or np.abs(my_wH_mom - wH_mom_targ) < wH_tol) 
                                and(not do_phi_H_calib or np.abs(my_phi_H_mom - phi_H_mom_targ) < phi_H_tol)):
                                # and np.abs(my_alpha_mom - alpha_mom_targ) < alpha_tol):
                                # print("Calibrating alpha")
                                alpha_calib, my_alpha_mom, state_sols, sims, shocks = calib_alpha(myPars, calib_path, alpha_tol, alpha_mom_targ)
                                my_w0_mu_mom = w0_mu_moment(myPars)
                                my_w0_sigma_mom = w0_sigma_moment(myPars)
                                my_w1_mom = w1_moment(myPars)
                                my_w2_mom = w2_moment(myPars)
                                my_wH_mom = wH_moment(myPars, myShocks)
                                my_phi_H_mom = phi_H_moment(myPars, sims["lab"], shocks)
                                # print(f"my_phi_H_mom after alpha calib = {my_phi_H_mom}")
                                # # calibrating phi_H
                                # if do_phi_H_calib:
                                #     # print("Calirating phi_H")
                                #     phi_H_calib, my_phi_H_mom, state_sols, sims = calib_phi_H(myPars, calib_path, phi_H_tol, phi_H_mom_targ, phi_H_min, phi_H_max)
                                # else:
                                #     my_phi_H_mom = phi_H_moment(myPars, sims["lab"], shocks)
                                # my_w0_mu_mom = w0_mu_moment(myPars)
                                # my_w0_sigma_mom = w0_sigma_moment(myPars)
                                # my_w1_mom = w1_moment(myPars)
                                # my_w2_mom = w2_moment(myPars)
                                # my_wH_mom = wH_moment(myPars, myShocks)
                                # my_alpha_mom = alpha_moment_giv_sims(myPars, sims)
                                if(np.abs(my_w0_mu_mom - w0_mu_mom_targ) < w0_mu_tol and np.abs(my_w0_sigma_mom - w0_sigma_mom_targ) < w0_sigma_tol 
                                    and np.abs(my_w1_mom - w1_mom_targ) < w1_tol and np.abs(my_w2_mom - w2_mom_targ) < w2_tol 
                                    and (not do_wH_calib or np.abs(my_wH_mom - wH_mom_targ) < wH_tol) 
                                    and (not do_phi_H_calib or np.abs(my_phi_H_mom - phi_H_mom_targ) < phi_H_tol) 
                                    and np.abs(my_alpha_mom - alpha_mom_targ) < alpha_tol):
                                    # and (not do_phi_H_calib or np.abs(my_phi_H_mom - phi_H_mom_targ) < phi_H_tol)):
                                    print(f"Calibration converged after {i+1} iterations")
                                    if not do_wH_calib:
                                        print("********** wH calibration was skipped **********")
                                    if not do_phi_H_calib:
                                        print("********** phi_H calibration was skipped **********")
                                    print(f"w0_weights = {w0_weights}, w0_mean = {my_w0_mu_mom}, w0_mean_targ = {w0_mu_mom_targ}") 
                                    print(f"w0_sd = {my_w0_sigma_mom}, w0_sd_targ = {w0_sigma_mom_targ}")
                                    print(f"w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_mom}, w1 mom targ = {w1_mom_targ}")
                                    print(f"w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_mom}, w2 mom targ = {w2_mom_targ}")
                                    print(f"wH = {myPars.wH_coeff}, wH moment = {my_wH_mom}, wH mom targ = {wH_mom_targ}")
                                    print(f"alpha = {myPars.alpha}, alpha moment = {my_alpha_mom}, alpha mom targ = {alpha_mom_targ}")
                                    print(f"phi_H = {myPars.phi_H}, phi_H moment = {my_phi_H_mom}, phi_H mom targ = {phi_H_mom_targ}")
                                    return myPars.alpha, myPars.lab_fe_weights, myPars.wage_coeff_grid[1,1], myPars.wage_coeff_grid[1,2], myPars.wH_coeff, myPars.phi_H, state_sols, sims

    # calibration does not converge
    print(f"Calibration did not converge after {myPars.max_calib_iters} iterations")
    if not do_wH_calib:
        print("********** wH calibration was skipped **********")
    if not do_phi_H_calib:
        print("********** phi_H calibration was skipped **********")
    print(f"w0_weights = {w0_weights}, w0_mean = {my_w0_mu_mom}, w0_mean_targ = {w0_mu_mom_targ}") 
    print(f"w0_sd = {my_w0_sigma_mom}, w0_sd_targ = {w0_sigma_mom_targ}")
    print(f"w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_mom}, w1 mom targ = {w1_mom_targ}")
    print(f"w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_mom}, w2 mom targ = {w2_mom_targ}")
    print(f"wH = {myPars.wH_coeff}, wH moment = {my_wH_mom}, wH mom targ = {wH_mom_targ}")
    print(f"alpha = {myPars.alpha}, alpha moment = {my_alpha_mom}, alpha mom targ = {alpha_mom_targ}")
    print(f"phi_H = {myPars.phi_H}, phi_H moment = {my_phi_H_mom}, phi_H mom targ = {phi_H_mom_targ}")
    return myPars.alpha, myPars.lab_fe_weights, myPars.wage_coeff_grid[1,1], myPars.wage_coeff_grid[1,2], myPars.wH_coeff, myPars.phi_H, state_sols, sims
    

if __name__ == "__main__":
    start_time = time.perf_counter()

    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

    # my_lab_fe_grid = np.array([10.0, 15.0, 20.0, 25.0])
    my_lab_fe_grid = np.array([5.0, 10.0, 15.0, 20.0])
    # my_lab_fe_grid = np.array([5.0, 10.0, 15.0])
    my_lab_fe_grid = np.log(my_lab_fe_grid)
    lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
    quad_wage_coeffs = [-0.000, -0.02, -0.02, -0.02] 
    cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

    num_FE_types = len(my_lab_fe_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])

    w_coeff_grid[0, :] = [my_lab_fe_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
    w_coeff_grid[1, :] = [my_lab_fe_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
    w_coeff_grid[2, :] = [my_lab_fe_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
    w_coeff_grid[3, :] = [my_lab_fe_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

    print("intial wage coeff grid")
    print(w_coeff_grid)
    my_lab_fe_weights = tb.gen_even_row_weights(w_coeff_grid)

    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]),
                nu_grid_size=1, alpha = 0.45, sim_draws=1000, lab_fe_grid = my_lab_fe_grid, lab_fe_weights = my_lab_fe_weights,
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 10, sigma_util = 0.9999,
                print_screen=0)


    tb.print_exec_time("Calibration main ran in", start_time)   