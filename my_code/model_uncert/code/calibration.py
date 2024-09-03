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

def calib_alpha(myPars: Pars, main_path: str, lab_tol: float, mean_lab_targ: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_alpha_calib_params.csv")
    mean_lab = -999.999
    state_sols = {}
    sim_lc = {}
    alpha_min = 0.00001 #alpha of 0 is not economically meaningful
    alpha_max = 1.0
    
    # define the lambda function to find the zero of     
    get_mean_lab_diff = lambda new_alpha: alpha_moment_giv_alpha(myPars, main_path, new_alpha)[0] - mean_lab_targ 
    # search for the alpha that is the zero of the lambda function
    calib_alpha = tb.bisection_search(get_mean_lab_diff, alpha_min, alpha_max, lab_tol, myPars.max_iters, myPars.print_screen) 
    myPars.alpha = calib_alpha # myPars is mutable this also happens inside solve_mean_lab_giv_alpha but i think its more readable here
    
    # solve, simulate and plot model for the calibrated alpha
    mean_lab, state_sols, sim_lc = alpha_moment_giv_alpha(myPars, main_path, calib_alpha)
    io.print_params_to_csv(myPars, path = main_path, file_name = "alpha_calib_params.csv")
    if myPars.print_screen >= 1:
        print(f"Calibration exited: alpha = {calib_alpha}, mean labor worked = {mean_lab}, target mean labor worked = {mean_lab_targ}")
    
    # return the alpha, the resulting mean labor worked, and the target mean labor worked; and the model solutions and simulations
    return calib_alpha, mean_lab, state_sols, sim_lc
    

def alpha_moment_giv_alpha(myPars : Pars, main_path : str, new_alpha: float) ->Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    '''
        this function solves the model for a given alpha and returns the alpha, the mean labor worked, and the target mean labor worked
        and the model solutions and simulations
    ''' 
    myPars.alpha = new_alpha
    # solve, simulate and plot model for a given alpha
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)

    # labor_sims = sim_lc['lab'][:,:,:,:,:myPars.J]
    # mean_lab = np.mean(labor_sims)
    mean_lab = alpha_moment_giv_sims(myPars, sim_lc) 
    return mean_lab, state_sols, sim_lc

def alpha_moment_giv_sims(myPars: Pars, sims: Dict[str, np.ndarray])-> float:
    labor_sims = sims['lab'][:, :, :, :myPars.J]
    weighted_labor_sims = model.gen_weighted_sim(myPars, labor_sims)
    mean_lab_by_age = np.sum(weighted_labor_sims, axis = tuple(range(weighted_labor_sims.ndim-1)))
    mean_lab = np.mean(mean_lab_by_age)
    return mean_lab

def get_alpha_targ(myPars: Pars) -> float:
    data_moments_path = myPars.path + '/input/labor_moments.csv'
    data_mom_col_ind = 1
    mean_labor_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return np.mean(mean_labor_by_age)




def calib_w0(myPars: Pars, main_path: str, mean_target: float, sd_target: float):
    # mean_target = 12.664071
    # sd_target = 3.1353517

    # calib.calib_w0(myPars, main_path, w0_mean_targ, w0_sd_targ)

    myShocks = Shocks(myPars)
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_calib_params.csv")
    mean_wage = -999.999
    sd_wage = -999.999
    sate_sols = {}
    sim_lc = {}

    first_per_wages = model.gen_wage_hist(myPars, myShocks)[:,:,:,0]
    first_per_wages = first_per_wages * (1/myPars.sim_draws)
    first_per_wages = first_per_wages * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis]
    # print("first_per_wages.shape:", first_per_wages.shape)
    # print("mean of first_per_wages",np.mean(first_per_wages))

    collapsed_weighted_wages = np.sum(first_per_wages, axis = tuple(range(1,first_per_wages.ndim)))
    # print("collapsed_weighted_wages.shape", collapsed_weighted_wages.shape)
    # print("collapsed_weighted_wages", collapsed_weighted_wages)

    first_per_wages = model.gen_wage_hist(myPars, myShocks)[:,:,:,0] 
    my_weights = tb.optimize_weights(collapsed_weighted_wages, mean_target, sd_target, first_per_wages, myPars.sim_draws, myPars.H_type_perm_weights)
    weigthed_mean, weighted_std = tb.weighted_stats(my_weights, collapsed_weighted_wages, first_per_wages, myPars.sim_draws, myPars.H_type_perm_weights)
    # print("my_weights", my_weights)
    # print("weigthed_mean", weigthed_mean, "weighted_std", weighted_std)
    # print("mean_targ", mean_target, "sd_targ", sd_target)

    #update the labor fixed effect weights
    myPars.lab_FE_weights = my_weights
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    mean_wage, sd_wage = w0_moments(myPars)
    # print(f"myPars updated lab_FE_weights = {myPars.lab_FE_weights}")
    # print(f"w0_moments_mean_wage = {mean_wage}, w0_moments_sd_wage = {sd_wage}")
    # print("mean_targ", mean_target, "sd_targ", sd_target)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: mean wage = {mean_wage}, target mean wage = {mean_target}, sd wage = {sd_wage}, target sd wage = {sd_target}")
        print(f"Calibrated weights = {my_weights}")
    return my_weights, mean_wage, sd_wage, state_sols, sim_lc 

def w0_moments(myPars: Pars)-> Tuple[float, float]:
    myShocks = Shocks(myPars)

    first_per_wages = model.gen_wage_hist(myPars, myShocks)[:,:,:,0]
    first_per_weighted_wages = model.gen_weighted_wage_hist(myPars, myShocks)[:,:,:,0] 
    mean_first_per_wage = np.sum(first_per_weighted_wages)
    # print(f"mean_first_per_wage = {mean_first_per_wage}")

    # Calculate the weighted variance
    # Calculate the deviations from the weighted mean
    deviations =  first_per_wages - mean_first_per_wage
    # Square the deviations
    squared_deviations = deviations ** 2
    # Apply the simulation weight (scalar)
    weighted_squared_deviations = squared_deviations * (1.0 / myPars.sim_draws)
    # Apply the H_type_weights along the second dimension
    weighted_squared_deviations = weighted_squared_deviations * myPars.H_type_perm_weights[np.newaxis, :, np.newaxis]
    # Apply the lab_FE_weights along the first dimension
    weighted_squared_variance = np.sum(weighted_squared_deviations * myPars.lab_FE_weights[:, np.newaxis, np.newaxis])
    # Calculate the weighted standard deviation
    sd_first_per_wage = np.sqrt(weighted_squared_variance)
    # print(f"sd_first_per_wage = {sd_first_per_wage}")
    
    return mean_first_per_wage, sd_first_per_wage

def get_w0_mean_targ(myPars: Pars)-> float:
    data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_mom_col_ind = 1
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    # return np.mean(mean_wage_by_age)
    return mean_wage_by_age[0]

def get_w0_sd_targ(myPars: Pars)-> float:
    data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_mom_col_ind = 2
    sd_wage_col= tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return sd_wage_col[0]
    # return np.std(mean_wage_by_age)

def calib_w1(myPars: Pars, main_path: str, tol: float, target: float, w1_min: float, w1_max: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w1_calib_params.csv")
    w1_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w1_diff = lambda new_coeff: w1_moment_giv_w1(myPars, main_path, new_coeff) - target
    calibrated_w1 = tb.bisection_search(get_w1_diff, w1_min, w1_max, tol, myPars.max_iters, myPars.print_screen)
    # update the wage coeff grid butleave the first element as is i.e. with no wage growth
    for i in range (1, myPars.lab_FE_grid_size):
        myPars.wage_coeff_grid[i, 1] = calibrated_w1

    # solve, simulate and plot model for the calibrated w1
    w1_moment = w1_moment_giv_w1(myPars,  main_path, calibrated_w1) 
    io.print_params_to_csv(myPars, path = main_path, file_name = "w1_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    # plot_lc.plot_lc_profiles(myPars, sim_lc, main_path)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: w1 = {calibrated_w1}, wage growth = {w1_moment}, target wage growth = {target}") 

    return calibrated_w1, w1_moment, state_sols, sim_lc

def w1_moment_giv_w1(myPars: Pars, main_path: str, new_coeff: float)-> float:
    for i in range (1, myPars.lab_FE_grid_size): #skip the first so the comparison group has no wage growth  
        myPars.wage_coeff_grid[i, 1] = new_coeff
    return w1_moment(myPars)

def w1_moment(myPars: Pars)-> float:
    myShocks = Shocks(myPars)
    wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    mean_wage = np.sum(wage_sims, axis=tuple(range(wage_sims.ndim - 1)))
    wage_diff = log(np.max(mean_wage)) - log(mean_wage[0])
    return wage_diff

def get_w1_targ(myPars: Pars)-> float:
    data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_mom_col_ind = 1
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    # want to get wages before age 60
    age_60_ind = 60 - myPars.start_age
    # print("age_60_ind:", age_60_ind)
    my_max = np.max(mean_wage_by_age[:age_60_ind])
    return log(my_max)- log(mean_wage_by_age[0])

def calib_w2(myPars: Pars, main_path: str, tol: float, target: float, w2_min: float, w2_max: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_w2_calib_params.csv")
    w2_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w2_diff = lambda new_coeff: w2_moment_giv_w2(myPars, main_path, new_coeff) - target
    # search for the w2 that is the zero of the lambda function
    calibrated_w2 = tb.bisection_search(get_w2_diff, w2_min, w2_max, tol, myPars.max_iters, myPars.print_screen)
    # update the wage coeff grid butleave the first element as is i.e. with no wage growth
    for i in range (1, myPars.lab_FE_grid_size):
        myPars.wage_coeff_grid[i, 2] = calibrated_w2
    
    # solve, simulate and plot model for the calibrated w2
    w2_moment = w2_moment_giv_w2(myPars, main_path, calibrated_w2)
    io.print_params_to_csv(myPars, path = main_path, file_name = "w2_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    # plot_lc.plot_lc_profiles(myPars, sim_lc, main_path)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: w2 = {calibrated_w2}, wage growth = {w2_moment}, target wage growth = {target}")

    return calibrated_w2, w2_moment, state_sols, sim_lc

def w2_moment_giv_w2(myPars: Pars, main_path, new_coeff: float)-> float:
    for i in range (1, myPars.lab_FE_grid_size):
        myPars.wage_coeff_grid[i, 2] = new_coeff
    return w2_moment(myPars)

def w2_moment(myPars: Pars)-> float:
    myShocks = Shocks(myPars)
    wage_sims = model.gen_weighted_wage_hist(myPars, myShocks)
    # or could just generate wages independently but this is more general
    # wage_sims = model.gen_wages(myPars)
    mean_wage = np.sum(wage_sims, axis=tuple(range(wage_sims.ndim - 1)))
    wage_diff = log(np.max(mean_wage)) - log(mean_wage[myPars.J-1])
    return wage_diff

def get_w2_targ(myPars: Pars)-> float:
    data_moments_path = myPars.path + '/input/wage_moments.csv'
    data_mom_col_ind = 1
    mean_wage_by_age = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    age_60_ind = 60 - myPars.start_age
    my_max = np.max(mean_wage_by_age[:age_60_ind])
    return log(my_max) - log(mean_wage_by_age[-1])

def calib_wH(myPars: Pars, main_path: str, tol: float, target: float, wH_min: float, wH_max: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    io.print_params_to_csv(myPars, path = main_path, file_name = "pre_wH_calib_params.csv")
    wH_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_wH_diff = lambda new_coeff: wH_moment_giv_wH(myPars, main_path, new_coeff) - target
    # search for the w2 that is the zero of the lambda function
    calibrated_wH = tb.bisection_search(get_wH_diff, wH_min, wH_max, tol, myPars.max_iters, myPars.print_screen)
    myPars.wH_coeff = calibrated_wH
    # solve, simulate and plot model for the calibrated wH
    wH_moment = wH_moment_giv_wH(myPars, main_path, calibrated_wH)
    io.print_params_to_csv(myPars, path = main_path, file_name = "wH_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    if myPars.print_screen >= 1:
        print(f"Calibration exited: wH = {calibrated_wH}, wage premium = {wH_moment}, target wage premium = {target}")
    return calibrated_wH, wH_moment, state_sols, sim_lc


def wH_moment_giv_wH(myPars: Pars, main_path: str, new_coeff: float)-> float:
    myPars.wH_coeff = new_coeff
    return wH_moment(myPars)

def wH_moment(myPars: Pars)-> float:
    wage_sims = model.gen_wages(myPars) 

    # get the mean of the wage sims when the agent is healthy
    healthy_wages = wage_sims[:,1,:][:,0]
    mean_healthy_wage_by_age = np.dot(myPars.lab_FE_weights, healthy_wages)
    mean_healthy_wage = np.mean(mean_healthy_wage_by_age)

    # get the mean of the wage sims when the agent is unhealthy
    unhealthy_wages = wage_sims[:,0,:][:,0]
    mean_unhealthy_wage_by_age = np.dot(myPars.lab_FE_weights, unhealthy_wages)
    mean_unhealthy_wage = np.mean(mean_unhealthy_wage_by_age)

    wage_diff = log(mean_healthy_wage) - log(mean_unhealthy_wage)
    # print(f"mean_healthy_wage = {mean_healthy_wage}, mean_unhealthy_wage = {mean_unhealthy_wage}, wage_diff = {wage_diff}")
    return wage_diff



def get_wH_targ(myPars: Pars)-> float:
    data_moments_path = myPars.path + '/input/MH_wage_moments.csv'
    data_mom_col_ind = 0
    mean_wage_diff = tb.read_specific_column_from_csv(data_moments_path, data_mom_col_ind)
    return mean_wage_diff[0]

def calib_all(myPars: Pars, calib_path: str, alpha_mom_targ: float,  w0_mean_targ: float, w0_sd_targ: float, w1_mom_targ: float, w2_mom_targ: float, wH_mom_targ: float,
        w1_min:float = 0.0, w1_max: float = 10.0, w2_min = -1.0, w2_max = 0.0, wH_min = -5.0, wH_max = 5.0, wH_tol: float = 0.001,
        alpha_tol: float = 0.001, w0_mom_tol: float = 0.001, w1_tol: float = 0.001, w2_tol: float = 0.001)-> (
        Tuple[float, np.ndarray, float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]):

    # set up return arrays
    state_sols = {}
    sims = {}

    my_alpha_moment = -999.999
    my_w0_mean_mom = -999.999
    my_w0_sd_mom = -999.999
    my_w1_moment = -999.999
    my_w2_moment = -999.999
    my_wH_moment = -999.999

    for i in range(myPars.max_calib_iters):
        print(f"Calibration iteration {i}")
        print("***Calibrating w0***")
        w0_weights, my_w0_mean_mom, my_w0_sd_mom, state_sols, sims = calib_w0(myPars, calib_path, w0_mean_targ, w0_sd_targ)
        print("***Checking calibration of w0***")
        if (np.abs(my_w0_mean_mom - w0_mean_targ) + np.abs(my_w0_sd_mom - w0_sd_targ) < w0_mom_tol):
            print("***w0 calibrated***")
            print("***Calibrating w1***")
            w1_calib, my_w1_moment, state_sols, sims = calib_w1(myPars, calib_path, w1_tol, w1_mom_targ, w1_min, w1_max)
            my_w0_mean_mom, my_w0_sd_mom = w0_moments(myPars)
            print("***Checking calibration of w1***")
            if (np.abs(my_w0_mean_mom - w0_mean_targ) + np.abs(my_w0_sd_mom - w0_sd_targ) < w0_mom_tol 
                and np.abs(my_w1_moment - w1_mom_targ) < w1_tol):
                print("***w1 calibrated***")
                print("***Calibrating w2***")
                w2_calib, my_w2_moment, state_sols, sims = calib_w2(myPars, calib_path, w2_tol, w2_mom_targ, w2_min, w2_max)
                my_w0_mean_mom, my_w0_sd_mom = w0_moments(myPars)
                my_w1_moment = w1_moment(myPars)
                print("***Checking calibration of w2***")
                if (np.abs(my_w0_mean_mom - w0_mean_targ) + np.abs(my_w0_sd_mom - w0_sd_targ) < w0_mom_tol 
                    and np.abs(my_w1_moment - w1_mom_targ) < w1_tol
                    and np.abs(my_w2_moment - w2_mom_targ) < w2_tol):
                    print("***w2 calibrated***")
                    print("***Calibrating wH***")
                    wH_calib, my_wH_moment, state_sols, sims = calib_wH(myPars, calib_path, wH_tol, wH_mom_targ, wH_min, wH_max)                        
                    my_w0_mean_mom, my_w0_sd_mom = w0_moments(myPars)
                    my_w0_mean_mom, my_w0_sd_mom = w0_moments(myPars)
                    my_w1_moment = w1_moment(myPars)
                    my_w2_moment = w2_moment(myPars)
                    print("***Checking calibration of wH***")
                    if (np.abs(my_w0_mean_mom - w0_mean_targ) + np.abs(my_w0_sd_mom - w0_sd_targ) < w0_mom_tol 
                        and np.abs(my_w1_moment - w1_mom_targ) < w1_tol and np.abs(my_w2_moment - w2_mom_targ) < w2_tol
                        and np.abs(my_wH_moment - wH_mom_targ) < wH_tol):
                        print("***wH calibrated***")
                        print("***Calibrating alpha***")
                        alpha_calib, my_alpha_moment, state_sols, sims = calib_alpha(myPars, calib_path, alpha_tol, alpha_mom_targ)
                        my_w0_mean_mom, my_w0_sd_mom = w0_moments(myPars)
                        my_w1_moment = w1_moment(myPars)
                        my_w2_moment = w2_moment(myPars)
                        my_wH_moment = wH_moment(myPars)
                        print("***Checking calibration of alpha***")
                        if(np.abs(my_w0_mean_mom - w0_mean_targ) + np.abs(my_w0_sd_mom - w0_sd_targ) < w0_mom_tol 
                            and np.abs(my_w1_moment - w1_mom_targ) < w1_tol and np.abs(my_w2_moment - w2_mom_targ) < w2_tol 
                            and np.abs(my_wH_moment - wH_mom_targ) < wH_tol 
                            and np.abs(my_alpha_moment - alpha_mom_targ) < alpha_tol):
                            # calibration converges
                            print("***alpha calibrated***")
                            print(f"Calibration converged after {i+1} iterations")
                            print(f"w0_weights = {w0_weights}, w0_mean = {my_w0_mean_mom}, w0_mean_targ = {w0_mean_targ}, w0_sd = {my_w0_sd_mom}, w0_sd_targ = {w0_sd_targ}")
                            print(f""" w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_moment}, w1 mom targ = {w1_mom_targ},
                                w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_moment}, w2 mom targ = {w2_mom_targ},
                                wH = {myPars.wH_coeff}, wH moment = {my_wH_moment}, wH mom targ = {wH_mom_targ},""")
                            print( f"alpha = {myPars.alpha}, alpha moment = {my_alpha_moment}, alpha mom targ = {alpha_mom_targ}")
                            return myPars.alpha, myPars.lab_FE_weights, myPars.wage_coeff_grid[1,1], myPars.wage_coeff_grid[1,2], myPars.wH_coeff, state_sols, sims

    # calibration does not converge
    print(f"Calibration did not converge after {myPars.max_calib_iters} iterations")
    print(f"w0_weights = {w0_weights}, w0_mean = {my_w0_mean_mom}, w0_mean_targ = {w0_mean_targ}, w0_sd = {my_w0_sd_mom}, w0_sd_targ = {w0_sd_targ}")
    print(f""" w1 = {myPars.wage_coeff_grid[1,1]}, w1 moment = {my_w1_moment}, w1 mom targ = {w1_mom_targ},
        w2 = {myPars.wage_coeff_grid[1,2]}, w2 moment = {my_w2_moment}, w2 mom targ = {w2_mom_targ},
        wH = {myPars.wH_coeff}, wH moment = {my_wH_moment}, wH mom targ = {wH_mom_targ},""")
    print( f"alpha = {myPars.alpha}, alpha moment = {my_alpha_moment}, alpha mom targ = {alpha_mom_targ}")
    return myPars.alpha, myPars.lab_FE_weights, myPars.wage_coeff_grid[1,1], myPars.wage_coeff_grid[1,2], myPars.wH_coeff, state_sols, sims
    

if __name__ == "__main__":
    start_time = time.perf_counter()

    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

    # my_lab_FE_grid = np.array([10.0, 15.0, 20.0, 25.0])
    my_lab_FE_grid = np.array([5.0, 10.0, 15.0, 20.0])
    # my_lab_FE_grid = np.array([5.0, 10.0, 15.0])
    my_lab_FE_grid = np.log(my_lab_FE_grid)
    lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
    quad_wage_coeffs = [-0.000, -0.02, -0.02, -0.02] 
    cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

    num_FE_types = len(my_lab_FE_grid)
    w_coeff_grid = np.zeros([num_FE_types, 4])


    w_coeff_grid[0, :] = [my_lab_FE_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
    w_coeff_grid[1, :] = [my_lab_FE_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
    w_coeff_grid[2, :] = [my_lab_FE_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
    w_coeff_grid[3, :] = [my_lab_FE_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

    print("intial wage coeff grid")
    print(w_coeff_grid)

    my_lab_FE_weights = tb.gen_even_weights(w_coeff_grid)

    myPars = Pars(main_path, J=51, a_grid_size=501, a_min= -100.0, a_max = 100.0, H_grid=np.array([0.0, 1.0]), H_weights=np.array([0.5, 0.5]),
                nu_grid_size=1, alpha = 0.45, sim_draws=1000, lab_FE_grid = my_lab_FE_grid, lab_FE_weights = my_lab_FE_weights,
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 10, sigma_util = 0.9999,
                print_screen=0)

    calib_w0(myPars, main_path, 12.664071, 3.1353517) 

    tb.print_exec_time("Calibration main ran in", start_time)   