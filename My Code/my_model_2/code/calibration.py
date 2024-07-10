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

# My code
import my_toolbox as tb
import solver
import model_no_uncert as model
import pars_shocks_and_wages as ps
from pars_shocks_and_wages import Pars, Shocks
import simulate
import plot_lc as plot_lc

def print_endog_params_to_tex(myPars: Pars, path: str = None)-> None:
    '''this generates a latex table of the parameters'''
    tab = ["\\begin{tabular}{l l l l} \n"]
    tab.append("\\hline \n")
    tab.append("Parameter & Description & Value & Target \\\\ \n") 
    tab.append("\\hline \n")   
    tab.append(f"$\\alpha$ & Consumptioni share & {np.round(myPars.alpha, 4)} & Mean Hours Worked \\\\ \n") 
    tab.append(f"$\\kappa$ & Borrowing constraint & {np.round(myPars.a_min, 4)} & Unconstrained \\\\ \n") 
    tab.append("\\hline \n")
    tab.append(f"\\end{{tabular}}")
    if path is None:
        path = myPars.path + 'calibration/'
    fullpath = path + 'parameters_endog.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)

def print_wage_coeffs_to_tex(myPars: Pars, path: str = None)-> None:
    '''this generates a latex table of the parameters'''
    tab = ["\\small\\begin{tabular}{l l l l l l l} \n"]
    tab.append("\\hline \n")
    tab.append(" Parameter & $\\gamma_1$ &  $\\gamma_2$ & $\\gamma_3$ & $\\gamma_4$ & Description & Source \\\\ \n") 
    tab.append("\\hline \n")   
    # for i in myPars.lab_FE_grid:
    #     tab.append(f"$\\beta_{{{i}\\gamma}}$ & {myPars.wage_coeff_grid[0][i]} &  {myPars.wage_coeff_grid[1][i]} & {myPars.wage_coeff_grid[2][i]} & $j^{{{i}}}$ Coeff. & Moment Matched \\\\ \n") 
    tab.append(f"""$w_{{0\\gamma}}$ & {round(myPars.wage_coeff_grid[0][0], 3)} & {round(myPars.wage_coeff_grid[1][0], 3)} 
               & {round(myPars.wage_coeff_grid[2][0], 3)} & {round(myPars.wage_coeff_grid[3][0])} 
               & Constant & Benchmark \\\\ \n""")
    tab.append(f"""$w_{{1\\gamma}}$ & {round(myPars.wage_coeff_grid[0][1], 3)} &  {round(myPars.wage_coeff_grid[1][1], 3)} 
               & {round(myPars.wage_coeff_grid[2][1], 3)} & {round(myPars.wage_coeff_grid[3][1], 3)} 
               & $j$ Coeff. & Wage Growth \\\\ \n""")
    tab.append(f"""$w_{{2\\gamma}}$ & {round(myPars.wage_coeff_grid[0][2], 3)} &  {round(myPars.wage_coeff_grid[1][2], 3)} 
               & {round(myPars.wage_coeff_grid[2][2], 3)} & {round(myPars.wage_coeff_grid[3][2], 3)} 
               & $j^{{2}}$ Coeff. & Wage Decline \\\\ \n""")
    tab.append("\\hline \n")
    tab.append(f"\\end{{tabular}}")
    if path is None:
        path = myPars.path + 'calibration/'
    fullpath = path + 'wage_coeffs.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)


def print_exog_params_to_tex(myPars: Pars, path: str = None)-> None:
    '''this generates a latex table of the parameters'''
    tab = ["\\begin{tabular}{l l l l} \n"]
    tab.append("\\hline \n")
    tab.append("Parameter & Description & Value & Source \\\\ \n") 
    tab.append("\\hline \n")
    tab.append(f"$R$ & Gross interest rate  & {np.round(1 + myPars.r, 4)} & Benchmark \\\\ \n")
    tab.append(f"$\\beta$ & Patience & {np.round(myPars.beta, 4)} & $1/R$ \\\\ \n")
    tab.append(f"$\\sigma$ & CRRA & {np.round(myPars.sigma_util, 4)} & Benchmark \\\\ \n")
    tab.append(f"$\\phi_n$ & Labor time-cost & {np.round(myPars.phi_n, 4)} & Benchmark \\\\ \n")
    tab.append(f"$\\phi_H$ & Health time-cost & {np.round(myPars.phi_H, 4)} & Benchmark \\\\ \n") 
    tab.append("\\hline \n")
    tab.append(f"\\end{{tabular}}")
    if path is None:
        path = myPars.path + 'calibration/'
    fullpath = path + 'parameters_exog.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)

def print_params_to_csv(myPars: Pars, path: str = None, file_name: str = "parameters.csv")-> None:
    # store params in a csv 
    # print a table of the calibration results
    if path is None:
        path = myPars.path
    my_path = path + file_name
    with open(my_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        for param, value in pars_to_dict(myPars).items():
            writer.writerow([param, value])

def pars_to_dict(pars_instance: Pars) -> Dict:
    return {
        # 'w_determ_cons': pars_instance.w_determ_cons,
        # 'w_age': pars_instance.w_age,
        # 'w_age_2': pars_instance.w_age_2,
        # 'w_age_3': pars_instance.w_age_3,
        # 'w_avg_good_health': pars_instance.w_avg_good_health,
        # 'w_avg_good_health_age': pars_instance.w_avg_good_health_age,
        # 'w_good_health': pars_instance.w_good_health,
        # 'w_good_health_age': pars_instance.w_good_health_age,
        'rho_nu': pars_instance.rho_nu,
        'sigma_eps_2': pars_instance.sigma_eps_2,
        'sigma_nu0_2': pars_instance.sigma_nu0_2,
        'nu_grid': pars_instance.nu_grid,
        'nu_grid_size': pars_instance.nu_grid_size,
        'nu_trans': pars_instance.nu_trans,
        'sigma_gamma_2': pars_instance.sigma_gamma_2,
        'lab_FE_grid': pars_instance.lab_FE_grid,
        'lab_FE_grid_size': pars_instance.lab_FE_grid_size,
        'beta': pars_instance.beta,
        'alpha': pars_instance.alpha,
        'sigma_util': pars_instance.sigma_util,
        'phi_n': pars_instance.phi_n,
        'phi_H': pars_instance.phi_H,
        'B2B': pars_instance.B2B,
        'G2G': pars_instance.G2G,
        'r': pars_instance.r,
        'a_min': pars_instance.a_min,
        'a_max': pars_instance.a_max,
        'a_grid_growth': pars_instance.a_grid_growth,
        'a_grid': pars_instance.a_grid,
        'a_grid_size': pars_instance.a_grid_size,
        'H_grid': pars_instance.H_grid,
        'H_grid_size': pars_instance.H_grid_size,
        'state_space_shape': pars_instance.state_space_shape,
        'state_space_shape_no_j': pars_instance.state_space_shape_no_j,
        'state_space_no_j_size': pars_instance.state_space_no_j_size,
        'state_space_shape_sims': pars_instance.state_space_shape_sims,
        'lab_min': pars_instance.lab_min,
        'lab_max': pars_instance.lab_max,
        'c_min': pars_instance.c_min,
        'leis_min': pars_instance.leis_min,
        'leis_max': pars_instance.leis_max,
        'dt': pars_instance.dt,
        'sim_draws': pars_instance.sim_draws,
        'J': pars_instance.J,
        'print_screen': pars_instance.print_screen,
        'interp_c_prime_grid': pars_instance.interp_c_prime_grid,
        'interp_eval_points': pars_instance.interp_eval_points,
        'H_by_nu_flat_trans': pars_instance.H_by_nu_flat_trans,
        'H_by_nu_size': pars_instance.H_by_nu_size,
        'sim_interp_grid_spec': pars_instance.sim_interp_grid_spec,
        'start_age': pars_instance.start_age,
        'end_age': pars_instance.end_age,
        'age_grid': pars_instance.age_grid,
        'path': pars_instance.path,
        'wage_coeff_grid': pars_instance.wage_coeff_grid
    }

def calib_alpha(myPars: Pars, main_path: str, max_iters: int, lab_tol: float, mean_lab_targ: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    print_params_to_csv(myPars, path = main_path, file_name = "pre_alpha_calib_params.csv")
    mean_lab = -999.999
    state_sols = {}
    sim_lc = {}
    alpha_min = 0.00001 #alpha of 0 is not economically meaningful
    alpha_max = 1.0
    
    # define the lambda function to find the zero of     
    get_mean_lab_diff = lambda new_alpha: alpha_moment_giv_alpha(myPars, main_path, new_alpha)[0] - mean_lab_targ 
    # search for the alpha that is the zero of the lambda function
    calib_alpha = tb.bisection_search(get_mean_lab_diff, alpha_min, alpha_max, lab_tol, max_iters) 
    myPars.alpha = calib_alpha # myPars is mutable this also happens inside solve_mean_lab_giv_alpha but i think its more readable here
    
    # solve, simulate and plot model for the calibrated alpha
    mean_lab, state_sols, sim_lc = alpha_moment_giv_alpha(myPars, main_path, calib_alpha)
    print_params_to_csv(myPars, path = main_path, file_name = "alpha_calib_params.csv")
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
    labor_sims = sims['lab'][:,:,:,:,:myPars.J]
    # print("preweights:")
    # print(labor_sims[1, 0, 0, 0, :])
    weighted_labor_sims = labor_sims
    # weighted_labor_sims = model.gen_weighted_sim(myPars, labor_sims)
    #cuz labor is between 0 and 1 this makes it so that weighting by percent doesnt fuck up the scale 
    # print("postweights:")
    # print(weighted_labor_sims[1, 0, 0, 0,: ])
    mean_lab = np.mean(weighted_labor_sims)
    print(f"mean labor worked = {mean_lab}")
    return mean_lab

def calib_w0(myPars: Pars, main_path: str, max_iters: int, mean_tol: float,  mean_target: float, sd_tol: float, sd_target: float):
    print_params_to_csv(myPars, path = main_path, file_name = "pre_w0_calib_params.csv")
    mean_wage = -999.999
    sd_wage = -999.999
    sate_sols = {}
    sim_lc = {}
    # np.linspace(myPars.lab_FE_grid[0], myPars.lab_FE_grid[-1], myPars.lab_FE_grid_size)
    my_lab_FE_grid, my_weights = tb.Taucheniid(sd_target, myPars.lab_FE_grid_size, mean = mean_target, 
                                               state_grid = myPars.lab_FE_grid)
    myPars.lab_FE_grid = my_lab_FE_grid
    # myPars.wage_coeff_grid[0] = my_weights 
    myPars.lab_FE_weights = my_weights
    # solve and simulate model for the calibrated w0
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    mean_wage, sd_wage = w0_moments(myPars)
    print(f"Calibration exited: mean wage = {mean_wage}, target mean wage = {mean_target}, sd wage = {sd_wage}, target sd wage = {sd_target}")
    print(f"Calibrated weights = {my_weights}")
    return my_weights, mean_wage, sd_wage, state_sols, sim_lc 

def w0_moments(myPars: Pars)-> Tuple[float, float]:
    my_weights = myPars.lab_FE_weights
    my_lab_FE = myPars.wage_coeff_grid[:, 0]
    mean_first_per_wage = np.sum(my_weights * my_lab_FE)
    sd_first_per_wage = np.sqrt(np.sum(my_weights * (my_lab_FE - mean_first_per_wage)**2))
    return mean_first_per_wage, sd_first_per_wage

def calib_w1(myPars: Pars, main_path: str, max_iters: int, tol: float, target: float, w1_min: float, w1_max: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    print_params_to_csv(myPars, path = main_path, file_name = "pre_w1_calib_params.csv")
    w1_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w1_diff = lambda new_coeff: w1_moment_giv_w1(myPars, main_path, new_coeff) - target
    calibrated_w1 = tb.bisection_search(get_w1_diff, w1_min, w1_max, tol, max_iters)
    # update the wage coeff grid butleave the first element as is i.e. with no wage growth
    for i in range (1, myPars.lab_FE_grid_size):
        myPars.wage_coeff_grid[i, 1] = calibrated_w1

    # solve, simulate and plot model for the calibrated w1
    w1_moment = w1_moment_giv_w1(myPars, main_path, calibrated_w1) 
    print_params_to_csv(myPars, path = main_path, file_name = "w1_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    # plot_lc.plot_lc_profiles(myPars, sim_lc, main_path)
    print(f"Calibration exited: w1 = {calibrated_w1}, wage growth = {w1_moment}, target wage growth = {target}") 

    return calibrated_w1, w1_moment, state_sols, sim_lc

def w1_moment_giv_w1(myPars: Pars, main_path: str, new_coeff: float)-> float:
    for i in range (1, myPars.lab_FE_grid_size): #skip the first so the comparison group has no wage growth  
        myPars.wage_coeff_grid[i, 1] = new_coeff
    return w1_moment(myPars)

def w1_moment(myPars: Pars)-> float:
    wage_sims = model.gen_weighted_wages(myPars)
    mean_wage = np.sum(wage_sims, axis=tuple(range(wage_sims.ndim - 1)))
    wage_diff = log(np.max(mean_wage)) - log(mean_wage[0])
    return wage_diff

def calib_w2(myPars: Pars, main_path: str, max_iters: int, tol: float, target: float, w2_min: float, w2_max: float)-> Tuple[float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    print_params_to_csv(myPars, path = main_path, file_name = "pre_w2_calib_params.csv")
    w2_moment = -999.999
    sate_sols = {}
    sim_lc = {}
    # define the lambda function to find the zero of
    get_w2_diff = lambda new_coeff: w2_moment_giv_w2(myPars, main_path, new_coeff) - target
    # search for the w2 that is the zero of the lambda function
    calibrated_w2 = tb.bisection_search(get_w2_diff, w2_min, w2_max, tol, max_iters)
    # update the wage coeff grid butleave the first element as is i.e. with no wage growth
    for i in range (1, myPars.lab_FE_grid_size):
        myPars.wage_coeff_grid[i, 2] = calibrated_w2
    
    # solve, simulate and plot model for the calibrated w2
    w2_moment = w2_moment_giv_w2(myPars, main_path, calibrated_w2)
    print_params_to_csv(myPars, path = main_path, file_name = "w2_calib_params.csv")
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars, main_path)
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    # plot_lc.plot_lc_profiles(myPars, sim_lc, main_path)
    print(f"Calibration exited: w2 = {calibrated_w2}, wage growth = {w2_moment}, target wage growth = {target}")

    return calibrated_w2, w2_moment, state_sols, sim_lc

def w2_moment_giv_w2(myPars: Pars, main_path, new_coeff: float)-> float:
    for i in range (1, myPars.lab_FE_grid_size):
        myPars.wage_coeff_grid[i, 2] = new_coeff
    return w2_moment(myPars)

def w2_moment(myPars: Pars)-> float:
    wage_sims = model.gen_weighted_wages(myPars)
    # or could just generate wages independently but this is more general
    # wage_sims = model.gen_wages(myPars)
    mean_wage = np.sum(wage_sims, axis=tuple(range(wage_sims.ndim - 1)))
    wage_diff = log(np.max(mean_wage)) - log(mean_wage[myPars.J-1])
    return wage_diff

def calib_all(myPars: Pars, calib_path: str, max_iters: int, alpha_mom_targ: float,  
        w0_mean_targ: float, w0_sd_targ: float, w1_mom_targ: float, w2_mom_targ: float)-> (
        Tuple[float, np.ndarray, float, float, Dict[str, np.ndarray], Dict[str, np.ndarray]]):

    # set up return arrays
    state_sols = {}
    sims = {}
    # alpha calib set up
    alpha_tol = 0.001

    # w0 calib set up
    w0_mean_tol = 2.0
    w0_sd_tol = 2.0
    # w0_min = 0.0
    # w0_max = 10.0

    # w1 calib set up
    w1_tol = 0.001
    #w1_targ = 0.50
    w1_min = 0.0
    w1_max = 10.0

    #w2 calib set up
    w2_tol = 0.001
    #w2_targ = 0.50
    w2_min = -1.0
    w2_max = 0.0

    for i in range(max_iters):
        print(f"Calibration iteration {i}")
        # calibrate alpha
        alpha_calib, alpha_moment, state_sols, sims = calib_alpha(myPars, calib_path, max_iters, alpha_tol, alpha_mom_targ)
        print(f"alpha = {alpha_calib}, alpha_moment = {alpha_moment}")
        w0_weights, w0_mean_mom, w0_sd_mom, state_sols, sims = calib_w0(myPars, calib_path, max_iters, 
                                                                        w0_mean_tol, w0_mean_targ, w0_sd_tol, w0_sd_targ)
        print(f"w0_weights = {w0_weights}, w0_mean = {w0_mean_mom}, w0_sd = {w0_sd_mom}")
        w1_calib, w1_moment, state_sols, sims = calib_w1(myPars, calib_path, max_iters, w1_tol, w1_mom_targ, w1_min, w1_max)
        print(f"w1 = {w1_calib}, w1_moment = {w1_moment}")
        w2_calib, w2_moment, state_sols, sims = calib_w2(myPars, calib_path, max_iters, w2_tol, w2_mom_targ, w2_min, w2_max)
        print(f"w2 = {w2_calib}, w2_moment = {w2_moment}")
        if is_calib_cond_met(myPars, sims, alpha_mom_targ, alpha_tol, 
                             w0_mean_targ, w0_mean_tol, w0_sd_targ, w0_sd_tol,
                             w1_mom_targ, w1_tol, w2_mom_targ, w2_tol):
            print(f"Calibration converged after {i+1} iterations: alpha = {alpha_calib}, alpha moment = {alpha_moment}, w1 = {w1_calib}, w1 moment = {w1_moment}, w2 = {w2_calib}, w2 moment = {w2_moment}")
            print(f"w0_weights = {w0_weights}, w0_mean = {w0_mean_mom}, w0_sd = {w0_sd_mom}")
            return alpha_calib, w0_weights, w1_calib, w2_calib, state_sols, sims
    print(f"Calibration did not converge after {max_iters} iterations")
    return alpha_calib, w0_weights, w1_calib, w2_calib, state_sols, sims
    
def is_calib_cond_met(myPars: Pars, sims: Dict[str, np.ndarray], alpha_mom_targ: float, alpha_tol: float, 
                      w0_mean_targ: float, w0_mean_tol: float, w0_sd_targ: float, w0_sd_tol: float,
                      w1_mom_targ: float, w1_tol: float, w2_mom_target:float, w2_tol: float)-> bool:
    # get alpha_moment and w1_moment
    alpha_moment = alpha_moment_giv_sims(myPars, sims)
    w0_mean, w0_sd = w0_moments(myPars)
    my_w1_moment = w1_moment(myPars)
    my_w2_moment = w2_moment(myPars)
    # if alpha_moment is within alpha_mom_targ and w1_moment is within w1_mom_targ
    if (abs(alpha_moment - alpha_mom_targ) < alpha_tol and abs(my_w1_moment - w1_mom_targ) < w1_tol 
        and abs(my_w2_moment - w2_mom_target) < w2_tol 
        and abs(w0_mean - w0_mean_targ) < w0_mean_tol and abs(w0_sd - w0_sd_targ) < w0_sd_tol):
        return True
    else:
        return False

if __name__ == "__main__":
        start_time = time.perf_counter()
        calib_path= "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/calibration/"
        main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/"
        
        # my_lab_FE_grid = np.array([10.0, 20.0, 30.0, 40.0])
        my_lab_FE_grid = np.array([10.0, 20.0, 30.0])
        lin_wage_coeffs = [0.0, 1.0, 1.0, 1.0]
        quad_wage_coeffs = [-0.000, -0.030, -0.030, -0.030] 
        cub_wage_coeffs = [0.0, 0.0, 0.0, 0.0]

        num_FE_types = len(my_lab_FE_grid)
        w_coeff_grid = np.zeros([num_FE_types, 4])
        
        w_coeff_grid[0, :] = [my_lab_FE_grid[0], lin_wage_coeffs[0], quad_wage_coeffs[0], cub_wage_coeffs[0]]
        w_coeff_grid[1, :] = [my_lab_FE_grid[1], lin_wage_coeffs[1], quad_wage_coeffs[1], cub_wage_coeffs[1]]
        w_coeff_grid[2, :] = [my_lab_FE_grid[2], lin_wage_coeffs[2], quad_wage_coeffs[2], cub_wage_coeffs[2]]
        #w_coeff_grid[3, :] = [my_lab_FE_grid[3], lin_wage_coeffs[3], quad_wage_coeffs[3], cub_wage_coeffs[3]]

        print("intial wage coeff grid")
        print(w_coeff_grid)

        myPars = Pars(main_path, J=50, a_grid_size=100, a_min= -500.0, a_max = 500.0, lab_FE_grid = my_lab_FE_grid,
                    H_grid=np.array([0.0, 1.0]), nu_grid_size=1, alpha = 0.45, sim_draws=1000,
                    wage_coeff_grid = w_coeff_grid,
                    print_screen=0)
        
        max_iters = 100
        alpha_mom_targ = 0.40
        w0_mean_targ = 20.0
        w0_sd_targ = 5.0
        w1_mom_targ = 0.20
        w2_mom_targ = 0.20

        alpha, w0_weights, w1, w2, state_sols, sims = calib_all(myPars, calib_path, max_iters, alpha_mom_targ, 
                                                                    w0_mean_targ, w0_sd_targ, w1_mom_targ, w2_mom_targ)
        tb.print_exec_time("Calibration main ran in", start_time)