"""
calibration.py

This file contains the calibration code for the model.

Author: Ben
Date: 2024-06-17 15:01:32
"""

# Import packages
import numpy as np
import csv

import my_toolbox as tb
import solver
import pars_shocks_and_wages as ps
from pars_shocks_and_wages import Pars, Shocks
import old_simulate as simulate
import plot_lc as plot_lc


def calib_alpha(myPars : Pars, main_path : str) -> float:
    start_alpha = myPars.alpha
    alpha_iters = 100
    lab_tol = 0.1
    lab_targ = 0.40
    # solve model for a given alpha
    shocks = Shocks(myPars)
    state_sols = solver.solve_lc(myPars)
    labor_sols = state_sols['lab']
    print('Mean Labor Sols:', np.mean(labor_sols))
    sim_lc = simulate.sim_lc(myPars, shocks, state_sols)
    plot_lc.plot_lc_profiles(myPars, sim_lc)
    
    labor_sims = sim_lc['c']
    #print(labor_sims)
    # get the mean labor worked across all labor fixed effect groups
    
    mean_lab = np.mean(labor_sims)
    print('Mean Labor Sims:', mean_lab)

    
    # check if model matches mean labor worked for one labor fixed effect group = 40
    # start with the low ability group and a guees for alpha of 0.45
    # write parameters to a file
    # if it is within a certain tolerance then return the alpha
    # if not then adjust alpha and repeat
    pass
def print_endog_params_to_tex(myPars: Pars, main_path: str):
    '''this generates a latex table of the parameters'''
    tab = ["\\begin{tabular}{l l l l} \n"]
    tab.append("\\hline \n")
    tab.append("Parameter & Description & Value & Target \\\\ \n") 
    tab.append("\\hline \n")   
    tab.append(f"$\\alpha$ & Capital share & {np.round(myPars.alpha, 4)} & Mean Hours Worked \\\\ \n") 
    tab.append(f"$\\kappa$ & Borrowing constraint & {np.round(-myPars.a_min, 4)} & Unconstrained \\\\ \n") 
    tab.append("\\hline \n")
    tab.append(f"\\end{{tabular}}")
    fullpath = main_path + 'parameters_endog.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)

def print_wage_coeffs_to_tex(myPars: Pars, main_path: str):
    '''this generates a latex table of the parameters'''
    tab = ["\\begin{tabular}{l l l l l l} \n"]
    tab.append("\\hline \n")
    tab.append(" Parameter & $\\gamma_1$ &  $\\gamma_2$ & $\\gamma_3$ & Description & Source \\\\ \n") 
    tab.append("\\hline \n")   
    tab.append(f"$\\beta_{{0\\gamma}}$ & {myPars.wage_coeff_grid[0][0]} &  {myPars.wage_coeff_grid[1][0]} & {myPars.wage_coeff_grid[2][0]} & Constant & Benchmark \\\\ \n")
    tab.append(f"$\\beta_{{1\\gamma}}$ & {myPars.wage_coeff_grid[0][1]} &  {myPars.wage_coeff_grid[1][1]} & {myPars.wage_coeff_grid[2][1]} & Linear Coeff. & Benchmark \\\\ \n")
    tab.append(f"$\\beta_{{2\\gamma}}$ & {myPars.wage_coeff_grid[0][2]} &  {myPars.wage_coeff_grid[1][2]} & {myPars.wage_coeff_grid[2][2]} & Quadratic Coeff. & Benchmark \\\\ \n")
    tab.append("\\hline \n")
    tab.append(f"\\end{{tabular}}")
    fullpath = main_path + 'wage_coeffs.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)


def print_exog_params_to_tex(myPars: Pars, main_path: str):
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
    fullpath = main_path + 'parameters_exog.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)

def print_params_to_csv(myPars: Pars, main_path: str):
    # store params in a csv 
    # print a table of the calibration results
    my_path = main_path + "parameters.csv"
    with open(my_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])
        for param, value in pars_to_dict(myPars).items():
            writer.writerow([param, value])

def pars_to_dict(pars_instance: Pars) -> dict:
    return {
        'w_determ_cons': pars_instance.w_determ_cons,
        'w_age': pars_instance.w_age,
        'w_age_2': pars_instance.w_age_2,
        'w_age_3': pars_instance.w_age_3,
        'w_avg_good_health': pars_instance.w_avg_good_health,
        'w_avg_good_health_age': pars_instance.w_avg_good_health_age,
        'w_good_health': pars_instance.w_good_health,
        'w_good_health_age': pars_instance.w_good_health_age,
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
#put a run if main function here
if __name__ == "__main__":
        main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/calibration/"
        myPars = Pars(main_path, J=50, a_grid_size=100, a_min= -500.0, a_max = 500.0, 
                    H_grid=np.array([1.0]), nu_grid_size=1, alpha = 0.45, sim_draws=1000,
                    print_screen=3)
        
        #calib_alpha(myPars, main_path)
        print_params_to_csv(myPars, main_path)
        print_exog_params_to_tex(myPars, main_path)
        print_endog_params_to_tex(myPars, main_path)
        print_wage_coeffs_to_tex(myPars, main_path)
