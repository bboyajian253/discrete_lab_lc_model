"""
My Model 2 - main file

Author: Ben Boyajian
Date: 2024-05-31 11:38:38
"""
# Import packages
# General
import time
import numpy as np
# My code
import pars_shocks as ps
from pars_shocks import Pars, Shocks
import model_uncert as model
import my_toolbox as tb
import solver
import simulate as simulate
import plot_lc as plot_lc
import run 
import io_manager as io

def main_1():
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

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
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 100, sigma_util = 0.9999,
                print_screen=0)
    myShocks = Shocks(myPars)

    trans0 = myPars.H_trans[0, :, :, :]
    print(f"main_1 H_trans0: {trans0}")

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, 
                          get_moments = True, output_flag = True, tex = True)
    


def main_io( H_trans_ind: int = 0, out_folder_name: str = None, H_trans_path: str = None):
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"
    if out_folder_name is not None:
        print(f"*****Running main_io with out_folder_name = {out_folder_name}*****")
    else:
        print(f"*****Running main_io with H_trans_ind ={H_trans_ind}*****")

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
                wage_coeff_grid = w_coeff_grid, max_iters = 100, max_calib_iters = 100, sigma_util = 0.9999,
                print_screen=0)
    

    out_path = None
    if out_folder_name is not None:
        out_path = myPars.path + out_folder_name + '/'
    if H_trans_path is not None:
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars, path = H_trans_path)
    elif H_trans_ind == 0:
        dummy_path = myPars.path + "input/MH_trans_no_uncert.csv"
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars, path = dummy_path)
        out_path = myPars.path + 'output_H_trans_no_uncert/'
    elif H_trans_ind == 1:
        myPars.H_trans = io.read_and_shape_H_trans_uncond(myPars)
        out_path = myPars.path + 'output_H_trans_uncond/'
    elif H_trans_ind == 2:
        myPars.H_trans = io.read_and_shape_H_trans_H_type(myPars)
        out_path = myPars.path + 'output_H_trans_H_type/'
    elif H_trans_ind == 3:
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars) 
        out_path = myPars.path + 'output_H_trans_full/'
    elif H_trans_ind == 4: # START HERE FOR 50th PERCENTILE
        trans_path = main_path + "input/50_50/MH_trans_uncond.csv"
        of_name = "50th_output_H_trans_uncond"
        myPars.H_trans = io.read_and_shape_H_trans_uncond(myPars, path = trans_path)
        out_path = myPars.path + of_name +'/'
    elif H_trans_ind == 5:
        trans_path = main_path + "input/50_50/MH_trans_H_type.csv"
        of_name = "50th_output_H_trans_H_type"
        myPars.H_trans = io.read_and_shape_H_trans_H_type(myPars)
        out_path = myPars.path + of_name +'/'
    elif H_trans_ind == 6:
        trans_path = main_path + "input/50_50/MH_trans_H_type.csv"
        of_name = "50th_output_H_trans_full"
        myPars.H_trans = io.read_and_shape_H_trans_full(myPars) 
        out_path = myPars.path + of_name +'/'
    
    print(f"Age {myPars.age_grid[0]} health transitions:")
    print(myPars.H_trans[:,0,:,:])
    myShocks = Shocks(myPars)

    sols, sims =run.run_model(myPars, myShocks, solve = True, calib = True, sim_no_calib = False, 
                          get_moments = True, output_flag = True, tex = True, output_path = out_path)

def main_io_k_means(main_path: str = None)-> None:
    if main_path is None:
        main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

    trans_path = main_path + "input/MH_trans_no_uncert.csv"
    of_name = "output_no_uncert"
    main_io(out_folder_name = of_name, H_trans_path = trans_path)

    trans_path = main_path + "input/MH_trans_low_uncert.csv"
    of_name = "output_low_uncert"
    main_io(out_folder_name = of_name, H_trans_path = trans_path)

    trans_path = main_path + "input/MH_trans_mod_uncert.csv"
    of_name = "output_mod_uncert"
    main_io(out_folder_name = of_name, H_trans_path = trans_path)

    trans_path = main_path + "input/MH_trans_high_uncert.csv"
    of_name = "output_high_uncert"
    main_io(out_folder_name = of_name, H_trans_path = trans_path)

    trans_path = main_path + "input/MH_trans_iid_mod_uncert.csv"
    of_name = "output_iid_mod_uncert"
    main_io(out_folder_name = of_name, H_trans_path = trans_path)

    for trans_ind in range(4):
        # print(f"*****Running main_io with trans_ind = {trans_ind}*****")
        main_io(H_trans_ind = trans_ind)

def main_io_50_perct(main_path: str = None)-> None:
    if main_path is None:
        main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

    # trans_path = main_path + "input/50_50/MH_trans_no_uncert.csv"
    # of_name = "50th_output_no_uncert"
    # main_io(out_folder_name = of_name, H_trans_path = trans_path)

    # trans_path = main_path + "input/50_50/MH_trans_low_uncert.csv"
    # of_name = "50th_output_low_uncert"
    # main_io(out_folder_name = of_name, H_trans_path = trans_path)

    # trans_path = main_path + "input/50_50/MH_trans_mod_uncert.csv"
    # of_name = "50th_output_mod_uncert"
    # main_io(out_folder_name = of_name, H_trans_path = trans_path)

    # trans_path = main_path + "input/50_50/MH_trans_high_uncert.csv"
    # of_name = "50th_output_high_uncert"
    # main_io(out_folder_name = of_name, H_trans_path = trans_path)

    # trans_path = main_path + "input/50_50/MH_trans_iid_mod_uncert.csv"
    # of_name = "50th_output_iid_mod_uncert"
    # main_io(out_folder_name = of_name, H_trans_path = trans_path)

    # loop from 4 to 6
    for trans_ind in range(4, 7):
        main_io(H_trans_ind = trans_ind)


#run stuff here
start_time = time.perf_counter()
print("Running main")
# main_io_k_means()
main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/"

trans_path = main_path + "input/MH_trans_mod_uncert.csv"
# of_name = "output"
of_name = None
main_io(out_folder_name = of_name, H_trans_path = trans_path)
# main_io_50_perct(main_path)
# main_io_k_means(main_path)

tb.print_exec_time("Main.py executed in", start_time) 