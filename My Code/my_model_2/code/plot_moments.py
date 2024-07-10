"""
plot_moments.py

Created on 2024-07-09 01:11:05
Created by Ben Boyaian

plots simulated moments and matched moments together to compare fit 

"""
#import stuff
#General
import numpy as np
import matplotlib.pyplot as plt
from typing import  List, Dict
import csv
import time

# My code
from pars_shocks_and_wages import Pars
import model_no_uncert as model
import my_toolbox as tb

def plot_wage_aggs_and_moms(myPars: Pars, path: str = None)-> None:
    if path == None:
        path = myPars.path + 'output/'
    # calcualte weighted mean wages by age
    weighted_wages = model.gen_weighted_wages(myPars) 
    mean_weighted_wages = np.sum(weighted_wages, axis=tuple(range(weighted_wages.ndim - 1)))
    
    # plot that shit
    j_last = myPars.J
    age_grid = myPars.age_grid[:j_last]
    values = mean_weighted_wages
    sim_y_label = "Average Wage (Weighted)"
    sim_key_label = "Simulated"
    data_moments_label = 'From the data'
    log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number
    for modifier in ['','log']:
        if myPars.print_screen >= 2:
                print(modifier,sim_y_label)
        if modifier == 'log':
             sim_values = log_values
        else:
             sim_values = values
        fig, ax = plt.subplots()
        ax.plot(age_grid, sim_values, label = sim_key_label)
        # get and plot data moments
        data_moments_path = myPars.path + '/input/wage_test.csv'
        data_moments_col_ind = 1
        data_moments = tb.read_specific_column_from_csv(data_moments_path, data_moments_col_ind) # 1 means read the second column
        ax.plot(age_grid, data_moments, label = data_moments_label)
        # specify axis and labels
        ax.set_xlabel('Age')
        ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2]) 
        ax.set_ylabel(modifier + ' ' + sim_y_label)
        ax.legend()

        short_name = 'wage'
        
        #save the figure
        fullpath = path + f'fig_fit_{short_name}_{modifier}.pdf'
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close()

        #save the data
        fullpath =  path + f'fig_fit_{short_name}_{modifier}.csv'
        with open(fullpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['age'] + list(age_grid))
            writer.writerow(['model'] + list(sim_values))


    # read in average wage moments by age

    
    # plot that shit

#run if main func
if __name__ == "__main__":
    start_time = time.perf_counter()
    calib_path= "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/output/calibration/"
    main_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/Main_Git_Clone/Model/My Code/my_model_2/"
    
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
                print_screen=3)
    
    sim_lc = np.ones(1)
    plot_wage_aggs_and_moms(myPars)
    