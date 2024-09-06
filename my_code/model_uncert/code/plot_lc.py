"""
plot_lc.py

Created on 2024-05-29 11:30:48
Created by Ben Boyaian

plots life-cycle simulations of calibrated model 

"""
# import packages
# General
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import csv
# My code
from pars_shocks import Pars


def plot_lc_profiles(myPars : Pars, sim_lc: Dict[str, np.ndarray], path: str = None)-> None:
    if path is None:
        path = myPars.path + 'output/'
    var_lables = ['Consumption', 'Labor', 'Assets', 'Wage', 'Labor Earnings']
    var_names_short = ['c', 'lab', 'a', 'wage', 'lab_earnings'] # these are the keys in the sim_lc dictionary
    for label, short_name in zip(var_lables, var_names_short):
        j_last = myPars.J # will change once i introduce retirement/death
        age_grid = myPars.age_grid[:j_last]
        if short_name == 'a':
            values = sim_lc[short_name][:, :, :, :j_last+1]
        else:
            values = sim_lc[short_name][:, :, :, :j_last] # for each var get the the array of choices until the last age
        log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number

        #Plot life-cycle profiles
        for modifier in ['', 'log']:
            if myPars.print_screen >= 2:
                print(modifier, label)
            fig, ax = plt.subplots()
            if modifier == 'log':
                lc = log_values
            else:
                lc = values
            fullpath =  path + f'fig_lc_{short_name}_{modifier}.csv'
            with open(fullpath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['age'] + list(age_grid))

            #iterate through labor fixed effect groups (this is basically ability groups)
            for H_type_perm_ind in range(myPars.H_type_perm_grid_size):
                for lab_fe_ind in range(myPars.lab_fe_grid_size):    
                    lc_mean = np.mean(lc[lab_fe_ind, H_type_perm_ind, :], axis=0)
                    myLab = f"FE:{round(np.exp(myPars.lab_fe_grid[lab_fe_ind]))} u_H:{round(myPars.H_type_perm_grid[H_type_perm_ind])}"
                    if short_name == 'a':
                        a_age_grid = np.append(age_grid, age_grid[-1] + 1)
                        ax.plot(a_age_grid, lc_mean, label=myLab) 
                    else:
                        ax.plot(age_grid, lc_mean, label=myLab)
                    #save the data
                    with open(fullpath, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([myLab] + list(lc_mean))
            
            ax.set_xlabel('Age')
            ax.set_xlim([age_grid[0] - 2, age_grid[-1] + 2]) #set the x axis limits
            ax.set_ylabel(modifier + ' ' + label)
            ax.legend()
            fullpath = path + f'fig_lc_{short_name}_{modifier}.pdf'
            fig.savefig(fullpath, bbox_inches='tight')
            plt.close()

# probably should plot policy functions as a function of states evetually
def plot_c_by_a(myPars : Pars, sim_lc):
    pass

if __name__ == "__main__":
    pass