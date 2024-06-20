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
import my_toolbox
from typing import List, Dict

# My code
from pars_shocks_and_wages import Pars
import csv

def plot_lc_profiles(myPars : Pars, sim_lc: Dict[str, np.ndarray], path: str = None)-> None:
    
    #define path
    if path is None:
        path = myPars.path
    #Generate variable labels lists
    var_lables = ['Consumption', 'Labor', 'Assets']
    #Generate the short names for the variables i think this should be their keys in the sim_lc dictionary
    var_names_short = ['c', 'lab', 'a']
    #for each variable in the variable list zip the short names and label names together and loop through them
    for label, short_name in zip(var_lables, var_names_short):
        #initialize the life-cycle shells
        #this will change as i introduce retirement
        # some lasts depend on or are equal to the retirment age 
        j_last = myPars.J

        age = myPars.age_grid[:j_last]

        #prep the sim variable for estimation
        values = sim_lc[short_name][:, :, :, :, :j_last] # for each var get the the array of choices until the last age
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

            #iterate through labor fixed effect groups (this is basically ability groups)
            for lab_fe_ind in range(myPars.lab_FE_grid_size):    
                #get the mean of the values over the labor fixed effect 
                lc_mean = np.average(lc[lab_fe_ind, 0, 0, :], axis=(0))
                ax.plot(age, lc_mean, label=myPars.lab_FE_grid[lab_fe_ind])
            
            #specify axes and legend
            ax.set_xlabel('Age')
            ax.set_xlim([age[0] - 2, age[-1] + 2]) #set the x axis limits
            ax.set_ylabel(modifier + ' ' + label)
            if short_name == 'lab' and modifier != 'log':
                ax.set_ylim([0, 1])
            # elif short_name == 'a' and modifier != 'log':
            #     ax.set_ylim([myPars.a_min - 2, myPars.a_max + 2])

            ax.legend()

            #save the figure
            fullpath = path + f'fig_lc_{short_name}_{modifier}.pdf'
            fig.savefig(fullpath, bbox_inches='tight')
            plt.close()

            #save the data
            fullpath =  path + f'fig_lc_{short_name}_{modifier}.csv'
            with open(fullpath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['age'] + list(age))
                for row in lc:
                    writer.writerows(['model'] + list(lc))

def plot_c_by_a(myPars : Pars, sim_lc):
    pass
    # fig, ax = plt.subplots()
    # for lab_FE_ind in range(myPars.lab_FE_grid_size):
    #     c_mean = np.average(sim_lc['c'][lab_FE_ind, 0, 0, :, :], axis=(0))
    #     ax.plot(myPars.a_grid, label = f'Age: {age}')

    
    
    
    
        
  


if __name__ == "__main__":
    pass