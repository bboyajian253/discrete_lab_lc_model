"""
plot_lc.py

Created on 2024-05-29 11:30:48
Created by Ben Boyaian

plots life-cycle simulations of calibrated model 

"""
from pars_shocks_and_wages import Pars
import numpy as np
import matplotlib.pyplot as plt
import my_toolbox
import csv

def plot_lc_profiles(myPars : Pars, sim_lc, path) :
    
    #define path
    #Generate variable labels lists
    var_lables = ['Assets', 'Labor', 'Consumption']
    #Generate the short names for the variables i think this should be their keys in the sim_lc dictionary
    var_names_short = ['a', 'lab', 'c']
    #for each variable in the variable list zip the short names and label names together and loop through them
    for label, short_name in zip(var_lables, var_names_short):
        #initialize the life-cycle shells
        #this will change as i introduce retirement
        # some lasts depend on or are equal to the retirment age 
        j_last = myPars.J

        age = myPars.age_grid[:j_last]

        #prep the sim variable for estimation
        values = sim_lc[short_name][:, :, :, :, :j_last] # for each var get the the array of choices for the last age
        log_values = np.log(np.where(values > 0, values, 1e-3)) # log these results replace negatives with a very small number

        #Plot life-cycle profiles
        for modifier in ['', 'log']:
            print(modifier, label)
            fig, ax = plt.subplots()

            if modifier == 'log':
                lc = log_values
            else:
                lc = values

            #iterate through labor fixed effect groups (this is basically ability groups)
            for lab_fe_ind in range(myPars.lab_FE_grid_size):    
                #get the mean of the values over the labor fixed effect 
                lc_mean = np.average(lc[lab_fe_ind, :, :, :], axis=1)
                ax.plot(age, lc_mean, label=myPars.lab_FE_grid[lab_fe_ind])
            
            #specify axes and legend
            ax.set_xlabel('Age')
            ax.set_xlim([age[0] - 2, age[-1] + 2]) #set the x axis limits
            ax.set_ylabel(modifier + ' ' + label)
            ax.legend()

            #save the figure
            fullpath = path + f'fig_lc_{short_name}_{modifier}.png'
            fig.savefig(fullpath, bbox_inches='tight')
            plt.close()
            with open(path + f'fig_lc_{short_name}_{modifier}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['age'] + list(age))
                writer.writerows(['model'] + list(lc))




    pass