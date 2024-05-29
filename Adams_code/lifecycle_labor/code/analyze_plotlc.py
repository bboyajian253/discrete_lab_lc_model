# lifecycle_labor: analyze_plotlc
# plots life-cycle simulations of calibrated model

import csv
import numpy as np
import matplotlib.pyplot as plt
import toolbox

def plot_lcprofiles(simlc, par, par2):
    """
    Output life-cycle simulations.
    """

    path = par2.path
    # Generate variable lists
    vlabels = ['Consumption', 'Capital', 'Hours', 'HC', 'Earnings', 'Earn per Hour',
               'Effective HC', 'True Hours', 'True Hourly Wages', 'True Earnings']
    vlabels_short = ['c', 'k', 'n', 'h', 'e', 'eph', 'heff', 'n_tru', 'eph_tru', 'e_tru']

    # iterate through variable list
    for vlbl, vlbls in zip(vlabels, vlabels_short):

        # Initialize life-cycle shells
        if vlbls == 'k' or vlbls == 'c':
            jj = par.J
        else:
            jj = par.JR
        age = par2.age[:jj]

        # prep sim variable for estimation
        # for each var get the the array of choices for the last age
        v = simlc[vlbls][:, :, :jj]
        # log these results replace negatives with a very small number
        lv = np.log(np.where(v > 0, v, 1e-3))

        # Plot life-cycle profiles
        for stat in ['', 'log']:
            print(stat, vlbl)
            fig, ax = plt.subplots()

            if stat == 'log':
                lc = lv
                # if we need to normalize the data to match the model
                if vlbls == 'eph' or vlbls == 'e' or vlbls == 'n' or vlbls == 'n_tru':
                    # model_mult[vlbls] is a dictionary that contains the normalization values to match the data
                    lc = lc + np.log(par2.model_mult[vlbls]) 
            else:
                lc = v
                # if we need normalization
                if vlbls == 'eph' or vlbls == 'e' or vlbls == 'n' or vlbls == 'n_tru':
                    # then perform normalization values to match the data
                    lc = lc * par2.model_mult[vlbls]

            # iterate through ability groups
            for na in range(par.Na):

                lc_mean = np.average(lc[na, :, :], axis=0)
                ax.plot(age, lc_mean, label=na)

            # specify axes and legend
            ax.set_xlim([age[0] - 2, age[-1] + 2])
            ax.set_xlabel('Age')
            ax.set_ylabel(stat + ' ' + vlbl)
            ax.legend()

            # output figure
            fullpath = path + f'fig_lc_{vlbls}_{stat}.pdf'
            fig.savefig(fullpath, bbox_inches='tight')
            plt.close()
            with open(path + f'fig_lc_{vlbls}_{stat}.csv', 'w', newline='') as f:
                pen = csv.writer(f)
                pen.writerow(['age'] + list(age))
                pen.writerow(['model'] + list(lc))
