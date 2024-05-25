# BPModel: calibrate
# calibrate model given data targets and state-specific solutions

import csv
import numpy as np
import data
import simulate
import toolbox


def print_params(r, w, par, par2):
    """
    print model and computational parameters
    """

    path = par2.path

    # Print all parameter values to csv file
    fullpath = path + 'parameters.csv'
    with open(fullpath, 'w', newline='') as f:
        pen = csv.writer(f)
        pen.writerow(
            ['grida',
             'J', 'JR', 'gamma', 'psi', 'omega', 'beta', 'tau0', 'tau1', 'tauc', 'T', 'ybar',
             'Na', 'Nk', 'Nsim',
             'mink', 'maxk', 'minn', 'maxn',
             'sigma_me_n', 'sigma_me_e', 'model_mult[e]', 'model_mult[eph]', 'model_mult[n]',
             'r', 'w', ])
        pen.writerow(
            [par.grida,
             par.J, par.JR, par.γ, par.ψ, par.ω, par.β, par.τ0, par.τ1, par.τc, par.T, par.ybar,
             par.Na, par.Nk, par.Nsim,
             par.gridk[0], par.gridk[-1], par.minn, par.maxn,
             par.σ_me_n, par.σ_me_e, par2.model_mult['e'], par2.model_mult['eph'], par2.model_mult['n'],
             r, w])

    # Print select parameter values to .tex file: exogenously-set parameters
    tab = ["Parameter & Interpretation & Value & Source \\\\ \n"]
    tab.append("\hline \\\\ \n")
    tab.append(f"$R$ & Gross interest rate  & {np.round(1 + r, 4)} & Benchmark \\\\ \n")
    tab.append(f"$\\beta$ & Patience & {np.round(par.β, 4)} & $1/\\beta$ \\\\ \n")
    tab.append(f"$\\sigma$ & CRRA & {np.round(par.ω, 4)} & Benchmark \\\\ \n")
    tab.append(f"$\\gamma$ & Frisch elasticity & {np.round(par.γ, 4)} & Benchmark \\\\ \n")
    fullpath = path + 'parameters_exog.tex'
    with open(fullpath, 'w', newline='\n') as pen:
        for row in tab:
            pen.write(row)
