# lifecycle_labor
# main file for lifecycle_labor

import parameters_shocks
import data
import runs
import numpy as np
import time

print("Running main")
start_time = time.time()

# set paths
path_data = '../input/'
path_output = '../output/'
folder = 0  # subfolder in path_output to direct output

#
#
#
# START of parameter specification
#
#
#

# frisch elasticity of labor
γ = 0.3

# disutility of labor
ψ = 10.0

# number of ability types
Na = 3

# quadratic terms for each ability
grida = np.zeros([Na, 3])
grida[0, :] = [0.5, 0.00, -0.000]
grida[1, :] = [1.0, 0.10, -0.002]
grida[2, :] = [2.0, 0.40, -0.008]

# measurement error in hours and earnings
σ_me_n = 0.10
σ_me_e = 0.10

# labor supply bounds
minn = 0.00
maxn = 0.80

# asset gridpoints
Nk = 301
mink, maxk = -50.0, 50.0

# ages
JR = 40
J = 55

# number of simulations for each type
Nsim = 1_000

# tax and transfer parameters
τ0, τ1, τc, T = 1.00, 0.000, 0.00, 0.0  # 0.81, 0.181, 0.07, 0.122
ybar = 0.5

# scales to make wages and hours comparable to data
model_mult_eph = 1  # 25
model_mult_n = 1  # 100 * 50

#
#
#
# END of parameter specification
#
#
#

# set parameters and draw shocks
par = parameters_shocks.Params_numba(grida=grida,
                                     JR=JR, J=J,
                                     Nk=Nk, mink=mink, maxk=maxk,
                                     maxn=maxn,
                                     γ=γ,
                                     ψ=ψ,
                                     Nsim=Nsim,
                                     Na=Na,
                                     τ0=τ0, τ1=τ1, τc=τc, T=T,
                                     ybar=ybar,
                                     σ_me_n=σ_me_n, σ_me_e=σ_me_e,
                                     model_mult_eph=model_mult_eph,
                                     model_mult_n=model_mult_n,
                                     )
par2 = parameters_shocks.Params_nonumba(par, path_output + f'{folder}/')
shock = parameters_shocks.Shocks(par, par2)

end_time = time.time()
execution_time = end_time - start_time
print("Pars and shocks init after:", execution_time, "seconds") 

# get empirical data
mD = data.import_data(path_data, par)

# set prices
w = 1.00
r = 1 / par.β - 1

results = runs.run_model(r, w, mD, par, par2, shock, calibrate_=False, no_calibrate_but_sim_=True, output_=True)
#Strange behaviour: seems to be robust to inputin shock twice, i.e. the below runs fine
#results = runs.run_model(r, w, mD, par, par2, shock, shock, calibrate_=False, no_calibrate_but_sim_=True, output_=True)
[state_solns, simlc] = results


end_time = time.time()
execution_time = end_time - start_time
print("Total execution time:", execution_time, "seconds") 