"""
Createdon 2024-05-18 01:09:42

by @author Ben Boyajian

"""
#file to store parameter and shockvalues for the model

import my_toolbox
import numpy as np
from math import exp,sqrt,log
from numba import njit, guvectorize, float64, int64, prange, types
from numba.experimental import jitclass
import time


#a big list of parameter values for the model

pars_spec = [  ('w_determ_cons', float64), # constant in the deterministic comp of wage regression
                ('w_age', float64),  # wage coeff on age  
                ('w_age_2', float64),  # wage coeff on age^2 
                ('w_age_3', float64), # wage coeff on age^3
                ('w_avg_good_health', float64), # wage coeff on avg or good health 
                ('w_avg_good_health_age', float64), # wage coeff on avg or good health X age 
                ('w_good_health', float64), # wage coeff on good health
                ('w_good_health_age', float64), # wage coeff on good healht X age 
                ('rho_nu', float64), # the autocorrelation coefficient for the earnings shock nu
                ('sigma_eps_2', float64), # variance of innovations
                ('sigma_nu0_2', float64), # variance of initial distribution of the persistent component
                ('nu_grid', float64[:]), # grid to hold the discretized ar1 process for labor prod.
                ('nu_grid_size', int64), #size oof the disccrete grid of shocks
                ('nu_trans', float64[:,:]), # 2d grid to hold the transitions between states in the discretized ar1 process
                ('sigma_gamma_2', float64), # variance of initial dist of fixed effect on labor prod
                ('lab_FE_grid', float64[:]), # a list of values for that fixed effect
                ('lab_FE_grid_size', int64), # the size of the list of values for that fixed effect
                ('beta', float64), # discount factor 
                ('alpha', float64), # cobb douglass returns to consumption
                ('sigma_util', float64), # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs
                ('phi_n', float64), # 40 hours + 5 commuting is the time cost to a discrete work/not work decision
                ('phi_H', float64), # cost to being in bad health, for now pretend there are only two states
                ('B2B', float64), #health transition probaility for bad to bad
                ('G2G', float64), # health transition prob for good to good
                ('r', float64) , # interest rate on assets
                ('a_min', float64), # smallest possible asset value = the borrowing constraint
                ('a_max', float64), # largest possible asset value
                ('a_grid_growth', float64), # spacing/curvature/growth of asset grid parameter
                ('a_grid', float64[:]), # stores the asset grid
                ('a_grid_size', int64), # total number of points on the asset grid
                ('H_grid', float64[:]), # stores the health grid
                ('H_grid_size', int64), # total number of points on the health grid
                ('H_trans', float64[:,:]), #matrix of health transition probabilities
                ('state_space_shape', int64[:]), #the shape/dimensions of the full state space with time/age J
                ('state_space_shape_no_j', int64[:]), #the shape/dimensions of the period state space with out time/age J
                ('state_space_shape_sims', int64[:]), #the shape/dimensions of the period state space for simulations
                ('lab_min', float64 ), #minimum possible choice for labor, cannot pick less than 0 hours
                ('lab_max', float64), # max possible for labor choice
                ('c_min', float64 ), #minimum possible choice for consumption, cannot pick less than 0
                ('leis_min', float64), #min possible choice for leisure
                ('leis_max', float64), #max possible choice for leisure 
                ('dt', int64),              # number of monte-carlo integration draws
                ('sim_draws', int64),       # number of simulation draws
                ('J', int64),                 # number of time periods -1 (period 0 is first
                ('print_screen', int64),  #indicator for what type of printing to do... may drop
                ('interp_c_prime_grid', float64[:]),
                ('interp_eval_points', float64[:]),
                ('H_by_nu_flat_trans', float64[:]),
                ('H_by_nu_size', int64),
                ('sim_interp_grid_spec', types.Tuple((float64, float64, int64))),
                ('start_age', int64), #age to start the model at
                ('end_age', int64), #age to end the model at
                ('age_grid', int64[:]), #grid of ages 
                #('wage_grid', float64[:,:,:,:]),
                #('_VF', float64[:, :, :, :])  # large 4D matrix to hold values functions probably dont need to initialize that in a params class 
       
        ]

@jitclass(pars_spec)
class Pars() :      
    def __init__(self,     
            # earnings parameters
            #(a lot of these are more like shocks and will be drawn in simulation)
            # either from data or randomly from a distribution

            # lambda_t_H deterministic component depends on health and age
            # will need to be estimated for each composite health type, likely normalizing lambda_t_GG
            # for know lets use captina's calibration she estimates two sets of coeffecients
            # one for each education group college/noncollege ill take the avg for now
            w_determ_cons = 1.011000, # constant in the deterministic comp of wage regression
            w_age = 0.078000, # wage coeff on age 
            w_age_2 = -0.001035, # wage coeff on age^2 
            w_age_3 = 0.000003, # wage coeff on age^3
            w_avg_good_health = 0.201000, # wage coeff on avg or good health
            w_avg_good_health_age = 0.000250, # wage coeff on avg or good healt X age
            w_good_health = 0.149000, # wage coeff on good health
            w_good_health_age = 0.000750, # wage coeff on good health X age

            # nu_t persistent AR(1) shock
            rho_nu = 0.9472, # the autocorrelation coefficient for the earnings shock nu
            sigma_eps_2 = 0.0198, # variance of innovations
            sigma_nu0_2 = 0.093, # variance of initial distribution of the persistent component
            nu_grid_size = 2, #sie of grid or discrete implementation of the ar1 shock process

            # gamma fixed productiviy drawn at birth
            sigma_gamma_2 = 0.051, # variance of initial dist of fixed effect on labor prod
            #a discrete list of productivities to use for testing
            lab_FE_grid = np.array([0.0,0.051]),
                        # utility parameters
            beta = 0.95, # discount factor
            alpha = .5, # cobb douglass returns to consumption
            sigma_util = 3, # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs

            # time costs and health costs
            phi_n = .01125, # 40 hours + 5 commuting is the time cost to a discrete work/not work decision
            phi_H = .10, # cost to being in bad health, for now pretend there are only two states
            # phi_H will need to be estimated for each composite health type, normalizing phi_GG = 0

            #health transition probabilities toy examples
            B2B = 0.25,
            G2G = 0.65,

            # interest rate and maybe taxes later
            r = 0.02, # interest rate on assets

            # define asset grid
            a_min = -50, # can think of this as borrowing no more than a_min*1000 dollars
            a_max = 250, # max of the asset grid
            a_grid_growth = 0.025, #detrmines growth rate and thus curvature of asset grid
            a_grid_size = 300, #set up for gride with curvature
            H_grid = np.array([0.0,1.0]),
            H_trans = np.array([[0.7, 0.3],
                               [0.2, 0.8]]), 

            lab_min = 0.0001,
            lab_max = 1.0,
            c_min = 0.0001,
            leis_min = 0.0,
            leis_max = 1.0,    

            # number of draws and other procedural parameters
            dt = 2500,              # number of monte-carlo integration draws
            sim_draws = 1000 ,       # number of simulation draws
            J = 50,                 # number of time periods -1 (period 0 is first)
            start_age = 25, #age to start the model at

            # printing level (defines how much to print)
            print_screen = 2  
        ):
        
        ###initialize earnings parameters###
        # lambda_t_H deterministic component
        # constant and age components
        self.w_determ_cons,self.w_age,self.w_age_2,self.w_age_3 = w_determ_cons,w_age,w_age_2,w_age_3
        # health components
        self.w_avg_good_health,self.w_avg_good_health_age,self.w_good_health,self.w_good_health_age = w_avg_good_health,w_avg_good_health_age,w_good_health,w_good_health_age
        # nu_t persistent AR(1) shock
        self.rho_nu, self.sigma_eps_2, self.sigma_nu0_2 = rho_nu, sigma_eps_2, sigma_nu0_2
        sigma_eps = sqrt(sigma_eps_2) #convert from variance to standard deviations
        self.nu_grid_size= nu_grid_size
        self.nu_grid,self.nu_trans = my_toolbox.rouwenhorst_numba(nu_grid_size, rho_nu, sigma_eps)
        #self.nu_trans = my_toolbox.rouwenhorst(nu_grid_size, rho_nu, sigma_eps)
        # gamma fixed productiviy drawn at birth
        self.sigma_gamma_2 = sigma_gamma_2
        self.lab_FE_grid = lab_FE_grid
        self.lab_FE_grid_size = len(self.lab_FE_grid)

        ###iniatlize utlity parameters###
        self.alpha,self.sigma_util = alpha,sigma_util
       
        #iniatlize health and time cost parameters
        self.phi_n,self.phi_H = phi_n,phi_H
        #health transition probabilities toy examples
        self.B2B,self.G2G = B2B,G2G

        ###interest rate and maybe taxes later
        self.r = r
        self.beta = 1/(1 + r)

        ###intialie grid(s)
        ###define asset grid###
        self.a_min,self.a_max = a_min,a_max
        self.a_grid_size = a_grid_size
        self.a_grid_growth = a_grid_growth
        self.a_grid = my_toolbox.gen_grid(a_grid_size, a_min, a_max, a_grid_growth)
        
        self.H_grid, self.H_trans = H_grid, H_trans
        self.H_grid_size = len(H_grid)

        self.H_by_nu_flat_trans = my_toolbox.gen_flat_joint_trans(self.H_trans, self.nu_trans)
        self.H_by_nu_size = self.H_grid_size * self.nu_grid_size

        self.interp_c_prime_grid = np.zeros(self.H_by_nu_size)
        self.interp_eval_points = np.zeros(1)

        self.lab_min = lab_min
        self.lab_max = lab_max

        self.c_min = c_min

        self.leis_min,self.leis_max = leis_min, leis_max

        ###initialize time/age, number of draws and other procedural parameters
        self.dt,self.J = dt,J

        self.start_age = start_age
        self.end_age = start_age + J + 1
        self.age_grid = np.arange(start_age, self.end_age, 1)

        self.sim_draws = sim_draws
        self.print_screen = print_screen

        self.state_space_shape = np.array([self.a_grid_size, self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size, self.J]) 
        self.state_space_shape_no_j = np.array([self.a_grid_size, self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size])
        self.state_space_shape_sims = np.array([self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size, self.sim_draws, self.J + 1])

        self.sim_interp_grid_spec = (self.a_min, self.a_max, self.a_grid_size)

        # self.wage_grid = self.gen_wages() 

        #value function for all states; 0=age/time, 1=assets, 2=health, 3=fixed effect
        #self._VF = np.zeros((self.nt+1,self.a_grid_size,2,2))

    #calculate the deterministic part of the wage

    def det_wage(self, age, health) :
        """returns the deterministic part of the wage"""
        age_comp = self.w_determ_cons + self.w_age*age + self.w_age_2*age*age + self.w_age_3*age*age*age 
        #gonna ignore average health which is in Capatina and the iniatialization of theis program for now, will work in more health states when i have composite types working.
        health_comp = self.w_good_health*(1-health) + self.w_good_health_age*age*(1-health)
        wage = age_comp + health_comp
        #print("Comp is ", comp) 
        return wage

    #calculate final wage

    def wage(self, age, fixed_effect, health, persistent_shock) :
        return exp(self.det_wage(age, health)) * exp(fixed_effect) * exp(persistent_shock)    
        
    #generate the wage grid
    def gen_wages(self):
        #initialize the wage grid
        wages = np.zeros((self.J, self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size))
        for j in prange(self.J):        
            for h_ind, health in enumerate(self.H_grid) :
                for nu_ind, nu in enumerate(self.nu_grid):  
                    for fe_ind, lab_FE in enumerate(self.lab_FE_grid):
                        wages[j, fe_ind, h_ind, nu_ind] = self.wage(j, lab_FE, health, nu)
        return wages

shock_spec = [
        ('myPars', Pars.class_type.instance_type),
        ('health_shocks', float64[:]),
        ('nu_shocks', float64[:]),
        ('eps_nu_shocks', float64[:]),
        ]

#i feel like drawing shocks using numpy instead of scipy should be jit-able
@jitclass(shock_spec)
class Shocks: 
    def __init__(
        self,
        myPars : Pars
                ):
        self.myPars = myPars

        #draw health shocks
        #reshape them appropriately
        #self.health_shocks = reshaped_draws
         

        #draw ar1 peristent lab shocks nu
        #reshape them appropriately
        #self.nu_shocks = reshaped_draws
        np.random.seed(1234)
        self.eps_nu_shocks = np.random.normal(0,  sqrt(myPars. sigma_eps_2), 1000)

        #could maybe calculate corresponding labor_income "shocks" here and pass those in as the shokcs?
        

if __name__ == "__main__":
        print("Running main")
        start_time = time.time()
        
        myPars = Pars() 
        print(myPars.state_space_shape_sims)  

        # myShocks = Shocks(myPars)
        # print(myShocks)
                
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
