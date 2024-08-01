"""
pars_shocks_and_wages.py

Created on 2024-05-18 01:09:42

by @author Ben Boyajian

"""
#file to store parameter and shockvalues for the model

import my_toolbox as tb
import numpy as np
from math import exp,sqrt,log,prod
from numba import njit, guvectorize, float64, int64, prange, types
from numba.core.types import unicode_type, UniTuple
from numba.experimental import jitclass
import time





#a big list of parameter values for the model

pars_spec = [   # ('w_determ_cons', float64), # constant in the deterministic comp of wage regression
                # ('w_age', float64),  # wage coeff on age  
                # ('w_age_2', float64),  # wage coeff on age^2 
                # ('w_age_3', float64), # wage coeff on age^3
                # ('w_avg_good_health', float64), # wage coeff on avg or good health 
                # ('w_avg_good_health_age', float64), # wage coeff on avg or good health X age 
                # ('w_good_health', float64), # wage coeff on good health
                # ('w_good_health_age', float64), # wage coeff on good healht X age 
                ('rho_nu', float64), # the autocorrelation coefficient for the earnings shock nu
                ('sigma_eps_2', float64), # variance of innovations
                ('sigma_nu0_2', float64), # variance of initial distribution of the persistent component
                ('nu_grid', float64[:]), # grid to hold the discretized ar1 process for labor prod.
                ('nu_grid_size', int64), #size oof the disccrete grid of shocks
                ('nu_trans', float64[:,:]), # 2d grid to hold the transitions between states in the discretized ar1 process
                ('sigma_gamma_2', float64), # variance of initial dist of fixed effect on labor prod
                ('lab_FE_grid', float64[:]), # a list of values for that fixed effect
                ('lab_FE_grid_size', int64), # the size of the list of values for that fixed effect
                ('lab_FE_weights', float64[:]), # the weights for the fixed effect
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
                ('state_space_shape', UniTuple(int64, 5)), #the shape/dimensions of the full state space with time/age J
                #('state_space_shape_no_j', int64[:]), #the shape/dimensions of the period state space with out time/age J
                ('state_space_shape_no_j', UniTuple(int64, 4)),
                ('state_space_no_j_size', int64), #size of the state space with out time/age J
                ('state_space_shape_sims', UniTuple(int64, 5)), #the shape/dimensions of the period state space for simulations
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
                ('path', unicode_type), #path to save results to
                ('wage_coeff_grid', float64[:,:]), #grid to hold the wage coefficients
                ('wage_min', float64), #minimum wage
                ('max_iters', int64), #maximum number of iterations for root finder
                ('max_calib_iters', int64), #maximum number of iterations for calibration
                # ('w0_min', float64), #minimum wage constant
                # ('w0_max', float64), #maximum wage constant
                # ('w0_grid_size', int64), #size of the wage constant grid this is probably redundant given labor_FE_grid_size
                #('wage_grid', float64[:,:,:,:]),
                #('_VF', float64[:, :, :, :])  # large 4D matrix to hold values functions probably dont need to initialize that in a params class 
       
        ]

@jitclass(pars_spec)
class Pars() :      
    def __init__(self, path,     

            wage_coeff_grid = np.array([[10.0,0.0,0.0,0.0], [20.0,0.5,-0.01,0.0], [30.0,1.0,-0.02,0.0], [40.0,1.5,-0.03,0.0]]),
            wage_min = 0.0001, #minimum wage

            # nu_t persistent AR(1) shock
            rho_nu = 0.9472, # the autocorrelation coefficient for the earnings shock nu
            sigma_eps_2 = 0.0198, # variance of innovations
            sigma_nu0_2 = 0.093, # variance of initial distribution of the persistent component
            nu_grid_size = 2, #sie of grid or discrete implementation of the ar1 shock process

            # gamma fixed productiviy drawn at birth
            sigma_gamma_2 = 0.051, # variance of initial dist of fixed effect on labor prod
            #a discrete list of productivities to use for testing
            lab_FE_grid = np.array([1.0, 2.0, 3.0]),
            lab_FE_weights = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0]),
            # weights for the fixed effect I want to these weights to the same for each FE and initialied given the number of FEs
            # but also want to dynamically allocate them based on the number of FE 
                        # utility parameters
            beta = 0.95, # discount factor
            alpha = 0.70, #.5, # cobb douglass returns to consumption
            sigma_util = 3, # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs

            # time costs and health costs
            phi_n = 1.125, # 40 hours + 5 commuting is the time cost to a discrete work/not work decision
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
            a_grid_growth = 0.0, #detrmines growth rate and thus curvature of asset grid at 0 just doe slinear space
            a_grid_size = 300, #set up for gride with curvature
            H_grid = np.array([0.0,1.0]),
            H_trans = np.array([[0.7, 0.3],
                               [0.2, 0.8]]), 

            lab_min = 0.00,
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
            print_screen = 2,
            max_iters = 100,
            max_calib_iters = 10,
            # w0_min = 5.0,
            # w0_max = 25.0,
            # w0_grid_size = 2  
        ):
        
        self.wage_coeff_grid = wage_coeff_grid
        self.wage_min = wage_min  
        # nu_t persistent AR(1) shock
        self.rho_nu, self.sigma_eps_2, self.sigma_nu0_2 = rho_nu, sigma_eps_2, sigma_nu0_2
        sigma_eps = sqrt(sigma_eps_2) #convert from variance to standard deviations
        self.nu_grid_size= nu_grid_size
        self.nu_grid,self.nu_trans = tb.rouwenhorst_numba(nu_grid_size, rho_nu, sigma_eps)
        #self.nu_trans = tb.rouwenhorst(nu_grid_size, rho_nu, sigma_eps)
        # gamma fixed productiviy drawn at birth
        self.sigma_gamma_2 = sigma_gamma_2
        self.lab_FE_grid = self.wage_coeff_grid[:,0]
        self.lab_FE_weights = lab_FE_weights
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
        self.a_grid = tb.gen_grid(a_grid_size, a_min, a_max, a_grid_growth)
        
        self.H_grid, self.H_trans = H_grid, H_trans
        self.H_grid_size = len(H_grid)

        self.H_by_nu_flat_trans = tb.gen_flat_joint_trans(self.H_trans, self.nu_trans)
        self.H_by_nu_size = self.H_grid_size * self.nu_grid_size

        self.interp_c_prime_grid = np.zeros(self.H_by_nu_size)
        self.interp_eval_points = np.zeros(1)

        self.c_min = c_min

        self.leis_min,self.leis_max = leis_min, leis_max


        self.lab_min = lab_min
        self.lab_max = leis_max / self.phi_n

        ###initialize time/age, number of draws and other procedural parameters
        self.dt,self.J = dt,J

        self.start_age = start_age
        self.end_age = start_age + J + 1
        self.age_grid = np.arange(self.start_age, self.end_age, 1)

        self.sim_draws = sim_draws
        self.print_screen = print_screen

        self.state_space_shape = (self.a_grid_size, self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size, self.J) 
        self.state_space_shape_no_j = (self.a_grid_size, self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size)
        self.state_space_shape_sims = (self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size, self.sim_draws, self.J + 1)
        self.state_space_no_j_size = self.a_grid_size * self.lab_FE_grid_size * self.H_grid_size * self.nu_grid_size

        self.sim_interp_grid_spec = (self.a_min, self.a_max, self.a_grid_size)

        self.path = path
        self.max_iters = max_iters
        self.max_calib_iters = max_calib_iters
        # self.w0_min = w0_min
        # self.w0_max = w0_max
        # self.w0_grid_size = w0_grid_size

        #self.wage_grid = self.gen_wages() 

        #value function for all states; 0=age/time, 1=assets, 2=health, 3=fixed effect
        #self._VF = np.zeros((self.nt+1,self.a_grid_size,2,2))

    #calculate the deterministic part of the wage
    # def to_dict(self) -> dict:
    #     return [('w_determ_cons', self.w_determ_cons), ('w_age', self.w_age), ('w_age_2', self.w_age_2), ('w_age_3', self.w_age_3), ('w_avg_good_health', self.w_avg_good_health), 
    #             ('w_avg_good_health_age', self.w_avg_good_health_age), ('w_good_health', self.w_good_health), ('w_good_health_age', self.w_good_health_age), ('rho_nu', self.rho_nu), 
    #             ('sigma_eps_2', self.sigma_eps_2), ('sigma_nu0_2', self.sigma_nu0_2), ('nu_grid', self.nu_grid), ('nu_grid_size', self.nu_grid_size), ('nu_trans', self.nu_trans),
    #             ('sigma_gamma_2', self.sigma_gamma_2), ('lab_FE_grid', self.lab_FE_grid), ('lab_FE_grid_size', self.lab_FE_grid_size), ('beta', self.beta), ('alpha', self.alpha),
    #             ('sigma_util', self.sigma_util), ('phi_n', self.phi_n), ('phi_H', self.phi_H), ('B2B', self.B2B), ('G2G', self.G2G), ('r', self.r), ('a_min', self.a_min), 
    #             ('a_max', self.a_max), ('a_grid_growth', self.a_grid_growth), ('a_grid', self.a_grid), ('a_grid_size', self.a_grid_size), ('H_grid', self.H_grid), 
    #             ('H_grid_size', self.H_grid_size), ('state_space_shape', self.state_space_shape), ('state_space_shape_no_j', self.state_space_shape_no_j), 
    #             ('state_space_no_j_size', self.state_space_no_j_size), ('state_space_shape_sims', self.state_space_shape_sims), ('lab_min', self.lab_min),
    #             ('lab_max', self.lab_max), ('c_min', self.c_min), ('leis_min', self.leis_min), ('leis_max', self.leis_max), ('dt', self.dt), ('sim_draws', self.sim_draws),
    #             ('J', self.J), ('print_screen', self.print_screen), ('interp_c_prime_grid', self.interp_c_prime_grid), ('interp_eval_points', self.interp_eval_points),
    #             ('H_by_nu_flat_trans', self.H_by_nu_flat_trans), ('H_by_nu_size', self.H_by_nu_size), ('sim_interp_grid_spec', self.sim_interp_grid_spec),
    #             ('start_age', self.start_age), ('end_age', self.end_age), ('age_grid', self.age_grid), ('path', self.path), ('wage_coeff_grid', self.wage_coeff_grid)]

    # def det_wage(self, age: int, health: float) -> float:
    #     """returns the deterministic part of the wage"""
    #     age_comp = self.w_determ_cons + self.w_age*age + self.w_age_2*age*age + self.w_age_3*age*age*age 
    #     #gonna ignore average health which is in Capatina and the iniatialization of theis program for now, will work in more health states when i have composite types working.
    #     health_comp = self.w_good_health*(1-health) + self.w_good_health_age*age*(1-health)
    #     wage = age_comp + health_comp
    #     #print("Comp is ", comp) 
    #     return wage

    #calculate final wage

    #calculate the wage given health, age, lab_fe, and nu i.e. the shocks
    def wage(self,  age: int, lab_fe_ind: int, health: float,  nu: float) -> float:
        """
        wage process
        """
        return tb.cubic(age, self.wage_coeff_grid[lab_fe_ind])
    
    #generate the wage grid
    def gen_wages(self) -> np.ndarray:
        #initialize the wage grid
        wages = np.zeros((self.J, self.lab_FE_grid_size, self.H_grid_size, self.nu_grid_size))
        for j in prange(self.J):        
            for h_ind, health in enumerate(self.H_grid) :
                for nu_ind, nu in enumerate(self.nu_grid):  
                    for lab_FE_ind, lab_FE in enumerate(self.lab_FE_grid):
                        wages[lab_FE_ind, h_ind, nu_ind, j] = self.wage(j, lab_FE_ind, health, nu)
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
        print("Running pars_shocks_and_wages.py")
        start_time = time.time()
        path = "C:/Users/benja/Documents/My Code/my_model_2"
        myPars = Pars(path) 
        print(myPars.beta * (1 + myPars.r))  

        # myShocks = Shocks(myPars)
        # print(myShocks)
                
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
