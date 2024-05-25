"""
Created on 2024-05-13 09:49:08

@author: Ben Boyaian
"""
#import stuff
import time
import numpy as np
import csv
from math import exp,sqrt,log
import pandas as pd
import statsmodels.formula.api as sm
import scipy as sp
import matplotlib.pyplot as plt
from numba import njit, prange, guvectorize # NUMBA speed up quite a lot, see the functions that have the decorator just above

# this is the meat of the exercise
#from scipy.optimize import brentq  # root-finding routine probably dont need since not solving to convergence

######Define the class and intialize some stuff
class toy_ls_model_2_EGM() :
    def __init__(self,  
    
             # earnings parameters 
            #(a lot of these are more like shocks and will be drawn in simulation)
            # either from data or randomly from a distribution
            
            # lambda_t_H deterministic component depends on health and age
            # will need to be estimated for each composite health type, likely normalizing lambda_t_GG
            # for know lets use captina's calibration she estimates two sets of coeffecients
            # one for each education group college/noncollege ill take the avg for now
            w_determ_cons = (0.732 + 1.29)/2 , # constant
            w_age = (0.074 + 0.082)/2 , 
            w_age_2 = (-0.00095 + -0.00112)/2 ,
            w_age_3 = (0.0000024 + 0.0000032)/2 ,
            w_avg_good_health = (0.232 + 0.17)/2 ,
            w_avg_good_health_age = (0.0 + 0.0005)/2 ,
            w_good_health = (0.21 + 0.088)/2 ,
            w_good_health_age = (0.0 + 0.0015)/2 ,    

             # nu_t persistent AR(1) shock
            rho_nu = 0.9472, # the autocorrelation coefficient for the earnings shock nu
            sigma_eps_2 = 0.0198, # variance of innovations
            sigma_nu0_2 = 0.093, # variance of initial distribution of the persistent component
            nu_grid_size = 5, #sie of grid or discrete implementation of the ar1 shock process 
            
            # gamma fixed productiviy drawn at birth
            sigma_gamma_2 = 0.051, # variance of initial dist of fixed effect on labor prod
            #a discrete list of productivities to use for testing
            lab_FE_list = np.array([0.0,0.051]),
                        # utility parameters
            beta = 0.95, # discount factor 
            alpha = .5, # cobb douglass returns to consumption
            sigma_util = 3, # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs
            
            # time costs and health
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
            # number of draws and other procedural parameters
            dt = 2500,              # number of monte-carlo integration draws
            dtdraws = 15000 ,       # number of simulation draws
            T = 5,                 # number of time periods -1 (period 0 is first)
            
            # printing level (defines how much to print)
            print_screen = 2               
    
    ) :
        ###initialize earnings parameters###
        # lambda_t_H deterministic component
        # constant and age components
        self.w_determ_cons,self.w_age,self.w_age_2,self.w_age_3 = w_determ_cons,w_age,w_age_2,w_age_3 
        # health components
        self.w_avg_good_health,self.w_avg_good_health_age,self.w_good_health,self.w_good_health_age = w_avg_good_health,w_avg_good_health_age,w_good_health,w_good_health_age
        # nu_t persistent AR(1) shock
        self.rho_nu, self.sigma_eps_2, self.sigma_nu0_2 = rho_nu, sigma_eps_2, sigma_nu0_2
        sigma_eps = sqrt(sigma_eps_2) #convert from variance to standard deviations
        self.nu_grid,self.nu_trans = self.rouwenhorst(nu_grid_size,rho_nu, sigma_eps)    
        # gamma fixed productiviy drawn at birth
        self.sigma_gamma_2 = sigma_gamma_2
        self.lab_FE_list = lab_FE_list
        
        ###iniatlize utlity parameters###
        self.beta,self.alpha,self.sigma_util = beta,alpha,sigma_util
        #iniatlize health and time cost parameters
        self.phi_n,self.phi_H = phi_n,phi_H
        #health transition probabilities toy examples
        self.B2B,self.G2G = B2B,G2G  
        
        ###interest rate and maybe taxes later
        self.r = r
        
        ###intialie grid(s)
        ###define asset grid###
        self.a_min,self.a_max = a_min,a_max
        self.a_grid_size = a_grid_size
        self.a_grid_growth = a_grid_growth
        self.a_grid = self.gen_grid(a_grid_size, a_min, a_max, a_grid_growth)
        
        ###initialize time/age, number of draws and other procedural parameters 
        self.dt,self.dtdraws,self.T = dt,dtdraws,T
        self.print_screen = print_screen

        #value function for all states; 0=age/time, 1=assets, 2=health, 3=fixed effect
        #self._VF = np.zeros((self.nt+1,self.a_grid_size,2,2))

        #define a dictionary containing all params
        self.param_dict = {
                key:value for key, value in self.__dict__.items() 
                if not key.startswith('__') 
                and not callable(value) 
                and not callable(getattr(value, "__get__", None)) # <- important
            }
        
        # if class is this class, then do some stuff
        if self.__class__.__name__ == 'toy_ls_model_2_EGM' :
            print("This is a toy labor supply lifecycle model")
            print('The PARAMETERS are as follows:')
            print()
            print(self.param_dict)

######initialization helper functions e.g. asset grid and discretiation of shock processes
    def rouwenhorst(self, N, rho, sigma, mu=0.0):
        """Rouwenhorst method to discretize AR(1) process"""
        q = (rho + 1)/2
        nu = ((N-1.0)/(1.0-rho**2))**(1/2)*sigma
        s = np.linspace(mu/(1.0-rho)-nu, mu/(1.0-rho)+nu, N) # states

        # implement recursively transition matrix
        P = np.array([[q, 1-q], [1-q, q]])

        for i in range(2,N): # add one row/column one by one
            P = (
                q*np.r_[np.c_[P, np.zeros(i)] , [np.zeros(i+1)]] + (1-q)*np.r_[np.c_[np.zeros(i), P] , 
                    [np.zeros(i+1)]] + (1-q)*np.r_[[np.zeros(i+1)] ,  np.c_[P, np.zeros(i)]] + q*np.r_[[np.zeros(i+1)] ,  np.c_[np.zeros(i), P]]
                    )
            
            P[1:i,:]=P[1:i,:]/2  # adjust the middle rows of the matrix

        return s, P
    
    def gen_grid(self, size, min, max, grid_growth = 0) :
        	#### Define the asset grid:
        if grid_growth == 0.0:
            #if no grid curvature 
            gA = np.linspace(min, max, size)
        elif grid_growth > 0.0:
            #if grid curvature
            gA = np.zeros(size)
            #make empty asset grid
            for i in range(size):
                #fill each cell with a point that accounts for the curvature the grid will be denser in certain places i.e where changes in mu are highest?
                gA[i]= min + (max - (min))*((1 + grid_growth)**i -1)/((1 + grid_growth)**(size-1.0) -1)
        return gA
    
    
 ######define some other helper functions

    # I took this function from state-space Jacobian package
    #because these functions are only really called internally and do not reference the instance parameters
    #i do not think i need to pass in self here i could be wrong though. keep this in mind if it throws an instance or type error
    @guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)')
    def interpolate_y(x, xq, y, yq): 
        """Efficient linear interpolation exploiting monotonicity.
        Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
        Extrapolates linearly when xq out of domain of x.
        Parameters
        ----------
        x  : array (n), ascending data points
        xq : array (nq), ascending query points
        y  : array (n), data points
        Returns
        ----------
        yq : array (nq), interpolated points
        """
        #get number of elements in xquery and x
        nxq, nx = xq.shape[0], x.shape[0]

        xi = 0 #index or ocunter maybe?
        x_low = x[0]
        x_high = x[1]

        #iterate through each query point
        for xqi_cur in range(nxq):
            # grab the current query point 
            xq_cur = xq[xqi_cur] 
            # while the index is less than the size of x -2 
            while xi < nx - 2: 
                #if the upper bound is greater than the current query point break and get new query point
                if x_high >= xq_cur: 
                    break
                #else we have x_high < xq_curr
                xi += 1 #inc counter
                #since the the upper end point is less than the query
                x_low = x_high #set the lower endpoint to the ipper one
                #raise the value of the upper end pint
                x_high = x[xi + 1]
            #get weights for the endpoints to use in the linear interpolation
            xqpi_cur = (x_high - xq_cur) / (x_high - x_low) 
            #weight the end points and do the interpolation store the result in yq at the curr query points location (index-i)
            yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1] 

    @guvectorize(['void(float64[:], float64[:], uint32[:], float64[:])'], '(n),(nq)->(nq),(nq)')
    def interpolate_coord(x, xq, xqi, xqpi):
        """Get representation xqi, xqpi of xq interpolated against x:
        xq = xqpi * x[xqi] + (1-xqpi) * x[xqi+1]
        Parameters
        ----------
        x    : array (n), ascending data points
        xq   : array (nq), ascending query points
        Returns
        ----------
        xqi  : array (nq), indices of lower bracketing gridpoints
        xqpi : array (nq), weights on lower bracketing gridpoints
        """
        nxq, nx = xq.shape[0], x.shape[0]

        xi = 0
        x_low = x[0]
        x_high = x[1]
        for xqi_cur in range(nxq):
            xq_cur = xq[xqi_cur]
            while xi < nx - 2:
                if x_high >= xq_cur:
                    break
                xi += 1
                x_low = x_high
                x_high = x[xi + 1]

            xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
            xqi[xqi_cur] = xi
            xqpi[xqi_cur] =  min(max(xqpi[xqi_cur],0.0), 1.0) # if the weight is outside 0 or 1, this will catch it


#####define the pieces of the HH problem
    #return leisure given labor choice and health status
    def leisure_giv(self, labor, health):
        leisure = 1 - labor*self.phi_n -health*self.phi_H
        return leisure   
    # the utility function given a consumption and labor choice and a current health status 
    def util(self, consum, labor, health) :
        leisure = self.leisure_giv(labor, health)
        util = ((consum**(self.alpha)*leisure**(1-self.alpha))**(1-self.sigma_util))/(1-self.sigma_util)
        #print("VALID CHOICES: ", consumption_choice, " and ", leisure_choice, " util is: ", util)
        return util
    #derivative of the util function wrt consumption c
    def util_c(self, c, labor, health):
        #assign/unpack variable and param values 
        l = self.leisure_giv(labor,health)
        a,s=self.alpha,self.sigma_util
        #evaluate the derivative determined analytically
        eval = a*c**(a - 1)*l**(1 - a)/(c**a*l**(1 - a))**s
        return eval
   
    def util_l(self, c, lab, health) :
        #assign/unpack variable and param values 
        l=self.leisure_giv(lab, health)
        a,s=self.alpha,self.sigma_util
        #evaluate the derivative determined  analytically
        eval = c**a*(1 - a)/(l**a*(c**a*l**(1 - a))**s)
        return eval

    # return the orpimal consumption decision given an asset choice a_prime and a current asset level, health statues, and labor income
    def c_star(self, a_prime, a, health, lab_inc, ) :
        consum = self.alpha*((lab_inc/self.phi_n)*(1 - self.phi_H*health)+(1 + self.r)*a - a_prime)
        return consum
    
    # return the optimal labor decision given an asset choice a_prime and a current asset level, health statues, and labor income
    def n_star(self, a_prime, a, health, lab_inc) :
        lab =  ( (self.alpha/self.phi_n)*(1 - self.phi_H*health)
                +((self.alpha - 1)/lab_inc)((1 + self.r)*a - a_prime)
                )
        return lab
    # return the orpimal leisure decision given an asset choice a_prime and a current asset level, health statues, and labor income
    def l_star(self, a_prime, a, health, lab_inc) :
        leis = self.leisure_giv(self.n_star(a_prime, a, health, lab_inc), health)
        return leis
            
#####define the function that solves the HHs problem
    #write this out as an algorythm in notes first and 
    #make its pieces as helper functions
    # this is the meat of the exercise

    #function definition
        #generate some useful state*shock or maybe just asset*shock grids
        #generate first 'guess' at consumption policy
        #starting with last year backwards induct or count down i.e. fo i in reveresed(range(nt))
            #perform endogenous grid method

#####main
if __name__ == "__main__":
    start_time = time.time()
    print("Running main")
    
    tm1 = toy_ls_model_2_EGM()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
