# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:00:37 2024

@author: Ben
"""
import time
import numpy as np
import csv
from numpy import exp,sqrt,log
import pandas as pd
import statsmodels.formula.api as sm
import scipy as sp
import matplotlib.pyplot as plt
import numba
from numba import njit, guvectorize, vectorize, float64, int64, prange # NUMBA speed up quite a lot, see the functions that have the decorator just above
from scipy.optimize import brentq  # root-finding routine
from numba.experimental import jitclass
from sys import maxsize

model_dat = [   ('w_determ_cons', float64), # constant
                ('w_age', float64), 
                ('w_age_2', float64),
                ('w_age_3', float64),
                ('w_avg_good_health', float64),
                ('w_avg_good_health_age', float64),
                ('w_good_health', float64),
                ('w_good_health_age', float64), 
                ('rho_nu', float64), # the autocorrelation coefficient for the earnings shock nu
                ('sigma_eps_2', float64), # variance of innovations
                ('sigma_nu0_2', float64), # variance of initial distribution of the persistent component 
                ('sigma_gamma_2', float64), # variance of initial dist of fixed effect on labor prod
                ('lab_FE_list', float64[:]),
                ('beta', float64), # discount factor 
                ('alpha', float64), # cobb douglass returns to consumption
                ('sigma_util', float64), # governs degree of non-seperability between c,l \\sigma>1 implies c,l frisch subs
                ('phi_n', float64), # 40 hours + 5 commuting is the time cost to a discrete work/not work decision
                ('phi_H', float64), # cost to being in bad health, for now pretend there are only two states
                ('B2B', float64),
                ('G2G', float64),
                ('r', float64) , # interest rate on assets
                ('min_a', float64), # can think of this as borrowing no more than 200,000 dollars
                ('max_a', float64),
                ('inc_a', float64),
                ('a_grid', float64[:]),
                ('a_grid_size', int64),
                ('dt', int64),              # number of monte-carlo integration draws
                ('dtdraws', int64),       # number of simulation draws
                ('nt', int64),                 # number of time periods -1 (period 0 is first
                ('print_screen', int64),
                ('_VF', float64[:, :, :, :])
       
        ]



@jitclass(model_dat)
class toy_ls_model() :
    def __init__(self,     
            # earnings parameters 
            #(a lot of these are more like shocks and will be drawn in simulation)
            # either from data or randomly from a distribution
            
            # lambda_t_H deterministic component depends on health and age
            # will need to be estimated for each composite health type, likely normalizing lambda_t_GG
            # for know lets use captina's calibration she estimates two sets of coeffecients
            # one for each education group college/noncollege ill take the avg for now
            w_determ_cons = 1.011000, # constant
            w_age = 0.078000, 
            w_age_2 = -0.001035,
            w_age_3 = 0.000003,
            w_avg_good_health = 0.201000,
            w_avg_good_health_age = 0.000250,
            w_good_health = 0.149000,
            w_good_health_age = 0.000750, 
            
            # nu_t persistent AR(1) shock
            rho_nu = 0.9472, # the autocorrelation coefficient for the earnings shock nu
            sigma_eps_2 = 0.0198, # variance of innovations
            sigma_nu0_2 = 0.093, # variance of initial distribution of the persistent component 
            
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
            min_a = 0, # can think of this as borrowing no more than 200,000 dollars
            max_a = 100,
            inc_a = 10,
            a_grid_size = 300,
            # number of draws and other procedural parameters
            dt = 1,              # number of monte-carlo integration draws
            dtdraws = 10000 ,       # number of simulation draws
            nt = 50,                 # number of time periods index starts at 0
            # printing level (defines how much to print)
            print_screen = 2   
        ):
        
        #initialize earnings parameters
        # lambda_t_H deterministic component
        # constant and age components
        self.w_determ_cons,self.w_age,self.w_age_2,self.w_age_3 = w_determ_cons,w_age,w_age_2,w_age_3 
        # health components
        self.w_avg_good_health,self.w_avg_good_health_age,self.w_good_health,self.w_good_health_age = w_avg_good_health,w_avg_good_health_age,w_good_health,w_good_health_age

        # nu_t persistent AR(1) shock
        self.rho_nu, self.sigma_eps_2, self.sigma_nu0_2 = rho_nu, sigma_eps_2, sigma_nu0_2  
           
        # gamma fixed productiviy drawn at birth
        self.sigma_gamma_2 = sigma_gamma_2
        self.lab_FE_list = lab_FE_list
        
        #iniatlize utlity parameters
        self.beta,self.alpha,self.sigma_util = beta,alpha,sigma_util
        
        #iniatlize health and time cost parameters
        self.phi_n,self.phi_H = phi_n,phi_H

        #health transition probabilities toy examples
        self.B2B,self.G2G = B2B,G2G  
        
        # interest rate and maybe taxes later
        self.r = r
        
        #initialize number of draws and other procedural parameters 
        self.dt,self.dtdraws,self.nt = dt,dtdraws,nt
        self.print_screen = print_screen

        # define asset grid
        self.min_a,self.max_a,self.inc_a = min_a,max_a,inc_a
        self.a_grid_size = a_grid_size
        self.a_grid = np.linspace(min_a, max_a, a_grid_size) 
        # self.a_grid_size = len(self.a_grid) 
               
        #value function for all states; 0=age/time, 1=assets, 2=health, 3=fixed effect
        self._VF = np.zeros((self.nt+1,self.a_grid_size,2,2)) 
        print("This is a toy labor supply lifecycle model")
        self.ValueFunction() 
     
    #make it an np matrix and pull the appropriate probabilites from that. this is faster than checking if statements
    def health_prob(self, age, curr_health, next_health) :
        """ returns the health transition probaility given the agents age, current health, and targer/next health state 
            for now things are age independent, will eventually read these probabilities  in from the file i estimated
            can then consider persistence and things like that if relevant        
        """
        if curr_health == 0 and next_health == 0: #bad to bad health
            return self.B2B
        elif curr_health == 0 and next_health == 1: #bad to good health
            return 1 - self.B2B
        elif curr_health == 1 and next_health == 1: #good to good health
            return self.G2G
        elif curr_health == 1 and next_health == 0: #good to bad health     
            return 1 - self.G2G
    
    def  det_lab_inc(self, age, health) :
        """returns the deterministic part of the wage"""
        age_comp = self.w_determ_cons + self.w_age*age + self.w_age_2*age*age + self.w_age_3*age*age*age
        #gonna ignore average health which is in Capatina and the iniatialization of theis program for now, will work in more health states when i have composite types working.
        health_comp = self.w_good_health*health + self.w_good_health_age*age*health
        comp = age_comp + health_comp
        #print("Comp is ", comp) 
        return exp(comp)
    
    def lab_inc(self, age, health, persistent_shock, fixed_effect) :
        return self.det_lab_inc(age, health)*exp(fixed_effect)*exp(persistent_shock)
    
    def util_funct(self, consumption_choice, labor_choice, health) :
        """returns a utility value given a level of labor, consumption, and health  """
        #print("Cons choice is ", consumption_choices)      
        if consumption_choice <=0:
            #print("BAD CONSUMPTION: ", consumption_choice)
            return -maxsize
        elif  labor_choice <0:
            #print("BAD LABOR: ", labor_choice)
            return -maxsize
        leisure_choice = 1 - labor_choice*self.phi_n -health*self.phi_H              
        if leisure_choice <=0:
            #print("BAD LEISURE: ", leisure_choice)
            return -maxsize
        util = ((consumption_choice**(self.alpha)*leisure_choice**(1-self.alpha))**(1-self.sigma_util))/(1-self.sigma_util)
        #print("VALID CHOICES: ", consumption_choice, " and ", leisure_choice, " util is: ", util)
        return util
    
    def a_choice_specific_util(self, a_tp1, a_t, age, health, fixed_effect, persistent_shock ) :
        """returns a utility value given a next period asset choice, a stock of assets, age, health staus, 
        and a labor productivity draw and fixed effect
        calculate optimal values for consumption and labor given the asset/savings decision 
        using formulas derived analytically and plug them into the utility function
        returns that utility
        """
        #print(a_tp1, a_t, age, health, fixed_effect, persistent_shock)
        lab_inc=self.lab_inc(age,health,persistent_shock,fixed_effect)
        #we know analytically that in each period 
        consum = self.alpha*((lab_inc/self.phi_n)*(1 - self.phi_H*health)+(1 + self.r)*a_t - a_tp1)
        labor = (self.alpha/self.phi_n)*(1-self.phi_H*health) +((self.alpha-1)/lab_inc)*((1+self.r)*a_t - a_tp1)
        return self.util_funct(consum,labor,health)
    
    def _choiceSpecificValue(self, a_tp1, a_t, age, health, fixed_effect, persistent_shock) :
        """ returns the choice specific value function 
            first argument is the choice
            which is assets next period or
            a_tp1=a_t+1=a_prime        
        """
        if (age == self.nt) : # in lst period choose zero assets
            #print("LAST PERIOD")
            return self.a_choice_specific_util(0, a_t, age, health, fixed_effect, persistent_shock)
        else : # if time from 0 to nt-1 
            a_tp1_loc = np.where(self.a_grid == a_tp1)[0][0] #get the location of this asset choice in the grid
            fe_loc = np.where(self.lab_FE_list == fixed_effect)[0][0]
            
            cont_val = self._VF[age+1,a_tp1_loc,health,fe_loc]
            curr_util = self.a_choice_specific_util(a_tp1, a_t, age, health, fixed_effect, persistent_shock)
            #print("The curr util is: ", curr_util, " and the cont val is: ", cont_val)
            return curr_util + self.beta * cont_val
        
    def _valueGivenShock(self, a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock) :
        """Given last periods persistent lab prod and this periods shock
            calculate this periods nu = persistent prod 
            return the choiceSpecificValue given that shock 
        """
        pers_shock = last_nu*self.rho_nu + nu_shock
        #print("the pers shock is ", pers_shock)
        retVal = self._choiceSpecificValue(a_tp1, a_t, age, health, fixed_effect, pers_shock)
        return retVal
    
    # ###a vectoried expected value given shock that can take the whole vector of nu_shocks as an input
    # @guvectorize([(float64[:], float64[:], int64, int64, float64,float64, float64, float64[:], float64[:])], '(n),(n),(),(),(),(),(),(n)->()', 
    #              nopython = True)
    # def _EV_choiceSpecVal_vect(self, a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector, output) :
    #     """ expected value 
    #         for a given vector of state variables
    #         averages across the vector of possible shocks
    #         to return an expected value for a given asset shoice
    #         this is essentially crude integration       
    #     """
    #     #EV = vectorize(self._valueGivenShock)(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector)
    #     #return EV
    #     sum = 0
    #     for draw in prange(self.dt) :
    #         #print(nu_shock_vector[draw])
    #         sum += self._valueGivenShock(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector[draw])
    #     output[0] = sum/self.dt



    def _EV_choiceSpecVal(self, a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector) :
        """ expected value 
            for a given vector of state variables
            averages across the vector of possible shocks
            to return an expected value for a given asset shoice
            this is essentially crude integration       
        """
        #EV = vectorize(self._valueGivenShock)(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector)
        # eval = self._valueGivenShock(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector)
        # EV = np.mean(eval)
        # # EV = np.mean(self._valueGivenShock(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector)) 
        # return EV
        sum = 0
        for draw in prange(self.dt) :
            #print(nu_shock_vector[draw])
            sum = sum + self._valueGivenShock(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector[draw])
        return sum/self.dt

        # sum = np.sum(self._valueGivenShock(a_tp1, a_t, age, health, fixed_effect, last_nu, nu_shock_vector[draw]) for draw in range(self.dt))
        # return sum/self.dt
    
    def ValueFunction(self) :
        """ returns the vector of value function for all states
            and the vector of shocks used to generate them
            list(_VF,persistent_shocks)
        """    
        # draw all shocks, persistent shocks that are ar1, how to do this properly?
        # i think just draw the shock and feed in previous health state maybe...
        # gotta store the history/sequence of shocks
        np.random.seed(1234)
        periods = self.nt + 1
        #draw all shocks and reshape them into a 2d array, alternatively for the AR(1) shock we can just use a discrete implementtion
        # this would actually probably speed things up a lot 
        nu_shock_mat = np.random.normal(0,self.sigma_eps_2,size=self.dt*periods).reshape(self.dt,periods)
         
        # for t/age in range  loop through this in reverse to solve via backwards induction
        for t in prange(self.nt,-1,-1) :  # start from last period  
            #print some stuff so ik its running
            if self.print_screen>=2 :
                #if t == self.nt :
                print('Computing value function for period',t,'... ')
                # elif t>0 :
                #     print(t,'... ' ,end="",flush=True)
                # else :
                #     print(t)
            last_nu =0
            #for now assume that this last periods labor prod was always 0 that is the shocks are drawn iid
            nu_shock_vector = nu_shock_mat[:,t]
            #pull the vector of shocks for this time period, this should be self.dt items long i believe
            
            for at_loc in prange(self.a_grid_size) :
                a_t = self.a_grid[at_loc]
                #get asset location or just use for index loop might be faster
                #  a_tp1_loc = np.where(self.a_grid == a_tp1)[0][0] #get the location of this asset choice in the grid
                #at_loc = np.where(self.a_grid == a_t)[0][0]
                #for gamma in fixed effects maybe keep it to just 2 for now
                for fixed_effect in self.lab_FE_list:
                    fe_loc = np.where(self.lab_FE_list == fixed_effect)[0][0]
                    #for health in health_statuss just two rn this is the previous health state deal with transition probs in the expectation
                    for curr_health in prange(2):                                                 
                        if t<self.nt: #its not the last period
                            #maximize with respect to a_prime
                            highest= -maxsize
                            for a_prime_i in prange(self.a_grid_size):
                                a_prime = self.a_grid[a_prime_i]
                                # given choice of next period assets a_prime calculate the
                                EV_choiceSpecVal_0 = self._EV_choiceSpecVal(a_prime, a_t, t, 0, fixed_effect, last_nu, nu_shock_vector) #expected value if health is 0 =good
                                EV_choiceSpecVal_1 = self._EV_choiceSpecVal(a_prime, a_t, t, 1, fixed_effect, last_nu, nu_shock_vector) #expected value if health is 1=bad
                                #print('EV_choiceSpecVal_0: ', EV_choiceSpecVal_0, 'and EV_choiceSpecVal_1 :', EV_choiceSpecVal_1)
                                #calculate the expected value across the two health types given a_prime
                                maybeHigher = (
                                self.health_prob(t, curr_health, 0) * EV_choiceSpecVal_0 +
                                self.health_prob(t, curr_health, 1) * EV_choiceSpecVal_1
                                )
                                # check if the expected value is higher than the prevvious highest and update if so
                                if maybeHigher > highest :
                                    #print('Highest updated: ', maybeHigher)
                                    highest = maybeHigher
                            self._VF[t,at_loc,curr_health,fe_loc] = highest
                        else: #its the last period
                            #pick a_prime=0 and calculate the expected value across health types 
                            EV_choiceSpecVal_0 = self._EV_choiceSpecVal(0, a_t, t, 0, fixed_effect, last_nu, nu_shock_vector)
                            EV_choiceSpecVal_1 = self._EV_choiceSpecVal(0, a_t, t, 1, fixed_effect, last_nu, nu_shock_vector)
                            self._VF[t,at_loc,curr_health,fe_loc] = (
                                self.health_prob(t, curr_health, 0) * EV_choiceSpecVal_0 +
                                self.health_prob(t, curr_health, 1) * EV_choiceSpecVal_1
                            )
        #return (self._VF,and_the_shocks_that_got_us_there)
        print('Value functions computed.')
        return (self._VF,nu_shock_vector)


if __name__ == "__main__":
    print("Running main")
    start_time = time.time()
    
    d1 = toy_ls_model()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

