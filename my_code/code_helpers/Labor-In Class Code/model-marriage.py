#%%
from ddcpmodel import ddcpmodel
import numpy as np
from math import exp,sqrt


class ddcp_marriage(ddcpmodel):
    # this is the class for the model with marriage dynamics
    # probability of staying married if married is p_stay_married
    # probability of getting married if unmarried is p_get_married

    def __init__(self, 
                 p_stay_married = .8, # probability of staying married
                 p_get_married = .5,  # probability of getting married
                 *args, **kwargs):
        
        self.p_stay_married = p_stay_married
        self.p_get_married = p_get_married
        
        # initialize the parent class to load the other parameters
        ddcpmodel.__init__(self, *args, **kwargs)
        
        # if class is ddcp_marriage, then compute the value function 
        if self.__class__.__name__ == 'ddcp_marriage' :
            self.ValueFunction()

        return

    def _choiceSpecificValue(self,work,time,x,edu,lw,mar,wsh,ush) :
        ''' update choice-specific value function for marriage model
        '''

        if (time == self.nt) :
            EV = self.currentUtility(0,x,edu,lw,mar,0,0)
        else :
            if mar==1:
                EV = self.currentUtility(work,x,edu,lw,mar,wsh,ush) + self.beta * ( \
                    self.p_stay_married * self._VF[time+1,x+work,edu,work,1] + \
                    (1-self.p_stay_married) * self._VF[time+1,x+work,edu,work,0])
            else:
                EV = self.currentUtility(work,x,edu,lw,mar,wsh,ush) + self.beta * ( \
                    self.p_get_married * self._VF[time+1,x+work,edu,work,1] + \
                    (1-self.p_get_married) * self._VF[time+1,x+work,edu,work,0])

        return EV

    def history(self,edu,mar, drawshock = 1, wshock=float('inf'),ushock=float('inf'),ofshock=float('inf')) :
        """ returns a single history 
            set drawshock = 0 and pass shocks as argument to check what happens with specified shocks
        """

        # initialize data
        x      = [0]*(self.nt+1) #np.zeros(self.nt+1,dtype=np.int)
        lw     = [0]*(self.nt+1) #np.zeros(self.nt+1,dtype=np.int)
        work   = [0]*(self.nt+1) #np.zeros(self.nt+1,dtype=np.int)
        owage  = [0]*(self.nt+1) #np.zeros(self.nt+1)
        offer  = [0]*(self.nt+1) #np.zeros(self.nt+1,dtype=np.int)
        time   = [0]*(self.nt+1) #np.zeros(self.nt+1,dtype=np.int)
        mh     = [0]*(self.nt+1)
        
        if drawshock == 1 :
            ushock = np.random.normal( 0, exp(self.alpha_cov_u_1_1),size=self.nt)
            wshock = np.random.lognormal(mean=0.0, sigma=sqrt(exp(self.alpha_var_wsh)),size=self.nt)
            ofshock = np.random.uniform(0,1,size=self.nt)
            mshock = np.random.uniform(0,1,size=self.nt+1)

        mh[0] = mar
        
        for t in range(self.nt+1) :
            time[t] = t         #want to include time in return function
            if t>0 :
                x[t] = x[t-1]+work[t-1]     
                lw[t] = work[t-1]
                if mh[t-1]==1 :
                    mh[t] = mshock[t]<self.p_stay_married
                else:
                    mh[t] = mshock[t]<self.p_get_married
                
                
            # now figure out choices
            if t==self.nt : 
                work[t] = 0 #nobody works last period
                owage[t] = -float('inf')
            else :    
                
                #worked last time or good draw: receives offer and must choose
                if lw[t] == 1 or ofshock[t] <= self.offerProb(edu,mar)  :
                    offer[t] = 1
                    chvalue =   self._choiceSpecificValue(0,t,x[t],edu,lw[t],mh[t],wshock[t],ushock[t]) \
                               ,self._choiceSpecificValue(1,t,x[t],edu,lw[t],mh[t],wshock[t],ushock[t])         
                    for choice in range(2) :
                        if chvalue[choice] == max(chvalue) :
                            work[t] = choice
                else : #no offer, no job
                    offer[t] = 0
                    work[t] == 0
                if offer[t]==1 :
                    owage[t] =  self.detwage(x[t],edu,lw[t])*wshock[t]
                else: 
                    owage[t] =  -9999

        
        return {'time':time,'work':work,'exp': x,'offer':offer,'owage':owage, 'mh': mh}


def test_equality_of_VF(mod1, mod2):

    return np.sum(mod1._VF != mod2._VF)/np.size(mod1._VF)         

def main():

    mod2 = ddcp_marriage(p_stay_married=.2, p_get_married=.5)
 #   mod1 = ddcpmodel()
 #   check = test_equality_of_VF(mod1, mod2)
 #   print('fraction incorrect:', check)

    return mod2

#%%
if __name__ == '__main__':
    mod2 = main()
    mod2.history(1,1)

# %%
