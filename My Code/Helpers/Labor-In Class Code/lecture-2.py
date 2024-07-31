#%% Model with unobserved heterogeneity 

from ddcpmodel import ddcpmodel
import pandas as pd

class ddcp_unhet(ddcpmodel):
    # this is the class for the model with unobserved heterogeneity
    # it inherits from ddcpmodel, extends it so that there are two types of agents
    # with different disutility of working
    # new parameters: u_work2, pi1 (fraction of type 1 agents)

    def __init__(self, 
                 u_work2 = -1, # disutility of type 2 agents
                 w_educ2 = [0,0,0], # returns to education of type 2
                 pi1 = .6,     # fraction of type 1 agents
                 *args, **kwargs):
        

        self.pi1 = pi1
        self.u_work2 = u_work2
        
        # initialize the parent class to load the other parameters
        ddcpmodel.__init__(self, *args, **kwargs)
        
        self.type1 = ddcpmodel(*args)
        self.type2 = ddcpmodel(u_work=self.u_work2, *args)
        
        # self._VF1 = self.type1._VF
        # self._VF2 = self.type2._VF
        
 
    def het_histories(self):
        
        df1 = self.type1.many_histories(rep=10)
        df2 = self.type2.many_histories(rep=10)
        df1['type'] = 1
        df2['type'] = 2
        
        df1selection = df1.sample(frac=self.pi1)
        df2selection = df2.sample(frac=1-self.pi1)
        
        return pd.concat([df1selection,df2selection], ignore_index=True)

        
def main():
    pass
    
#%%
if __name__ == '__main__':
    mod1 = ddcp_unhet(u_work=-2, pi1=.234535445, u_work2=-.234)
    
    histories = mod1.het_histories()


# %%
