import numpy as np
from scipy.stats import randint
from scipy.signal import convolve
import matplotlib.pyplot as plt
from copy import copy
from pandas import DataFrame

from scipy.special import binom

def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])

class dice_roll:
    
    def __init__(self,dice_tuple,drop=0,post_func=False):
        '''
        dice_roll constructor takes two arguments:
        
        dice_tuple: (# dice, # sides)
            e.g., 2d6 is (2,6)
        
        post_func: applies func after evaluating the dice
           e.g., to count if rolling at least a 6 is a hit
           lambda x:x>=6
        '''
        # Check if we're dropping too many dice
        n_dice, n_sides = dice_tuple
        self.dice_rolls = [dice_tuple]
        
        assert n_dice > drop
        # if dropping all dice, return 0 with probability 1
        if n_dice == drop:
            self.pmf = [1]
            self.support = [0]
            return

        

        
        # compute outcomes
        base_domain = np.array([x+1 for x in range(n_sides)])
        base_dist = np.array([randint.pmf(k,1,n_sides+1) for k in base_domain])
        if drop:
            # Modify base dist to reflect droppping lowest values
            cmf = np.cumsum(base_dist)
            # if we're dropping 'drop', then we calculate
            # the dist of dropth order stat
            order_dist_cmf = [np.sum([binom(n_dice,j)*(1-cmf[x])**j *(cmf[x])**(n_dice-j) for j in range(0,n_dice-drop)]) for x in range(0,len(base_domain))]
            # get pmf
            order_dist = np.append(order_dist_cmf[0],np.diff(order_dist_cmf))
            # compute sum distribution conditioning on lower bound
            
            self.support = np.arange(base_domain.min()*(n_dice-drop),base_domain.max()*(n_dice-drop)+1)
            self.pmf = np.zeros(len(self.support))
            
            for x in range(0,len(base_domain)):
                conditioned_dist_base = base_dist
                conditioned_dist_base[x:] = base_dist[x:]/np.sum(base_dist[x:])
                # add back in zeroes
                conditioned_dist_base[:x]*=0
                
                out_dist = conditioned_dist_base
                out_domain = base_domain
                for i in range(0,n_dice-drop-1):
                    out_dist = convolve(out_dist,conditioned_dist_base)
                    out_domain = np.arange(base_domain.min()+out_domain.min(),base_domain.max()+out_domain.max()+1)
                    
                self.pmf+=out_dist*order_dist[x]
            
        else:
            self.pmf = base_dist
            self.support = base_domain
            for i in range(0,n_dice-1):
                self.pmf = convolve(self.pmf,base_dist)
                self.support  = np.arange(base_domain.min()+self.support.min(),base_domain.max()+self.support.max()+1)

        
        
        if post_func:
            fd = post_func(self.support).astype(int)
            self.support = np.arange(fd.min(),fd.max()+1)
            
            # Accumulate
            self.pmf = np.array([np.sum(self.pmf[fd==f]) for f in self.support])
        
    @property
    def cmf(self):
        return np.cumsum(self.pmf)
    
    @classmethod
    def add(cls,diceroll_array):
        '''
        Adds the dicerolls in the array and returns a new array.
        '''
        d_list = list(diceroll_array)
        outroll = copy(d_list.pop())
        for new_roll in d_list:
            outroll.dice_rolls+=new_roll.dice_rolls
            outroll.pmf = convolve(outroll.pmf,new_roll.pmf)
            outroll.support = np.arange(outroll.support.min()+new_roll.support.min(),outroll.support.max()+new_roll.support.max()+1)
            
        return outroll
    
    def __add__(self,other):
        '''
        Adds the dicerolls in the array and returns a new array.
        Also adds integers.
        '''
        if isinstance(other,dice_roll):
            
            outroll = copy(self)
            outroll.dice_rolls+=other.dice_rolls
            outroll.pmf = convolve(outroll.pmf,other.pmf)
            outroll.support = np.arange(outroll.support.min()+other.support.min(),outroll.support.max()+other.support.max()+1)
                
            return outroll
        elif isinstance(other,int):
            return self.apply_function(lambda x: x+other)
        else:
            raise TypeError("Can only add dice_roll to dice_roll or int")
    
    def __sub__(self,other):
        '''
        Adds the dicerolls in the array and returns a new array.
        '''
        if isinstance(other,dice_roll):
            temp = other.apply_function(lambda x: -x)
        elif isinstance(other,int):
            temp = -1*other
        else:
            raise TypeError("Can only subtract dice_roll with dice_roll or int")
        return self + temp
    
    def sum(self,k,drop=0):
        '''
        Adds a sample of size k using this dice_roll object
        Can drop rolls
        '''
        assert k > drop
        
        outroll = copy(self)
        # if dropping all dice, return 0 with probability 1
        if k == drop:
            outroll.pmf = [1]
            outroll.support = [0]
            return

        # compute outcomes
        base_domain = copy(self.support)
        base_dist = copy(self.pmf)
        if drop:
            # Modify base dist to reflect droppping lowest values
            cmf = np.cumsum(base_dist)
            # if we're dropping 'drop', then we calculate
            # the dist of dropth order stat
            order_dist_cmf = [np.sum([binom(k,j)*(1-cmf[x])**j *(cmf[x])**(k-j) for j in range(0,k-drop)]) for x in range(0,len(base_domain))]
            # get pmf
            order_dist = np.append(order_dist_cmf[0],np.diff(order_dist_cmf))
            # compute sum distribution conditioning on lower bound
            
            outroll.support = np.arange(base_domain.min()*(k-drop),base_domain.max()*(k-drop)+1)
            outroll.pmf = np.zeros(len(outroll.support))
            
            for x in range(0,len(base_domain)):
                conditioned_dist_base = base_dist
                conditioned_dist_base[x:] = base_dist[x:]/np.sum(base_dist[x:])
                # add back in zeroes
                conditioned_dist_base[:x]*=0
                
                out_dist = conditioned_dist_base
                out_domain = base_domain
                for i in range(0,k-drop-1):
                    out_dist = convolve(out_dist,conditioned_dist_base)
                    out_domain = np.arange(base_domain.min()+out_domain.min(),base_domain.max()+out_domain.max()+1)
                    
                outroll.pmf+=out_dist*order_dist[x]
            
        else:
            outroll.pmf = base_dist
            outroll.support = base_domain
            for i in range(0,k-1):
                outroll.pmf = convolve(outroll.pmf,base_dist)
                outroll.support  = np.arange(base_domain.min()+outroll.support.min(),base_domain.max()+outroll.support.max()+1)

        return outroll

    def apply_function(self,func):
        '''
        Adds func the diceroll.
        
        Example: to count if rolling at least a 6 is a hit on 2d6,
        dice_roll((2,6)).apply_function(lambda x:x>=6)
        
        Function output will be cast to ints
        '''
        outroll = copy(self)
        
        fd = func(self.support).astype(int)
        
        outroll.support = np.arange(fd.min(),fd.max()+1)
        outroll.pmf = np.array([np.sum(outroll.pmf[fd==f]) for f in outroll.support])
        
        return outroll
        
    def plot(self,pmf=True,title="",filename=False):
        '''
        Plots the probability mass function or cumulative mass function
        depending on if pmf is true or not.
        
        Can add a title with the title parameter
        
        If filename is present, will write the output to that location.
        '''
        if title:
            plt.title(title)
        if pmf:
            plt.bar(self.support,self.pmf,width = .4)
        else:
            plt.bar(self.support,self.cmf,width = .4)
            
        if filename:
            plt.savefig(filename)

    @property
    def df(self):
        '''
        Outputs the pmf or cmf to a pandas dataframe
        '''
        m = np.where(self.pmf)
        
        return DataFrame({'domain' : self.support[m],'pmf' : self.pmf[m],'cmf' : self.cmf[m]}).set_index("domain")
        
    def __repr__(self):
        return str(self.df)
    
    @property
    def mean(self):
        return np.sum(self.pmf*self.support)

    @property
    def median(self):
        return self.quantiles([.5])[0]

    def quantiles(self,q=[0,0.25,0.5,.75,1]):
        return self.support[[np.min(np.where(self.cmf>=qq)) for qq in q]]
    
    @property
    def mode(self):
        return self.support[np.argmax(self.pmf)]
        
    