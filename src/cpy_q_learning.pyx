import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, Optional, List, Dict, Callable, Any, Tuple
from types import ModuleType

from scipy.stats import norm 
from scipy.stats import uniform 
from scipy.optimize import minimize

import visTools

cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
###################################
#####  mathematical notation #####
###################################
cdef extern from "math.h":
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double exp(double x)
    double log(double x)
    double INFINITY
    double acos(double x)
    double pow(double x, double y)
    int round(double x)
    double fabs(double x)
    double fmod(double x, double y)
    double floor(double x)

cpdef double pi():
    return( 3.141592653589793)

@cython.cdivision(True)
cdef double uniformRand():
    '''generate uniform distribution     
    '''
    cdef double r = rand()
    return( r/  RAND_MAX)

@cython.cdivision(True)
cdef double normPdf(double x,double mu = 0,double sd = 1 ):
    ''' calculate probability density function of normal distritbuion.     
    '''
    return( 1/( sd*sqrt(2*pi())) *exp(-1/2.0*pow( (x - mu)/sd,2) )  )

DEF N_PAR = 201
cdef class Core():
    """This class works as iterating the process internally. 
    Store and change steps should be done outside this class.
    
    Methods:
        - set_parameters/__init__: set model related parameters as attributes.
        - set_distributions: Using parameters, calculate distribution. 
        - next_trial_realizeation: generate and set next trial realizeation 
            based on distributions. 
        - calculate_ll: 
        - update_parameters: Update mean and variance of 
            both experience and non-reward distributions. 
    """
    cdef public init_names
    cdef public double[N_PAR] t_dist
    cdef public double t_diff 
    cdef public double alpha_X
    cdef public double alpha_Y
    cdef public double gamma 
    cdef public double beta 
    cdef public double t_M
    cdef public double mu_X
    cdef public double sigma2_X
    cdef public double mu_Y
    cdef public double sigma2_Y 
    cdef public double sigma2_M
    cdef public double t_real
    cdef public double sigma2_r

    cdef public double[N_PAR-1] P_exp
    cdef public double[N_PAR-1] P_non_reward
    cdef public double[N_PAR-1] P_conf
    cdef public double[N_PAR-1] P_choice
    cdef public double[N_PAR-1] P_suv

    cdef public double mu_X0
    cdef public double sigma2_X0
    cdef public double mu_Y0
    cdef public double sigma2_Y0

    def __init__(self, init_params: Dict[str,float]) -> None:
        """When initialization, initial parameters are set to 
        attributes. If needed attributes are not found in init_params, 
        raise error. 

        Args:
            init_params : initial parameters take the following. 
                - sigma2_r : Variance of  the actual timing of start. 
                - sigma2_M : Variance of the adjustment term. 
                - alpha_X : Learning rate of the experience dist.
                - alpha_Y : Learning rate of the non-reward dist.
                - gamma : Learning rate for the adjustment term. 
                - beta : Inverse temperature. 
                - mu_X0 : Initial value for the mean of the experience dist.
                - sigma2_X0 : Initial value for the variance of the experience dist. 
                - mu_Y0 : Initial value for the mean of non-reward dist.
                - sigma2_Y0 : Initial value for the variance of the experience dist. 
        """
        self.t_dist = np.linspace(0,20,N_PAR)
        self.t_diff = self.t_dist[1] - self.t_dist[0]
        self.init_names = ["sigma2_r", "sigma2_M", "alpha_X", "alpha_Y", 
                      "gamma", "beta", "t_M" , 
                      "mu_X0", "sigma2_X0", "mu_Y0", "sigma2_Y0"]

        for name in self.init_names:
            if name not in init_params.keys():
                s = f"init_params do not have the following keys : {name}\n"
                s += f"Needed keys are \n{name}"
                raise Exception(s)
            setattr(self, name, init_params[name])
        
        self.mu_X = self.mu_X0
        self.sigma2_X = self.sigma2_X0
        self.mu_Y = self.mu_Y0
        self.sigma2_Y = self.sigma2_Y0
    

    cpdef double update_params(self):
        """Update the means and variancecs of experience and non-reward distributions. 
        Note that t_real is not updated here. 
        """

        cdef double new_mu_X
        cdef double new_sigma2_X
        cdef double new_mu_Y
        cdef double new_sigma2_Y
        cdef bint flag_premature
        flag_premature = self.t_real < self.t_M 
        if flag_premature: 
            new_mu_X = (1 - self.alpha_X )*self.mu_X +\
                        self.alpha_X*self.t_real 
            new_sigma2_X = (1 - self.alpha_X)**2 *self.sigma2_X +\
                        self.alpha_X**2 *self.sigma2_r 
        else:
            new_mu_X = (1 - self.alpha_X - self.gamma)*self.mu_X +\
                        self.alpha_X*self.t_real +  self.gamma*self.t_M
            new_sigma2_X = (1 - self.alpha_X - self.gamma)** 2 *self.sigma2_X +\
                        self.alpha_X**2 *self.sigma2_r + self.gamma**2 *self.sigma2_M
        self.mu_X = new_mu_X
        self.sigma2_X =  new_sigma2_X

        if flag_premature:
            new_mu_Y = (1 - self.alpha_Y )*self.mu_Y +\
                        self.alpha_Y*self.t_real 
            new_sigma2_Y = (1 - self.alpha_Y)**2 * self.sigma2_Y +\
                        self.alpha_Y**2 *self.sigma2_r
        else:
            new_mu_Y = self.mu_Y
            new_sigma2_Y = self.sigma2_Y 

        self.mu_Y = new_mu_Y
        self.sigma2_Y = new_sigma2_Y

    def discritize_normal_distribution(self,mean, sigma2, t):
        """Return discritize normal distribution. 
        Note that compared with t, index 0 object does not exist. 
        Therefore if t takes t[ind], that probability is disc_pdf[ind-1].
        """
        cdf = norm(loc=mean, scale = np.sqrt(sigma2) )\
                .cdf(t)
        disc_pdf = [ cdf[i+1] - cdf[i] for i in range(len(cdf) - 1)] 
        disc_pdf = np.array(disc_pdf)
        disc_pdf = disc_pdf/np.sum(disc_pdf)/self.t_diff
        return(disc_pdf)


    @cython.cdivision(True)
    cpdef double set_distributions(self):
        """Probability density function of experience, non-reward, confidence, 
        choice and survival distributions are obtained.
        All distributions are discritized.  

        """
        cdef int i 
        cdef double C_norm
        cdef double suv_sum
        cdef double[N_PAR] t
        cdef double[N_PAR-1] P_exp
        cdef double[N_PAR-1] P_non_reward
        cdef double[N_PAR-1] P_conf
        cdef double[N_PAR-1] P_choice
        cdef double[N_PAR-1] P_suv

        for i in range(N_PAR):
            t[i] = self.t_dist[i]
        t_diff = self.t_diff 
        # TO DO: discritize version should be created.
        # P_exp = self.discritize_normal_distribution(self.mu_X, self.sigma2_X, t)
        # P_non_reward = self.discritize_normal_distribution(self.mu_Y, self.sigma2_Y, t)

        for i in range(1,N_PAR):
            P_exp[i-1] = normPdf(t[i], self.mu_X, sqrt(self.sigma2_X) )
            P_non_reward[i-1] = normPdf(t[i], self.mu_Y, sqrt(self.sigma2_Y) )


        C_norm = 0
        for i in range(N_PAR-1):
            P_conf[i] = exp(self.beta*P_exp[i])/\
                        (exp(self.beta*P_exp[i]) + exp(self.beta*P_non_reward[i]) )
            C_norm += P_exp[i]*P_conf[i]* t_diff

        suv_sum  = 0
        for i in range(N_PAR-1):
            P_choice[i] = P_exp[i]*P_conf[i]/ C_norm
            suv_sum += P_choice[i]
            P_suv[i] =  1- suv_sum*t_diff

        for i in range(N_PAR-1):
            self.P_exp[i] = P_exp[i]
            self.P_non_reward[i] = P_non_reward[i]
            self.P_conf[i] = P_conf[i]
            self.P_choice[i] = P_choice[i]
            self.P_suv[i] = P_suv[i]

    cpdef double next_trial_realization(self):
        """Generate next trial realizeation from survival function. 
        This should be run after the "set_distributions" 
        This generated value is t_{n+1} based on X_{n} variables. 
        """
        cdef int i 
        cdef int arg 
        cdef double r 
        cdef double[N_PAR -1] errors
        r = uniformRand()
        arg = 0
        min_ = 10000000
        for i in range(N_PAR -1 ):
            errors[i] = (self.P_suv[i] - r)*(self.P_suv[i] - r)
            if min_ > errors[i]:
                arg = i
                min_ = errors[i]
        self.t_real = self.t_dist[arg+1]

    @cython.cdivision(True)
    cpdef double calculate_loglikelihood(self):
        """Calculate likelihood of q-learning of one session. 
        To run this function, set self.t_real and run self.set_distributions()
        """
        cdef double[N_PAR-1] P_decide
        cdef double sum_
        cdef double ll
        cdef int arg

        sum_ = 0
        for i in range(N_PAR-1):
            sum_ += self.P_exp[i]*self.P_choice[i]
        for i in range(N_PAR-1):
            P_decide[i] = self.P_exp[i]*self.P_choice[i]/sum_

        arg = <int>(self.t_real*10) - 1  # Because data have 0.1 scale. 
        ll = log(P_decide[arg])
        return(ll)

    def visualize_distributions(self,title=None) -> None:
        """Visualize 5 distirbutions.
        See discitize_normal_distribution for t and probability settings. 
        """
        cdef double[N_PAR-1] t 
        cdef int i 
        for i in range(1,N_PAR):
            t[i-1] = self.t_dist[i] 
        with visTools.BasicPlot(xlabel="timing of start", ylabel="probability",
                                title = title ) as p:
            p.ax.plot(t, self.P_exp, label="experience") 
            p.ax.plot(t, self.P_non_reward, label="non-reward")
            p.ax.plot(t, self.P_conf, label="confidence")
            p.ax.plot(t, self.P_choice, label="choice density")
            p.ax.plot(t, self.P_suv, label="choice survival")
            p.ax.vlines(x = 5,ymin=0,ymax=1, color="black")
            plt.legend()

