import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint 

from typing import Union, Optional, List, Dict, Callable, Any, Tuple
from types import ModuleType

from scipy.stats import norm 
from scipy.stats import uniform 
from scipy.optimize import minimize

import visTools

import cpy_q_learning 


class Simulation():
    """Simualte q-learning process given init_params. 
    Attributes of parameter names of this class hold the 
    record of simulation processes.
    Also visualization methods are implemented. 
    """
    record_parameters = ["mu_X", "sigma2_X", "mu_Y", "sigma2_Y", "t_real"]

    def __init__(self, init_params) -> None:
        np.random.seed(101)
        self.core = cpy_q_learning.Core(init_params)

        # For record purpose. 
        for name in self.record_parameters:
            setattr(self, name, [])

    def record_values(self) -> None:
        """class variable of record_parameters are stored from 
        self.core (cpy_q_learning.Core) class. 
        This function only works as recoding parameters. 
        """
        for name in self.record_parameters: 
            lis = getattr(self, name)
            value = getattr(self.core, name)
            lis.append(value)

    def one_step(self) -> None:
        self.core.set_distributions()
        self.core.next_trial_realization()
        self.record_values()
        self.core.update_params()

    def start(self, n: int = 3000) -> None:
        self.n = n
        if len(self.mu_X):
            raise Exception("Already run one or more times") 

        for i in range(n):
            self.one_step()

    def vis_trial_realizations(self, values=None, title: Optional[str] = None):
        """Visualize trail realizations.  mu_X and mu_Y are also displayed. 

        Args:
            values: real data. Note, simulation data exist inside instance object.
            title: If None, default title is set.
        """
        if title == None:
            title = "The trial time for each trial through time."
        ind = [ i for i in range(len(self.t_real))]
        with visTools.BasicPlot(xlabel="trial number", ylabel="time from start",
                               title=title) as p:
            p.ax.scatter(ind, self.t_real, s=1)
            p.ax.plot(self.mu_X, label="mu_X")
            p.ax.plot(self.mu_Y, label="mu_Y")
            p.ax.hlines(y=5,xmin=0, xmax=self.n, alpha=0.5, label="5 sec")
            if not isinstance(values, type(None)):
                p.ax.scatter(ind, values, s=1)
            plt.legend()

    def vis_specific_index(self, index: int, title: Optional[str] = None,
                           show : bool = True) -> None: 
        """This functin is for visualize 5 distributions 
        for specific index. 
        """
        if title == None:
            title = f"5 distributions, index={index}"
        for name in self.record_parameters:
            v = getattr(self, name)[index]
            setattr(self.core, name, v)
            self.core.set_distributions()

        if show:
            self.core.visualize_distributions(title=title)

    def vis_variances(self, title: Optional[str] = None) -> None:
        """Visualize variances of experience and non-reward distribution.
        """
        if title == None:
            title = "variance values through time"

        with visTools.BasicPlot(xlabel="trial number", ylabel="value of variance",
                                title=title) as p:
            p.ax.plot(self.sigma2_X, label="sigma2_X")
            p.ax.plot(self.sigma2_Y, label="sigma2_Y")
            plt.legend()

    def average_correct_premature_time(self, n_div: int = 20) -> None:
        """Average correct premature time. 
        Divide self.n over 20 sessions.
        """
        each_n = int(self.n/n_div)

        corrects = []
        prematures = []
        prematures_prop = []
        omissions_prop = []
        for i in range(n_div):
            t_sess = self.t_real[i*each_n:(i+1)*each_n]
            t_sess = np.array(t_sess)
            corrects.append(np.mean(t_sess) )
            t_pre = t_sess[t_sess < self.core.t_M] 
            prematures.append(np.mean(t_pre))
            prop = len(t_pre)/each_n*100
            prematures_prop.append(prop)
            t_omission = t_sess[t_sess > 11]
            prop = len(t_omission)/each_n*100
            omissions_prop.append(prop)  

        with visTools.BasicPlot(xlabel=f"{each_n} sessions", ylabel="time from start",
                                title="correct and premature mean values") as p:
            p.ax.plot(corrects, label="correct",)
            p.ax.plot(prematures, label="premature")
            plt.legend()

        with visTools.BasicPlot(xlabel=f"{each_n} sessions", ylabel="percentage (%)",
                                title="premature proportions for each session") as p:
            p.ax.plot(prematures_prop, label="premature (%)")
            p.ax.plot(omissions_prop, label="omission (%)")
            plt.legend()

class IndividualFitting():
    """Fitting individual data by scipy minimize method. 
    """
    def __init__(self, values: np.ndarray, verbose: bool = True, 
             params_setting: Dict[str, dict] = None
         ) -> None :
        """Set values data. 
        Also Here we add params_setting attributes. 
        This attributes generates are needed list for fitting. 

        Args:
            values: time from the start of the trial to the end of the trial. 
                Exclude omission values from this trial. 
            verbose: When calculating negative log likelihood, print 
                the values of nll and parameters or not. 
        """
        self.values= values 
        self.n = len(values)
        self.verbose = verbose
        self.params_setting = params_setting

    def nll(self, init_params: Dict[str,float]):
        """Negative loglikelihood of one simulation is calculated
        given initial parameters. 
        Initial parameters should be same to cpy_q_learning.Core class. 
        """
        ll = 0
        core = cpy_q_learning.Core(init_params)
        for v in self.values:
            core.set_distributions()
            core.t_real = v 
            ll += core.calculate_loglikelihood()
            core.update_params()
        nll = - ll 
        if self.verbose:
            print("#",nll)
            print(init_params)
        return(nll)

    def wrapper(self, par: List[float]):
        """To fit using minimize, function should take par for estimating. 
        This function is wrapper for fitting. 
        """
        ind = 0
        init_params = {} 
        for k, dic in self.params_setting.items():
            init_params[k] = par[ind] if dic['use'] else dic['x0']
            if dic['use']:
                ind += 1 

        nll = self.nll(init_params )
        return(nll)

    def minimize(self):
        """scipy minimize fitting is done here.
        """
        x0 = [ dic['x0'] for k , dic in self.params_setting.items() if dic['use'] ] 
        bounds = [ dic['bounds'] for k , dic in self.params_setting.items() if dic['use'] ] 
        n_params = len(x0)
        self.res = minimize(fun= self.wrapper, 
                            x0=x0,
                            args= (),
                            bounds = bounds,
                            method="SLSQP",
                            options={"maxiter":500}
                            )

    def sim_res(self, n: int = None, verbose: bool = True):
        """Using fitted results, simulation is done. 
        After running this method, self.sim contains the Simulation class. 
        
        Args:
            n : if n is None, the same number of simulation for the size of values 
                are performed.
            verbose : print out the simulated parameters.
        """
        if n ==None:
            n = self.n
        ind = 0
        init_params = {} 
        for k, dic in self.params_setting.items():
            init_params[k] = self.res["x"][ind] if dic['use'] else dic['x0']
            if dic['use']:
                ind += 1 
        if verbose:
            pprint.pprint(init_params)
        self.sim = Simulation(init_params)
        self.sim.start(n)

def ll_q_learning_model(init_params: Dict[str,float], values: List[float],
                        verbose: bool = False):
    """Loglikelihood of one simulation is calculated
    given initial parameters. 
    Initial parameters should be same to cpy_q_learning.Core class. 
    """
    ll = 0
    init_params = init_params.copy()
    core = cpy_q_learning.Core(init_params)
    for v in values:
        core.set_distributions()
        core.t_real = v 
        ll += core.calculate_loglikelihood()
        core.update_params()
    if verbose:
        print("#",nll)
        print(init_params)
    return(ll)

class EmceeWrapper():
    def __init__(self, d_dict : Dict[str,Dict[str,any]]):
        """

        Args:
            data_dic : dictionary containing data for fitting. 
                See the detail in q_utils.convert_data_to_pymc3_dict.
        """
        self.d_dict = d_dict
        self.n = len(d_dict.keys())
        self.cnt = 0
        self.set_mu_X0_Y0()
        
        # use first session mean of mu. 


    def set_mu_X0_Y0(self):
        """set mu_X0, mu_Y0 from self.d_dict.
        """
        mu_X0 = []
        mu_Y0 = []
        for sub in self.d_dict.keys():
            dfM = self.d_dict[sub]["data"]
            cond = dfM["session_index"] == 1
            dfMM = dfM.loc[cond]
            cond_pre = dfMM["categorical_label"] == "premature"
            muX = dfMM["total_time"].mean()
            muY = dfMM.loc[cond_pre, "total_time"].mean()

            mu_X0.append(muX)
            mu_Y0.append(muY)
        
        self.mu_X0 = mu_X0
        self.mu_Y0 = mu_Y0

    def set_init_params(self, params: Dict[str,any], sub : str) -> Dict[str,float]:
        """Set initi parameters given one subject name and parameters. 
        If you want to change behavior of initiating parameters, edit this function.

        Args:
            params : declaring in loglikelihood.
            sub : one subject name.
        """
        sub_info = self.d_dict[sub]
        # shared parameters. 
        init_params = dict(
            t_M = 5,
            sigma2_r = params["sigma2_r"], 
            sigma2_M = params["sigma2_M"], 
            gamma = params["gamma"], 
            sigma2_X0 = params["sigma2_X0"], 
            sigma2_Y0 = params["sigma2_Y0"],
        )

        # inidividual specific.
        index = sub_info["index"]
        init_params["mu_X0"] = params["mu_X0"][index]
        init_params["mu_Y0"] = params["mu_Y0"][index]

        # type, sex specific.
        init_params["alpha_X"] = params["alpha_X0"] 
        init_params["alpha_Y"] = params["alpha_Y0"] 
        init_params["beta"] = params["beta0"] 

        if sub_info["type"] == "D5KO":
            init_params["alpha_X"] += params["alpha_X_D5KO"]
            init_params["alpha_Y"] += params["alpha_Y_D5KO"]
            init_params["beta"] += params["beta_D5KO"]

        if sub_info["sex"] == "male":
            init_params["alpha_X"] += params["alpha_X_male"]
            init_params["alpha_Y"] += params["alpha_Y_male"]
            init_params["beta"] += params["beta_male"]

        return(init_params)

    def params_constraint(self, p: Dict[str, any]) -> bool :
        """Constraint for paramers. 

        Return:
            If irregular values exist, return True.
        """
#        for i in range(self.n):
#            if p["mu_X0"][i] <= 0:
#                return(True)
#
#            if p["mu_Y0"][i] <= 0:
#                return(True)

        gamma = p["gamma"]
        alphaX0 = p["alpha_X0"]
        alphaX1 = p["alpha_X0"] + p["alpha_X_D5KO"] 
        alphaX2 = p["alpha_X0"] + p["alpha_X_male"] 
        alphaX3 = p["alpha_X0"] + p["alpha_X_D5KO"] + p["alpha_X_male"]

        alphaY0 = p["alpha_Y0"]
        alphaY1 = p["alpha_Y0"] + p["alpha_Y_D5KO"] 
        alphaY2 = p["alpha_Y0"] + p["alpha_Y_male"] 
        alphaY3 = p["alpha_Y0"] + p["alpha_Y_D5KO"] + p["alpha_Y_male"]

        l = [gamma, alphaX0, alphaX1, alphaX2, alphaX3, alphaY0, alphaY1, alphaY2, alphaY3]
        for a in l:
            if a <= 0 or a >= 1:
                return(True)

        beta0 = p["beta0"]
        beta1 = p["beta0"] + p["beta_D5KO"]
        beta2 = p["beta0"] + p["beta_male"]
        beta3 = p["beta0"] + p["beta_D5KO"] + p["beta_male"]
        l = [p["sigma2_r"], p["sigma2_M"], p["sigma2_X0"], p["sigma2_Y0"], beta0, beta1, beta2, beta3]
        for b in l:
            if b <= 0 :
                return(True)
        return(False)


    def loglikelihood(self, params: Dict[str,any]) -> float:
        """Calculate loglikelihood with parameters.

        Args:
            params : contain the followings. 
                  - sigma2_r : float, 
                  - sigma2_M : float, 
                  - alpha_X0 : float, 
                  - alpha_X_male : float, 
                  - alpha_X_D5KO : float,
                  - alpha_Y0 : float, 
                  - alpha_Y_male : float, 
                  - alpha_Y_D5KO : float,
                  - gamma : float, 
                  - beta_0 : float,
                  - beta_male: float,
                  - beta_D5K0 : float,
                  - mu_X0 : List[float],
                  - sigma2_X0 : float, 
                  - mu_Y0 : List[float],
                  - sigma2_Y0 : float
        """
        ll = 0
        for sub in self.d_dict.keys():
            flag = self.params_constraint(params)
            if flag:
                return(-np.inf)
            init_params = self.set_init_params(params, sub)
            ll += ll_q_learning_model(init_params, self.d_dict[sub]["values"], False)

        return(ll)

    def nll(self,params: Dict[str,any]) -> float:
        """Calculate negative log likelihood.
        """
        nll = - self.loglikelihood(params)
        return(nll)


    def wrapper(self, p: List[float]) -> float:
        """Wrapper parameters. 

        Args:
            p: List arranged fitting parameter wrapper.

        Return:
            log-likelihood.
        """
        
        est_params = dict(
            t_M = 5,
            sigma2_r = p[0], 
            sigma2_M = p[1], 
            gamma = p[2], 
            sigma2_X0 = p[3], 
            sigma2_Y0 = p[4],
            alpha_X0 = p[5],
            alpha_X_D5KO = p[6],
            alpha_X_male = p[7],
            alpha_Y0 = p[8],
            alpha_Y_D5KO = p[9],
            alpha_Y_male = p[10],
            beta0 = p[11],
            beta_D5KO = p[12],
            beta_male = p[13],
            mu_X0 = self.mu_X0, 
            mu_Y0 = self.mu_Y0
        )

        ll = self.loglikelihood(est_params)
        self.cnt += 1 
        return(ll)

def set_type_sex_effect(d_ : Dict[str,float], tp: str, sex: str,
                        t_M: float, mu_X0: float, mu_Y0: float
                        ) -> Dict[str,float]:
    """Set type effect.

    Returns:
        alpha_X, alpha_Y and beta was added.
    """
    d = d_.copy()
    d["alpha_X"] = d["alpha_X0"]
    d["alpha_Y"] = d["alpha_Y0"]
    d["beta"] = d["beta0"]
    if tp == "D5KO":
        d["alpha_X"] += d["alpha_X_D5KO"]
        d["alpha_Y"] += d["alpha_Y_D5KO"]
        d["beta"] += d["beta_D5KO"]
    if sex == "male":
        d["alpha_X"] += d["alpha_X_male"]
        d["alpha_Y"] += d["alpha_Y_male"]
        d["beta"] += d["beta_male"]

    d["t_M"] = 5
    d["mu_X0"] = 6
    d["mu_Y0"] = 4
    return(d)





