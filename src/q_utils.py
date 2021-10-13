import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from typing import Union, Optional, List, Dict, Callable, Any, Tuple
from types import ModuleType

import json
import pickle
import emcee

import pathList as pL
import visTools
import q_learning

def convert_category_to_continuous(df: pd.DataFrame):
    """Convert category to continuous value started from the 
    trial. 5 sec and 9 sec is flushing time. 
    See "makeDataFrame" in basicChara for reference code. 

    Args:   
        df: cleaned dataframe. Ex., pL.merged_structured_df. 
    """
    df = df.copy()
    df["total_time"] = np.nan
    # Control variables 
    D = 0.5
    L = 6
    I = 5
    I9 = 9
    O = 5
    Q = 1
    T = 100
    E = 60
    error = False
    flags = ["correct_tag", "premature_reseponse_tag","incorrect_response_tag","omission_tag"]

    cond_ITI = df["SD"].str.contains("ITI")
    cond_omission = df["categorical_label"] == "omission"
    cond_correct = df["categorical_label"] == "correct" 
    cond_incorrect = df["categorical_label"] == "incorrect" 
    cond_premature = df["categorical_label"] == "premature" 

    cond5_omission =  ~cond_ITI & cond_omission 
    cond9_omission = cond_ITI & cond_omission 
    cond5_correct = ~cond_ITI & cond_correct
    cond9_correct = cond_ITI & cond_correct
    cond5_incorrect = ~cond_ITI & cond_incorrect
    cond9_incorrect = cond_ITI & cond_incorrect
    cond5_premature = ~cond_ITI & cond_premature
    cond9_premature = cond_ITI & cond_premature

    df.loc[cond5_omission , "total_time" ] = I + L
    df.loc[cond9_omission , "total_time" ] = I9 + L
    df.loc[cond5_correct, "total_time" ] = I  + df.loc[cond5_correct, "correct_response_latency"]
    df.loc[cond9_correct, "total_time" ] = I9 + df.loc[cond9_correct, "correct_response_latency"]
    df.loc[cond5_incorrect, "total_time" ] = I  + df.loc[cond5_incorrect, "incorrect_response_latency"]
    df.loc[cond9_incorrect, "total_time" ] = I9 + df.loc[cond9_incorrect, "incorrect_response_latency"]
    df.loc[cond5_premature, "total_time" ] = df.loc[cond5_premature, "premature_latency"]
    df.loc[cond9_premature, "total_time" ] = df.loc[cond9_premature, "premature_latency"]

    return(df)

def read_total_time_data(path: str, valid_mice: List[str]):
    """Read and cleaning data.
    Extract only valid subject data, and convert category to total time.

    Args:
        path : path for pL.merged_structured_df.
        valid_mice : mice list included in analysis.
    """
    df = pd.read_csv(path)

    cond = df["subject"].apply(lambda x: any([m == x for m in valid_mice]))
    df = df.loc[cond]
    df = convert_category_to_continuous(df) 

    return(df)


class CaseParams:
    """parameter holder class. For modeling. 
    """
    case1 = {
        'sigma2_r': {'use': True, 'x0':1, 'bounds':(0,None)},
        'sigma2_M': {'use': True, 'x0':0.5, 'bounds':(0,None)},
        'alpha_X': {'use': True, 'x0':0.001, 'bounds':(0,1)},
        'alpha_Y': {'use': True, 'x0':0.02, 'bounds':(0,1)},
        'beta': {'use': True, 'x0':10, 'bounds':(0,None)},
        'gamma': {'use': True, 'x0':0.001, 'bounds':(0,1)},
        't_M': {'use': False, 'x0':5, 'bounds':None},
        'mu_X0': {'use': True, 'x0':7, 'bounds':(5,None)},
        'sigma2_X0': {'use': True, 'x0':15, 'bounds':(0,None)},
        'mu_Y0': {'use': True, 'x0':4, 'bounds':(0,5)},
        'sigma2_Y0': {'use': True, 'x0':15, 'bounds':(0,None)},
    }

    param_names =  ["sigma2_r", "sigma2_M", "alpha_X", "alpha_Y", 
                      "gamma", "beta", "t_M" , 
                      "mu_X0", "sigma2_X0", "mu_Y0", "sigma2_Y0"]

    valid_mice = [
        "23--1", "23--2", "D175", "D178", "D198", "D214", 
        "D264", "D266", "D274", "D280", "D294", "D295", "D363", 
        "HM283", "D136", "D157", "D188", "D199", "D205", "D207", 
        "D238", "D240", "D246", "D260", "D282", "D159",  "D163", 
        "D166", "D170", "D215", "D267", "D268", "D278", "D281", 
        "D283", "D309", "D141", "D184", "D201", "D211", "D237",
        "D287", "D288", "D358", "D365", "D378"
    ] 


def filter_for_fit(dfM: pd.DataFrame, till: int = 10) -> pd.DataFrame:
    """simple filter for fitting q-learning model.
    """
    index = "session_index"
    cond1 = dfM[index] <= till
    cond2 = dfM["omission_tag"] != 1
    dfMM = dfM.loc[cond1 & cond2].sort_values(by="n_sd.1")
    return(dfMM)

def individual_fitting(df: pd.DataFrame, subject: str, till: int = 10):
    """Given dataframe, fitting and save one individual data.

    Args:
        df: whole dataset. 
        subject: subject name. 
        till: including session_index, takes 10 or 20.
    """
    save_path = f"../data/dt_q_case1/case1_{subject}_{till}.json"
    if(os.path.exists(save_path)):
       return
    cond = df["subject"] == subject
    dfM = df.loc[cond].copy()
    dfMM = filter_for_fit(dfM, till=till)

    values = dfMM["total_time"].values
    fit = q_learning.IndividualFitting(values, verbose=False, params_setting = CaseParams.case1)
    fit.minimize()
    print(f"{subject}: {fit.res.message}")

    # save data as json.
    save_res = fit.res.copy()
    save_res["jac"] = fit.res.jac.tolist()
    save_res["x"] = fit.res.x.tolist()
    save_res["success"] = bool(fit.res.success)

    save_json = {
        "res": save_res,
        "n": len(values),
        "till": till,
        "subject":subject,
    }
    with open(save_path, "w") as f:
        json.dump(save_json,f)

def load_fitted_data(subjects: List[str], till: int = 10) -> List[dict]:
    """Load data of fitted data obtained from "individual_fitting" function.
    """
    paths = [ f"../data/dt_q_case1/case1_{subject}_{till}.json" for subject in subjects]
    data = []
    for p in paths:
        with open(p, "r") as f:
            data.append(json.load(f))
    return(data)

def simulate_one_from_fitted_res(d: dict, dfM: pd.DataFrame
                                 ) -> (q_learning.IndividualFitting, List[float]):
    """One simulation is done from the fitted result. 

    Args:   
        d : one element of result data. Loaded by "load_fitted_data" function.
        dfM: dataset filtered by one subject.
    """
    dfMM = filter_for_fit(dfM,  d["till"])
    values = dfMM["total_time"].values
    fit_res = q_learning.IndividualFitting([] , verbose=False, params_setting=CaseParams.case1)
    fit_res.res = d["res"]
    fit_res.sim_res(n=d["n"], verbose=False)
    return(fit_res, values)

def record_df_generator(data: List[dict], df: pd.DataFrame):
    """Create record dataframe for parameters for analysis. 

    Args:   
        data: fitted results.
        df: all dataset.
    """

    record = {k:[] for k in CaseParams.case1.keys()}
    record = {**record, **{ k:[] for k in ["subject", "sex" , "type", "n", "sigma2_X", "sigma2_Y", "mu_X", "mu_Y"]}}
    record["subject"] = []
        
    for d in data:
        cond = df["subject"] == d["subject"]
        dfM = df.loc[cond]
        fit_res, values = simulate_one_from_fitted_res(d, dfM)
        for k in CaseParams.case1.keys():
            v = getattr(fit_res.sim.core,k)
            record[k].append(v)
        for k in ["sigma2_X", "sigma2_Y", "mu_X", "mu_Y"]:
            v = getattr(fit_res.sim, k)[-1]
            record[k].append(v)
            
        record["sex"].append(dfM["sex"].unique()[0])
        record["type"].append( dfM["type"].unique()[0])
        record["subject"].append(d["subject"])
        record["n"].append(d["n"])

    df_record = pd.DataFrame(record)
    df_record["type_sex"] = df_record["type"]  + "_" + df_record["sex"]
    return(df_record)

def check_unique(ser : pd.Series) -> any:
    """Check uniquness. If duplicated, raise Error. 
    """
    l = ser.unique()
    if len(l) != 1:
        raise Exception("Duplication found", l)
    return(l[0])

def convert_data_to_pymc3_dict(df: pd.DataFrame, valid_mice: List[str]
                              ) -> Dict[str,Dict[str,any]]:
    """Convert dataframe to dictionary format can be used in pymc3.

    Args:
        df : dataframe read by q_utils.read_total_time_data.
        valid_mice : mice list included in analysis.

    Return:
        keys are subject names.
        values have the following keys.
            - type : takes "wt" or "D5KO".
            - sex : takes "male" or "female".
            - values : takes total time from start.
            - data : pd.Dataframe for each subject.
            - index : index number for each subject.
    """
    df = df.copy()
    df = filter_for_fit(df, till=10)

    dic_ = {} 
    for i, m in enumerate(valid_mice):
        dic_[m] = {}
        cond = df["subject"] == m 
        dfM = df.loc[cond]
        dic_[m]["index"] = i 
        dic_[m]["type"]  = check_unique( dfM["type"] ) 
        dic_[m]["sex"] = check_unique( dfM["sex"] )
        dic_[m]["data"] = dfM
        dic_[m]["values"] = dfM["total_time"].values
    return(dic_)

def stats2gamma(mu, sigma) -> Tuple[float,float]:
    """convert mean and sd of any distribution into gamma alpha and beta.
    """
    alpha = mu*mu/sigma/sigma
    beta = mu/sigma/sigma
    return(alpha, beta)

def save_sampler(path, sampler):
    """save model using pickle.
    """
    print(path)
    with open(path,"wb") as buff:
        pickle.dump({"sampler":sampler}, buff)

def load_sampler(path):
    with open(path,"rb") as buff:
        data = pickle.load(buff)
    return(data["sampler"])

def emcee_automatic(sampler: emcee.ensemble.EnsembleSampler, 
                    pos : np.ndarray,
                    max_n=100000):
    """Automatic save, run and stop iterations using emcee. 

    Args:
        sampler : initialized with log-likelihood function.
        pos : default values.
        max_n : max iteration. 
    """

    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

def sim_distribution_results(init_dict: Dict[str,float], n_iter: int, n_step: int,
                             **kargs) -> Dict[str,np.ndarray]:
    """Return each distribution results for n_iter times. 
    This is for averaging simulation results.

    Args:
        init_dict: dict
        n_iter: iteration time for each simulation. 
        n_step : number of steps for one simulation. 
        kargs: passed to q_learning.set_type_sex_effect. 
    
    Examples:
        Example set of kargs is. 
            >>> kargs = dict(tp="D5KO", sex="male", t_M=5, mu_X0=6, mu_Y0=4)
            >>> dists = q_utils.sim_distribution_results(init_dict, n_iter=100,n_step=100,**kargs)
    """
    d = q_learning.set_type_sex_effect(init_dict,**kargs)
    base = np.ndarray(shape=(n_iter,200))

    dist_keys = ["P_exp", "P_non_reward", "P_conf", "P_choice", "P_suv"]
    dists = {v:base.copy() for v in dist_keys}
    dists["t_real"] = []

    for i in range(n_iter):
        sim = q_learning.Simulation(d)
        sim.start(n_step)
        sim.vis_specific_index(n_step-1, show=False)
        for k in dist_keys:
            dists[k][i] = getattr(sim.core, k)
        dists["t_real"].append(sim.t_real)
    return(dists)

