import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from scipy.optimize import fmin, fminbound
from scipy.stats import norm
from sklearn.linear_model import lasso_path

def fdp_tpp_cal(estimator, true_param):
    '''
    objective: 
        calculate fdp and tpp from the estimation of coefficient and their true values 
    input: 
        estimator, true_param (np.array)
    output: 
        fdp, tpp (float)
    '''
    tp = 0
    fp = 0
    for i in range(0, len(true_param)):
        if estimator[i] != 0 and true_param[i] != 0:
            tp += 1
        if estimator[i] != 0 and true_param[i] == 0:
            fp += 1
    tpp = tp / max(sum(true_param != 0), 1)
    fdp = fp / max(sum(estimator !=0 ), 1)
    return tpp, fdp

# below block contains code ported from Matlab implementation of the paper Su, Bogdan & Candes (2016)

def epsilonDT(delta):
    minus_f = lambda x: -(delta + 2 * x * norm.pdf(x) - 2 * (1 + x ** 2) *
                          norm.cdf(-x))/(1 + x ** 2 - 2 * (1 + x ** 2) * norm.cdf(-x) + 2 * x * norm.pdf(x))
    alpha_phase = fminbound(minus_f, 0, 8)
    epsilon = -minus_f(alpha_phase)
    return epsilon

def powermax(delta, epsi):
    if delta >= 1:
        power = 1
        return power
    epsi_star = epsilonDT(delta)
    if epsi <= epsi_star:
        power = 1
        return power
    power = (epsi - epsi_star) * (delta - epsi_star)/epsi/(1 - epsi_star) + epsi_star/epsi
    return power

def lsandwich(t, tpp, delta, epsi):
    Lnume = (1 - epsi) * (2 * (1 + t ** 2) * norm.cdf(-t) - 2 * t * norm.pdf(t)) + epsi * (1 + t ** 2) - delta
    Ldeno = epsi * ((1 + t ** 2) * (1 - 2 * norm.cdf(-t)) + 2 * t * norm.pdf(t))
    L = Lnume / Ldeno
    return L

def rsandwich(t, tpp):
    R = (1 - tpp) / (1 - 2*norm.cdf(-t))
    return R

def fdrlasso(tpp, delta, epsi):
    '''
    Return the boundary curve q*, defined in Section 2.3 of the paper
    '''
    if tpp > powermax(delta, epsi):
        # print('Invalid input!') # comment out to enable warning
        return
    if tpp == 0:
        q = 0
        return q
    ##################################################################
    # make stepsize smaller for higher accuracy
    stepsize = 0.1
    tmax = max(10, math.sqrt(delta/epsi/tpp) + 1)
    tmin = tmax - stepsize
    
    while tmin > 0:
        if lsandwich(tmin, tpp, delta, epsi) < rsandwich(tmin, tpp):
            break
        tmax = tmin
        tmin = tmax - stepsize
    if tmin <= 0:
        stepsize = stepsize / 100
        tmax = max(10, math.sqrt(delta / epsi / tpp) + 1)
        tmin = tmax - stepsize
        while tmin > 0:
            if lsandwich(tmin, tpp, delta, epsi) < rsandwich(tmin, tpp):
                break
            tmax = tmin
            tmin = tmax - stepsize
    diff = tmax - tmin
    while diff > 1e-6:
        tmid = 0.5 * tmax + 0.5 * tmin
        if lsandwich(tmid, tpp, delta, epsi) > rsandwich(tmid, tpp):
            tmax = tmid
        else:
            tmin = tmid
        diff = tmax - tmin
    t = (tmax + tmin) / 2
    q = 2 * (1 - epsi)  * norm.cdf(-t) / (2 * (1 - epsi)  * norm.cdf(-t) + epsi * tpp)
    return q

# generate tpp and fdp bound
def asym_tpp_fdp(delta, epsi):
    tpp = np.arange(0, 1.05, 0.05)
    n = tpp.shape[0]
    fdp = np.zeros(n)
    for i in range(n):
        fdp[i] = fdrlasso(tpp[i], delta, epsi)
        if np.isnan(fdp[i]) == True:
            fdp[i] = 1.0
            
    return tpp, fdp

