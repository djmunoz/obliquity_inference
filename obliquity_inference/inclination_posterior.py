import numpy as np
import scipy.special as spec
from scipy.integrate import quad

"""
Bayesian inference routines for the line-of-star inclination of a single star
or of a collection of stars on an object-by-object basis

"""


def sample_veq_vals(Pval,Perr,Rval,Rerr,N=20000):
    """
    From (double sided) Gaussian distributions in rotational period and stellar radius,
    compute a PDF of the stellar equatorial velocity

    Inputs:

    - Pval (FLOAT): Measured period (expectation value of the period PDF) in DAYS.

    - Perr (FLOAT or FLOAT list): uncertainty in the measured period (standard deviation in the period PDF)

    - Rval (FLOAT): Measured/inferred stellar radius (expectation value of the radius PDF) in SOLAR RADII.

    - Rerr (FLOAT or FLOAT list): One-sided (FLOAT) or two-sided (two-element list) radius uncertainties.
    
    Output:

    - Numpy array of size N containing the Monte-Carlo sampled values of the equatorial velocity
    distribution

    """
    
    p_vals = norm(Pval,Perr).rvs(N)
    while (len(p_vals[p_vals < 0]) > 0):
           p_vals[p_vals < 0] =  norm(Pval,Perr).rvs(len(p_vals[p_vals < 0]))
    try:
       n = len(Rerr)
       if (n >1):
           r_vals_upp = norm(Rval,Rerr[0]).rvs(N)
           r_vals_upp =  r_vals_upp[r_vals_upp >= Rval] 
           r_vals_low = norm(Rval,Rerr[1]).rvs(N)
           while (len(r_vals_low[r_vals_low < 0]) > 0):
               r_vals_low[r_vals_low < 0] =  norm(Rval,Rerr[1]).rvs(len(r_vals_low[r_vals_low < 0]))
           r_vals_low =  r_vals_low[r_vals_low < Rval]
           r_vals = rd.choice(np.append(r_vals_upp,r_vals_low),N)
    except TypeError:
        r_vals = norm(Rval,Rerr).rvs(N)
        while (len(r_vals[r_vals < 0]) > 0):
            r_vals[r_vals < 0] =  norm(Rval,Rerr).rvs(len(r_vals[r_vals < 0]))
    
    # starspot distribution
    l_vals = norm(20.0,20.0).rvs(N) * np.pi/180.0 
    peq_vals = p_vals * (1 - ALPHA * np.sin(l_vals)**2)

    v_vals = 2 * np.pi * r_vals / peq_vals * Rsun_in_km/ day_in_seconds

    return v_vals


def posterior_cosi_full(cosi,vsini_dist,veq_dist):

    def cos_like_integrand(v):
        y = v / np.sqrt(1.0 - cosi * cosi)
        return vsini_dist(v) * veq_dist(y)

    post = quad(cos_like_integrand,0.0,np.inf,epsrel=1.e-8,limit=200)[0]
    
    return post


def posterior_cosi_analytic(cosi,vsini_val,vsini_err,veq_val,veq_err):

    A = 1.0/np.sqrt(2 * np.pi * (vsini_err**2 / (1 - cosi**2) + veq_err**2)) * np.exp(-(vsini_val - veq_val*np.sqrt(1 - cosi**2))**2/2/(vsini_err**2 + veq_err**2*(1 - cosi**2))) 
    v_bar = (vsini_val * veq_err**2 * np.sqrt(1.0 - cosi**2) + veq_val * vsini_err**2)/(veq_err**2 * (1 - cosi**2) + vsini_err**2)
    sigma_bar = 1.0 / np.sqrt((1 - cosi**2)/ vsini_err**2 + 1.0 / veq_err**2)
    post = A * 0.5 * (1.0 + spec.erf(v_bar/np.sqrt(2)/sigma_bar))

    return post





def  compute_cosipdf_from_dataframe(df, columns = ['Vsini','dVsini','Veq','dVeq_plus','dVeq_minus'], analytic_approx=True, Npoints = 200):


    cosi_arr = np.linspace(0.0,0.99999999,Npoints)
    
    post_list = []

    for index,row in df.iterrows():
        
        vs = float(row[columns[0]])
        dvs = float(row[columns[1]])
        veq = float(row[columns[2]])
        dveq = float(0.5 * (row[columns[3]] + row[columns[4]]))
    

    
        if (analytic_approx):
            post_list.append(np.asarray([posterior_cosi_analytic(c,vs,dvs,veq,dveq) for c in cosi_arr]))
        else:
            post_list.append(np.asarray([posterior_cosi_analytic(c,vs,dvs,veq,dveq) for c in cosi_arr]))
        
        

        
    return cosi_arr, post_list
