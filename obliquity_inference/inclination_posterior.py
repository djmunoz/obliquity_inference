import numpy as np
import scipy.special as spec
from scipy.integrate import quad

"""
Bayesian inference routines for the line-of-star inclination of a single star
or of a collection of stars on an object-by-object basis

"""


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
