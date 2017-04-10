import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time, sys
"""
General routines to perform hierarchical bayesian inference based on the method
of Hogg, Myets and Bovy (2010)


"""

np.random.seed(42)

def update_progress(completed,total):
    """
    update_progress() : Displays or updates a console progress bar
    Accepts a float between 0 and 1. Any int will be converted to a float.

    """

    progress = (completed + 1) * 1.0/total
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rCalculating: [{0}] {1:.2f}% {2}".format( "="*block + " "*(barLength-block), progress*100, status)
    if (total-completed == 1): text = text+"\n"
    sys.stdout.write(text)
    sys.stdout.flush()


def sample_distribution(dist_grid,val_grid,nsamples=1):
    """ 
    Nsamples from a tabulated posterior 
    """
    
    cdf = cumtrapz(dist_grid,val_grid)
    cdf /= cdf.max()
    u = np.random.random(size=nsamples)
    inds = np.digitize(u,cdf)
    vals= val_grid[inds]

    return vals


def compute_hierachical_likelihood(parameter_vals, y_pdf_given_parameter,
                                   y_vals, y_measurement_pdfs,y_measurement_priors, K = 1000):
    """
    Parameter estimation:
    
    Perform hierarchical Bayesian inference for a one-parameter distribution function
    of a quantity Y for a list of values of said parameter. The goal is to obtain a posterior
    probability distribution function for the parameter being estimated.


    Args:
    - parameter_vals: Float array of length M - List of values of the parameter being estimated via 
    posterior pdfs. 
    - y_pdf_given_parameter: a function -  pdf of y values given a specific value for the hierarchical parameter
    - y_vals: possible values of the quantity 'y'
    - y_measurement_pdfs: Python list of float arrays of length N_measurements - Each array in 
    the list consists of a pdf (a posterior resulting from a first-layer of Bayesian inference) of
    the value y. Each entry in this list is an *independent* data entry (e.g. different objects in
    a catalog). These pdfs will be sampled.
    - y_measurement_priors: Python list of float arrays of length N_measurements - In principle,
    each realization of these measurement pdfs has an associated prior probability value. 
    - K: number of realizations/samples drawn from each pdf in 'y_measurement_pdfs'
 
    Returns:
    - Float array of length M -  posterior pdfs, evaluated at the parameter values of
    stored in 'parameter_vals'.

    """
    

    M = parameter_vals.shape[0]
    N_measurements = len(y_measurement_pdfs)

    lnlike = np.zeros(M)

    
    for i in range(N_measurements):
        update_progress(i,N_measurements)
        y_post = y_measurement_pdfs[i]
        sampled_measurements = sample_distribution(y_post,y_vals,nsamples= K)
        pi0_yk = interp1d(y_vals,y_measurement_priors)(sampled_measurements)
        #pi0_yk = np.repeat(1,K)
        
        for k in range(M):
            f_yk = np.vectorize(y_pdf_given_parameter)(sampled_measurements,parameter_vals[k])
            lnlike[k] += np.log((f_yk[:]/pi0_yk[:]).mean())
        #y, p = np.meshgrid(sampled_measurements,parameter_vals)
        #y, p = y.flatten(), p.flatten()
        #f_yk = np.vectorize(y_pdf_given_parameter)(y,p).reshape(M,K)
        #lnlike[:] += np.log((f_yk/pi0_yk[None,:]).mean(axis=1)).flatten()


    concentration_likelihood = np.exp(lnlike) 

    return concentration_likelihood


    
