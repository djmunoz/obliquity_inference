import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import time, sys
from inclination_distribution import sample_distribution



"""
General routines to perform hierarchical bayesian inference based on the method
of Hogg, Myets and Bovy (2010)


"""


def update_progress(completed,total,tag = ''):
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
    text = "\rCalculating"+tag+": [{0}] {1:.2f}% {2}".format( "="*block + " "*(barLength-block), \
                                                              progress*100, status)
    if (total-completed == 1): text = text+"\n"
    sys.stdout.write(text)
    sys.stdout.flush()


def compute_hierachical_likelihood(parameter_vals, y_pdf_given_parameter,
                                   y_vals, y_measurement_pdfs,y_measurement_priors = None,
                                   k_samples = True,
                                   K = 1000,
                                   full = False):
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

    if (full):
        lnlike  = compute_hierachical_likelihood_contributions(parameter_vals, y_pdf_given_parameter,
                                                               y_vals, y_measurement_pdfs,
                                                               y_measurement_priors = y_measurement_priors,
                                                               k_samples = k_samples,K = K).sum(axis=1)
    else:
        lnlike = np.zeros(M)
        for i in range(N_measurements):
            update_progress(i,N_measurements)
            y_post = y_measurement_pdfs[i]
            sampled_measurements = sample_distribution(y_post,y_vals,nsamples= K)
            if (y_measurement_priors is not None):
                pi0_yk = interp1d(y_vals,y_measurement_priors[i])(np.sort(sampled_measurements))
            else:
                pi0_yk = np.ones(K)

            for k in range(M):
                f_yk = y_pdf_given_parameter(np.sort(sampled_measurements),parameter_vals[k])
                lnlike[k] = lnlike[k] + np.log((f_yk[:]/pi0_yk[:]).mean())
            
            #y, p = np.meshgrid(np.sort(sampled_measurements),parameter_vals)
            #y, p = y.flatten(), p.flatten()
            #f_yk = y_pdf_given_parameter(y,p).reshape(M,K)
            ##f_yk = np.vectorize(y_pdf_given_parameter)(y,p).reshape(M,K)
            #lnlike[:] += np.log((f_yk/pi0_yk[None,:]).mean(axis=1)).flatten()
        

    concentration_likelihood = np.exp(lnlike) 

    return concentration_likelihood


def compute_hierachical_likelihood_contributions(parameter_vals, y_pdf_given_parameter,
                                                 y_vals, y_measurement_pdfs,y_measurement_priors = None,
                                                 k_samples = True, K = 1000):
    """
    Parameter estimation:
    
    Perform the preliminary step of hierarchical Bayesian inference for a one-parameter distribution function
    of a quantity Y for a list of values of said parameter: compute the contribution of N measurements to the loglikelihood
    of the parameter being estimated (array of M values). The combined log-likelihood is the sum of all the N arrays of length M.


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
    - Float matrix of size MxN -  log-likelihood contribution from N measurements, evaluated at M parameter values of
    stored in 'parameter_vals'.

    """
    

    M = parameter_vals.shape[0]
    N = len(y_measurement_pdfs)

    lnlike_cont = np.zeros([M,N])
    
    for i in range(N):
        update_progress(i,N)

        if (y_measurement_priors is not None): mprior = y_measurement_priors[i]
        else: mprior = None

        if (k_samples):
            lnlike_cont[:,i] = compute_hierarchical_likelihood_single_ksamples(parameter_vals,y_pdf_given_parameter,\
                                                                           y_vals, y_measurement_pdfs[i],\
                                                                           mprior, K=K)
        else:
            lnlike_cont[:,i] = compute_hierarchical_likelihood_single_exact(parameter_vals,y_pdf_given_parameter,\
                                                                        y_vals, y_measurement_pdfs[i],\
                                                                        mprior)
        
        #y_post = y_measurement_pdfs[i]
        #sampled_measurements = sample_distribution(y_post,y_vals,nsamples= K)
        #if (y_measurement_priors is not None):
        #    pi0_yk = interp1d(y_vals,y_measurement_priors[i])(np.sort(sampled_measurements))
        #else:
        #    pi0_yk = np.ones(K)
            
        #for k in range(M):
        #    f_yk = y_pdf_given_parameter(np.sort(sampled_measurements),parameter_vals[k])
        #    lnlike_cont[k,i] = np.log((f_yk[:]/pi0_yk[:]).mean())



    return lnlike_cont


def compute_hierarchical_likelihood_single_ksamples(parameter_vals, y_pdf_given_parameter,
                                                    y_vals, y_measurement_pdf,y_measurement_prior = None, K = 1000):


    sampled_measurements = sample_distribution(y_measurement_pdf,y_vals,nsamples= K)
    if (y_measurement_prior is not None):
        pi0_yk = interp1d(y_vals,y_measurement_priors[i])(np.sort(sampled_measurements))
    else:
        pi0_yk = np.ones(K)

    M = parameter_vals.shape[0]
    delta_loglike = np.zeros(M)
    for k in range(M):
        f_yk = y_pdf_given_parameter(np.sort(sampled_measurements),parameter_vals[k])
        delta_loglike[k] = np.log((f_yk[:]/pi0_yk[:]).mean())
    
    return delta_loglike


def compute_hierarchical_likelihood_single_exact(parameter_vals, y_pdf_given_parameter,
                                                 y_vals, y_measurement_pdf,y_measurement_prior = None):

    if (y_measurement_prior is not None):
        pi0_yk = y_measurement_prior
    else:
        pi0_yk = np.ones(len(y_vals))
    
    M = parameter_vals.shape[0]
    delta_loglike = np.zeros(M)
    for k in range(M):
        integrand = y_pdf_given_parameter(y_vals,parameter_vals[k]).flatten() * y_measurement_pdf / pi0_yk[:]
        delta_loglike[k] = np.log(trapz(integrand, x = y_vals))
    
    
    return delta_loglike
