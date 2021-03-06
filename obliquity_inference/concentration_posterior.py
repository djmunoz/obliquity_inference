from cosi_pdf import cosi_pdf
from lambda_pdf import lambda_pdf 
import numpy as np
from numpy import cosh, sqrt, sinh, pi
from scipy.integrate import quad, cumtrapz, trapz
from scipy.interpolate import interp2d, RectBivariateSpline
from hierarchical_inference import compute_hierarchical_likelihood
from significance import hellinger_distance

"""
Compute the concentration parameter kappa (Fabrycky & Winn, 2009) of spin-orbit
alignment from a dataset of multiple targets

"""

def cosi_integrand(y, k, z):
    """
    Integrand to Eq. (11) of Morton & Winn (2014)
    """
    return cosh(k*sqrt(1-y*y)) / sqrt(1-y*y) / sqrt(1-(z/y)*(z/y))

def cosi_pdf2(z,kappa=1):
    """
    Equation (11) of Morton & Winn (2014)
    """
    return 2*kappa/(pi * sinh(kappa)) * quad(cosi_integrand,z,1,args=(kappa,z,))[0] 



def kappa_prior_function(k):
    return 1.0 / (1 + k**2)**0.75


def kappa_prior_function_alt(k):
    return 1.0 / (1 + k**2)**0.5


def compute_kappa_posterior_from_cosI(kappa_vals,cosi_post_list,cosi_vals, cosi_pdf_function = cosi_pdf,
                                      k_samples = True , K = 1000, max_cosi_value = None):

    """
    Use a collection of PDFs (N different stars) for cosI - the cosine of the
    line-of-sight spin-orbit angle
    
    """


    concentration_prior = np.vectorize(kappa_prior_function)(kappa_vals)

    # Prepare the cosi prior for each target
    cosi_prior_list = []
    for k in range(len(cosi_post_list)):
        cosi_prior_list.append(np.ones(len(cosi_post_list[k]))) # for now, just a flat prior
    cosi_prior_list = None
    concentration_likelihood = compute_hierarchical_likelihood(kappa_vals,cosi_pdf_function,
                                                               cosi_vals,cosi_post_list,
                                                               y_measurement_priors=cosi_prior_list,
                                                               k_samples = k_samples,
                                                               K = K, maxyvalue = max_cosi_value)

    concentration_posterior = concentration_likelihood[:] #* concentration_prior[:]
    
    return concentration_posterior/trapz(concentration_posterior,x=kappa_vals)

def compute_two_population_significance(kappa_vals,cosi_post_list,cosi_vals,size1,size2,
                                        cosi_pdf_function = cosi_pdf):

    return
    


def compute_kappa_posterior_from_lambda(kappa_vals,lambda_post_list,lambda_vals,lambda_pdf_function = lambda_pdf,
                                        k_samples = True, K = 1000, max_lambda_value = None):
    """
    Use a collection of PDFs (N different stars) for lambda - the sky-projected
    spin-orbit misalignment

    """

    concentration_prior = np.vectorize(kappa_prior_function)(kappa_vals)

    # Prepare the lambda prior for each target
    lambda_prior_list = []
    for k in range(len(lambda_post_list)):
        lambda_prior_list.append(np.ones(len(lambda_post_list[k]))) # for now, just a flat prior
        
    lambda_prior_list = None
    concentration_likelihood = compute_hierarchical_likelihood(kappa_vals,lambda_pdf_function,
                                                               lambda_vals,lambda_post_list,
                                                               y_measurement_priors=lambda_prior_list,
                                                               k_samples = k_samples,
                                                               K = K, maxyvalue = max_lambda_value)


    concentration_posterior = concentration_likelihood[:] * concentration_prior[:]
    
    
    return concentration_posterior/trapz(concentration_posterior,x=kappa_vals)



def concentration_two_sample_splitting(cosivals,cosipdf_list,kappa_vals,deltaloglike_contr,ind,draws=5000):
    '''
    Function to separate a sample of cosI PDFs into TWO subsets as given by an array of boolean
    indices (True -> subset set #1, False--> subset #2), compute the posterior of the concentration
    parameter for each subset, and address the statistical significance of the difference between
    those two posteriors.



    Returns - kappa posterior of subset #1 (numpy array)

            - kappa posterior of subset #2 (numpy array)

            - Squared Hellinger distance between the two PDFs (numpy float)

            - Statistical significance of said distance between the PDFs
 
    '''

    cosipdf_list1 = np.asarray(cosipdf_list)[ind,:]
    cosipdf_list2 = np.asarray(cosipdf_list)[np.invert(ind),:]
    size1 = (cosipdf_list1).shape[0]
    size2 = (cosipdf_list2).shape[0]
    
    kappa_post_a = np.exp(deltaloglike_contr[:,ind].sum(axis = 1)) * kappa_prior_function(kappa_vals)
    kappa_post_b = np.exp(deltaloglike_contr[:,np.invert(ind)].sum(axis = 1)) * kappa_prior_function(kappa_vals)
    kappa_post_a /= trapz(kappa_post_a,x=kappa_vals)
    kappa_post_b /= trapz(kappa_post_b,x=kappa_vals)

    # Do random sampling-and-splitting of the total population
    indices = np.random.permutation(np.append(np.ones(size1),np.zeros(size2)).astype(bool))
    hellinger_list = np.zeros(draws)
    for jj in range(draws):
        indices = np.random.permutation(indices)
        kappa_post_groupa = np.exp(deltaloglike_contr[:,indices].sum(axis = 1)) * kappa_prior_function(kappa_vals)
        kappa_post_groupb = np.exp(deltaloglike_contr[:,np.invert(indices)].sum(axis = 1)) * kappa_prior_function(kappa_vals)
        kappa_post_groupa /= trapz(kappa_post_groupa,x=kappa_vals)
        kappa_post_groupb /= trapz(kappa_post_groupb,x=kappa_vals)
        hellinger_list[jj]= hellinger_distance(kappa_post_groupa,kappa_post_groupb,kappa_vals)

    hel_dist = hellinger_distance(kappa_post_a,kappa_post_b,kappa_vals)
    significance = (1.0 - hellinger_list[hellinger_list > hel_dist].shape[0] * 1.0 / draws) * 100
    
    
    return kappa_post_a,kappa_post_b, hel_dist,significance
