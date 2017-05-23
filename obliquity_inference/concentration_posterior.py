from cosi_pdf import cosi_pdf 
import numpy as np
from numpy import cosh, sqrt, sinh, pi
from scipy.integrate import quad, cumtrapz
from scipy.interpolate import interp2d, RectBivariateSpline
from hierarchical_inference import compute_hierachical_likelihood


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


def compute_kappa_posterior_from_cosI(kappa_vals,cosi_post_list,cosi_vals, cosi_pdf_function = cosi_pdf):

    """
    Use a collection of PDFs (N different stars) for cosI - the cosine of the
    line-of-sight spin-orbit angle
    
    """

    # Prepare the kappa prior
    def prior_function(k):
        return 1.0 / (1 + k**2)**0.75

    concentration_prior = np.vectorize(prior_function)(kappa_vals)

    # Prepare the cosi prior for each target
    cosi_prior_list = []
    for k in range(len(cosi_post_list)):
        cosi_prior_list.append(np.ones(cosi_post_list[k].shape[0])) # for now, just a flat prior
    cosi_prior_list = None
    concentration_likelihood = compute_hierachical_likelihood(kappa_vals,cosi_pdf_function,cosi_vals,
                                                              cosi_post_list,y_measurement_priors=cosi_prior_list)

    concentration_posterior = concentration_likelihood[:] * concentration_prior[:]
    
    return concentration_posterior/cumtrapz(concentration_posterior,x=kappa_vals,initial=0)[-1]



def compute_kappa_posterior_from_lambda():
    """
    Use a collection of PDFs (N different stars) for lambda - the sky-projected
    spin-orbit misalignment

    """

    return concentration_posterior
