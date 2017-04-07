from cosi_pdf import cosi_pdf 

"""
Compute the concentration parameter kappa (Fabrycky & Winn, 2009) of spin-orbit
alignment from a dataset of multiple targets

"""



def compute_kappa_posterior_from_cosI(kappa_vals,cosi_post_list,cosi_vals):

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
    
    
    concentration_likelihood = compute_hierachical_likelihood(kappa_vals,cosi_pdf,cosi_vals,
                                                              cosi_post_list,cosi_prior_list)

    concentration_posterior = concentration_likelihood[:] * concentration_prior[:]
    
    return concentration_posterior



def compute_kappa_posterior_from_lambda():
    """
    Use a collection of PDFs (N different stars) for lambda - the sky-projected
    spin-orbit misalignment

    """

    return concentration_posterior
