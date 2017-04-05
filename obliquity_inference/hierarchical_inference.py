import numpy as np

"""
General routines to perform hierarchical bayesian inference based on the method
of Hogg, Myets and Bovy (2010)


"""

def sample_distribution(dist_grid,val_grid,nsamples=1):
    """ 
    Nsamples from a tabulated posterior 
    """

    cdf = cumtrapz(dist_grid,val_grid)
    cdf /= cdf.max()
    u = rd.random(size=nsamples)
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
        y_post = y_measurement_pdfs[i]
        pi0_
        sampled_measurements = sample_distribution(y_post,y_vals,nsamples=10000)
        

        for k in range(M):
            f_yk = np.vectorize(y_pdf_given_parameter)(sampled_measurements,parameter_vals[k])
            lnlike[k] += np.log((f_yk[:]/pi0_yk[:]).mean())
        

    
    for kk,post in enumerate(post_data[:]):
        print kk,target_names[kk],star_names[kk]
        #norm = trapz(post,x=cosi_grid)
        #plt.plot(cosi_grid,post/norm)
        samples = sample_distribution(post,cosi_grid,nsamples=K)
        #prob, bins, patches = plt.hist(samples, bins=24, normed=True)
        #plt.show()
        for jj,kappa in enumerate(kappa_grid):
            #plt.plot(cosi_grid,np.vectorize(cosi_pdf)(cosi_grid,kappa),'r-')
            #plt.plot(cosi_grid,np.vectorize(cosi_pdf2)(cosi_grid,kappa),'b-')
            #plt.show()
            #f_ck = np.vectorize(cosi_pdf)(samples,kappa)
            
            pi0_ck = np.repeat(1,K)
           


    concentration_posterior = np.exp(lnlike) * concentration_prior

    concentration_posterior/=cumtrapz(concentration_posterior,initial=0)[-1]
    cum = cumtrapz(concentration_posterior,initial=0)
    kappa_median = kappa_grid[cum <= 0.5][-1]
    kappa_low = kappa_median - kappa_grid[cum <= 0.16][-1]
    kappa_upp = kappa_grid[cum <= 0.84][-1]-kappa_median
