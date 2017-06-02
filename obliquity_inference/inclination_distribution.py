from numpy import sinh, exp, cos, sin, pi
import numpy as np
from scipy.integrate import cumtrapz


"""
Properties of inclination distributions of spin-orbit misalignment according
to the Fisher concentration distribution.

"""

def sample_distribution(dist_grid,val_grid,nsamples=1):
    """ 
    Nsamples from a tabulated posterior 
    """

    cdf = cumtrapz(dist_grid,x=val_grid)
    cdf /= max(cdf)
    u = np.random.random(size=nsamples)
    inds = np.digitize(u,cdf)
    vals= np.asarray(val_grid)[inds]

    
    return vals


def theta_dist(theta,kappa=1):
    if (theta > pi):
        f = 0
    else:
        f = kappa/ sinh(kappa) * exp(kappa * cos(theta)) * sin(theta)

    return f

def phi_dist(phi):
    if (phi < 0) | (phi > 2.0 * pi):
        f = 0
    else:
        f = 0.5 / pi

    return f


def generate_orientation_sample(nsamples,kappa=0):
    
    
    theta_grid = np.linspace(0.0,np.pi,1000)
    phi_grid = np.linspace(0.0,2*np.pi,1000)

    dist_grid_theta = np.vectorize(theta_dist)(theta_grid,kappa)
    theta_samples = sample_distribution(dist_grid_theta,theta_grid,nsamples=nsamples)
    dist_grid_phi = np.vectorize(phi_dist)(phi_grid)
    phi_samples = sample_distribution(dist_grid_phi ,phi_grid,nsamples=nsamples)


    return theta_samples, phi_samples
    
    
