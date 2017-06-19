from numpy import sinh, exp, cos, sin, pi
import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import norm

"""
Properties of inclination distributions of spin-orbit misalignment according
to the Fisher concentration distribution.

"""

def twosided_gaussian(xmid,xupp,xlow,xgrid = None):
    if (xgrid is None):
        xgrid = np.linspace(xmid - 4 * xlow, xmid + 4 * xupp,600)
        
    x_dist = np.append(norm.pdf(xgrid[xgrid < xmid],xmid,xlow) * np.sqrt(2 * np.pi) * xlow,
                       norm.pdf(xgrid[xgrid >= xmid],xmid,xupp) * np.sqrt(2 * np.pi) * xupp)
    x_dist/= trapz(x_dist,x=xgrid)

    return x_dist

def sample_measurement(value,unc,nsamples=20000):

    if (len(unc) == 2):
        value_grid = np.linspace(value - 4 * unc[1], value + 4 * unc[0],600)
        sampled_values = sample_distribution(twosided_gaussian(value,unc[0],unc[1],xgrid=value_grid),
                                             value_grid,nsamples=nsamples)
    elif (len(unc) == 1):
        sampled_values = norm.(value,unc).rvs(nsamples)


    return sampled_values
        

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
    
    
