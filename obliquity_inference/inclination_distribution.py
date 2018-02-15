from numpy import sinh, exp, cos, sin, pi
import numpy as np
from scipy.integrate import cumtrapz,trapz
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
Properties of inclination distributions of spin-orbit misalignment according
to the Fisher concentration distribution.

"""

def twosided_gaussian(xmid,xupp,xlow,xgrid = None):
    if (xgrid is None):
        xgrid = np.linspace(xmid - 4 * xlow, xmid + 4 * xupp,600)
        
    x_dist = np.append(norm.pdf(xgrid[xgrid < xmid],xmid,xlow) * np.sqrt(2 * np.pi) * xlow,
                       norm.pdf(xgrid[xgrid >= xmid],xmid,xupp) * np.sqrt(2 * np.pi) * xupp)
    x_dist /= 0.5 * (np.sqrt(2 * np.pi) * xlow + np.sqrt(2 * np.pi) * xupp)

    return x_dist

def sample_measurement(value,unc,nsamples=20000,positive=False):

    try:
        if (len(unc) == 2): twosided = True
    except TypeError:
        twosided = False
        
    if twosided:
        value_grid = np.linspace(value - 4 * max(unc), value + 4 * max(unc),800)
        sampled_values = sample_distribution(twosided_gaussian(value,unc[0],unc[1],xgrid=value_grid),
                                             value_grid,nsamples=nsamples)
        if positive:
            while (len(sampled_values[sampled_values < 0]) > 0):
                sampled_values[sampled_values < 0] = sample_distribution(twosided_gaussian(value,unc[0],unc[1],xgrid=value_grid),
                                                                         value_grid,nsamples=len(sampled_values[sampled_values < 0]))
    else:
        sampled_values = norm(value,unc).rvs(nsamples)
        if positive:
            while (len(sampled_values[sampled_values < 0]) > 0):
                sampled_values[sampled_values < 0] = norm(value,unc).rvs(len(sampled_values[sampled_values < 0]))  

    return sampled_values
        

def sample_distribution(dist_grid,val_grid,nsamples=1,max_value=None):
    """ 
    Nsamples from a tabulated posterior 
    """
    cdf = cumtrapz(dist_grid,x=val_grid)
    cdf /= max(cdf)
    u = np.random.random(size=nsamples)
    inds = np.digitize(u,cdf)
    vals= np.asarray(val_grid)[inds]

    if (max_value is not None):
        vals[vals > max_value] = max_value
        
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
    
    
