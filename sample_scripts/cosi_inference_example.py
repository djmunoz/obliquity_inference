import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad,trapz
from obliquity_inference import posterior_cosi_analytic

"""
Simple example script to compute the posterior PDF of the cosine of the inclination
angle given measurementes of VsinI and Veq.

D.J. Munoz

"""

def gaussian(x,x0,sigmax):
    return np.exp((x - x0)**2/2/sigmax)/np.sqrt(2 * np.pi) / sigmax
    

def integrand(x,vs,dvs,veq,dveq,cosi):
    return norm.pdf(x, vs / np.sqrt(1 - cosi**2), dvs/ np.sqrt(1 - cosi**2)) * norm.pdf(x, veq, dveq)
    #return gaussian(x, vs / np.sqrt(1 - cosi**2), dvs/ np.sqrt(1 - cosi**2)) * gaussian(x, veq, dveq)

if __name__ == "__main__":

    vs0 = 10.0
    dvs0 = 0.5
    veq0 = 10.0
    dveq0 = 0.5

    
    v_arr = np.linspace(0,20,200)
    cosi_arr = np.linspace(0,0.99999,200)

    # integrating the function object with quad
    cosi_post = [quad(integrand,min(v_arr),max(v_arr),args=(vs0,dvs0,veq0,dveq0,c,))[0]\
                 for c in cosi_arr]
    cosi_post = np.asarray(cosi_post)/trapz(cosi_post,x=cosi_arr)
    plt.plot(cosi_arr[::5],cosi_post[::5],'bo')

    
    # intgrate a tabulated distribution with the trapezoidal rule
    v_grid, cosi_grid = np.meshgrid(v_arr,cosi_arr)
    vsini_dist_grid = norm.pdf(v_grid,vs0/ np.sqrt(1.0 - cosi_grid * cosi_grid),dvs0/ np.sqrt(1.0 - cosi_grid * cosi_grid))
    veq_dist_grid = norm.pdf(v_grid, veq0, dveq0)
    integrand_grid = vsini_dist_grid * veq_dist_grid

    cosi_post = trapz(integrand_grid,x=v_grid, axis=1)
    cosi_post = np.asarray(cosi_post)/trapz(cosi_post,x=cosi_arr)
    plt.plot(cosi_arr,cosi_post,color='g')

    # use the analytic approximation
    cosi_post_analytic = np.asarray([posterior_cosi_analytic(c,vs0,dvs0,veq0,dveq0) for c in cosi_arr])
    cosi_post_analytic = np.asarray(cosi_post_analytic)/trapz(cosi_post_analytic,x=cosi_arr)
    plt.plot(cosi_arr,cosi_post_analytic,color='r',lw=8,alpha=0.5,zorder=0)
    
    plt.show()

    
