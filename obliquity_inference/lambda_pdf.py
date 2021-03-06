import numpy as np
import os
from math import sqrt, cosh, sinh, tan, cos, pi, exp
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")
from obliquity_inference import data_dir

def lambda_integrand(y, k, l):
    """

    """
    return exp(k*y) / sqrt(1 - y * y) * y / sqrt(1 - tan(l)**2  * y**2 / (1- y**2))

def lambda_pdf(l,kappa=1):
    """

    """
    return kappa/(pi * sinh(kappa)) / cos(l)**2 * quad(lambda_integrand,0,cos(l),args=(kappa,l,))[0] 





def recompute_lambda_pdf(lambda_array=None,kappa_array=None,save=False):
    """
    Compute a tabulated version of the lambda-given-kappa conditional PDF
    for later interpolation.

    """
    
    if (lambda_array is None):
        y = np.linspace(-np.pi,np.pi,200)
    else:
        y = lambda_array
    if (kappa_array is None):
        #p = np.append(np.logspace(-2.5,1.9,400),np.linspace(80,600,600))
        p = np.append(np.linspace(0.001,79.5,400),np.linspace(80,600,600))
    else:
        p = kappa_array
        
    yy, pp = np.meshgrid(y, p)
    z = np.vectorize(lambda_pdf)(yy,pp)
    interp_lambda = RectBivariateSpline(y,p,z.T,kx=1,ky=1)

    if save:
        table = np.vstack([np.append(np.nan,y),np.vstack([p.T,z.T]).T])
        np.savetxt(os.path.join(data_dir,'lambda_pdf_data.txt'),table)

    return interp_lambda


def lambda_pdf_interp(l,kappa):
    return interp_lambda(l,kappa)





if (os.path.isfile(os.path.join(data_dir,'lambda_pdf_data.txt'))):
    lambda_pdf_data = np.loadtxt(os.path.join(data_dir,'lambda_pdf_data.txt'))
    y = lambda_pdf_data[0,1:]
    p = lambda_pdf_data[1:,0]
    z = lambda_pdf_data[1:,1:]
    interp_lambda = RectBivariateSpline(y,p,z.T,kx=1,ky=1)
else:
    interp_lambda = recompute_lambda_pdf(save = True)


