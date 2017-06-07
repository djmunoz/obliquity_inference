import numpy as np
import os
from math import sqrt, cosh, sinh, tan, cos, pi
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def lambda_integrand(y, k, l):
    """

    """
    return cosh(k*y) / sqrt(1-y*y) * y / sqrt(1-(y * y/(1 - y * y)) * tan(l)**2)

def lambda_pdf(l,kappa=1):
    """

    """
    return 2*kappa/(pi * sinh(kappa)) / cos(l)**2 * quad(lambda_integrand,0,cos(l),args=(kappa,l,))[0] 





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
        p = np.append(np.logspace(-2.5,1.7,250),np.linspace(51,221,170))
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


