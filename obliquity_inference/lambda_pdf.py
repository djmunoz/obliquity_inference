import numpy as np
from math import sqrt, cosh, sinh, tan, cos, pi
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")




def lambda_integrand(y, k, l):
    """

    """
    return cosh(k*y) / sqrt(1-y*y) * y / sqrt(1-(y * y/(1 - y * y)) * tan(l)**2)

def lambda_pdf(l,kappa=1):
    """

    """
    return 2*kappa/(pi * sinh(kappa)) / cos(l)**2 * quad(lambda_integrand,0,cos(l),args=(kappa,l,))[0] 


y = np.linspace(0.000001,0.9999999,200)
p = np.linspace(0.005,300,200)
yy, pp = np.meshgrid(y, p)
z = np.vectorize(lambda_pdf)(yy,pp)
interp = RectBivariateSpline(y,p,z.T,kx=1,ky=1)

def lambda_pdf_interp(l,kappa):
    return interp(l,kappa)
