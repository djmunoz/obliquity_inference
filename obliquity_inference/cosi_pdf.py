from _cosi_pdf import _cosi_pdf
import numpy as np
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings("ignore")

def cosi_pdf(cosi,kappa):

    return _cosi_pdf(cosi,kappa)


y = np.linspace(0.00001,0.9999999,180)
p = np.linspace(0.01,200,140)
yy, pp = np.meshgrid(y, p)
z = np.vectorize(cosi_pdf)(yy,pp)
interp = RectBivariateSpline(y,p,z.T,kx=1,ky=1)

def cosi_pdf_interp(cosi,kappa):
    return interp(cosi,kappa)
