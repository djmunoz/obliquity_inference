from _cosi_pdf import _cosi_pdf
import numpy as np
from scipy.interpolate import RectBivariateSpline
import os
import warnings
warnings.filterwarnings("ignore")
from obliquity_inference import data_dir

def cosi_pdf(cosi,kappa):

    return _cosi_pdf(cosi,kappa)


def recompute_cosi_pdf(cosi_array=None,kappa_array=None,save=False):
    """
    Compute a tabulated version of the cosi-given-kappa conditional PDF
    for later interpolation.

    """
    
    if (cosi_array is None):
        y = np.linspace(0.000001,0.9999999,200)
    else:
        y = cosi_array
    if (kappa_array is None):
        p = np.append(np.logspace(-2.5,1.7,250),np.linspace(51,221,170))
    else:
        p = kappa_array
        
    yy, pp = np.meshgrid(y, p)
    z = np.vectorize(cosi_pdf)(yy,pp)
    interp_cosi = RectBivariateSpline(y,p,z.T,kx=1,ky=1)

    if save:
        table = np.vstack([np.append(np.nan,y),np.vstack([p.T,z.T]).T])
        np.savetxt(os.path.join(data_dir,'cosi_pdf_data.txt'),table)

    return interp_cosi


def cosi_pdf_interp(cosi,kappa):
    return interp_cosi(cosi,kappa)



if (os.path.isfile(os.path.join(data_dir,'cosi_pdf_data.txt'))):
    cosi_pdf_data = np.loadtxt(os.path.join(data_dir,'cosi_pdf_data.txt'))
    y = cosi_pdf_data[0,1:]
    p = cosi_pdf_data[1:,0]
    z = cosi_pdf_data[1:,1:]
    interp_cosi = RectBivariateSpline(y,p,z.T,kx=1,ky=1)
else:
    interp_cosi = recompute_cosi_pdf(save = True)
