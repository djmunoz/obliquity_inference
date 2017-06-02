import pandas as pd
import numpy as np
import obliquity_inference as obl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RectBivariateSpline

if __name__ == "__main__":

    #create 100 stars oriented randomly
    Nstars = 100
    cosi = np.random.random(Nstars)
    lamb = np.random.random(Nstars) * 2 * np.pi
    periods = np.random.rayleigh(3, Nstars) * 86400 # in seconds
    radii = np.random.normal(1.0,0.2, Nstars) * 6.957e5 # in kms
    
    # compute observables
    veq = 2 * np.pi * radii / periods
    vsini = veq * np.sqrt(1 - cosi * cosi)
    
    # add uncertainties
    dveq = np.random.normal(0.5,0.1,Nstars)
    dvsini = np.random.normal(0.5,0.1,Nstars)
    for i in range(Nstars):
        veq[i]+= np.random.normal(0.0,dveq[i],1)
        vsini[i]+= np.random.normal(0.0,dvsini[i],1)

    # Create a dataframe
    df_synth = pd.DataFrame(np.array([vsini,dvsini,veq,dveq,dveq]).T,\
                            columns=['Vsini','dVsini','Veq','dVeq_plus','dVeq_minus'],\
                            index=np.arange(Nstars)+1)

    # Compute the posterior inclination from the observed data
    cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df_synth,Npoints=400)
    #Thus, you can plot these posteriors

    #for pdf in cosipdf: plt.plot(cosi_vals,pdf,color='b',lw=0.6)
    
    #plt.xlabel(r'$\cos I_{*,k}$',size=20)
    #plt.ylabel(r'PDF   $p(\cos I_{*,k}| D)$',size=18)
    #plt.show()



    
    y = np.linspace(0.00001,0.9999999,180)
    p = np.linspace(0.01,200,100)
    yy, pp = np.meshgrid(y, p)
    z = np.vectorize(obl.cosi_pdf)(yy,pp)
    cosi_pdf_interp = RectBivariateSpline(y,p,z.T,kx=1,ky=1)

    kappa_vals=np.linspace(0.01,10,100)
    kappa_post = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf,cosi_vals,cosi_pdf_function=cosi_pdf_interp)

    print kappa_post
    
    plt.plot(kappa_vals,kappa_post)
    plt.show()
