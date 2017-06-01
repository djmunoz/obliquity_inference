import obliquity_inference as obl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import trapz, cumtrapz


filename = '../data/morton2014.csv'

if __name__ == "__main__":

    """
    This code can be used to reproduce the kappa inference calculations of
    Morton & Winn (2014).

    """
    
    # read data
    df_mw = pd.read_csv(filename)

    # add equatorial velocities
    obl.compute_equatorial_velocity_dataframe(df_mw,columns = ['R','dR_plus','dR_minus','Prot','dProt'])
    
    # compute inclination posteriors from data
    cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df_mw,Npoints=200)
    ind = df_mw['Nplanets'] == 1
    cosipdf_singles = np.asarray(cosipdf)[ind].tolist()
    cosipdf_multis = np.asarray(cosipdf)[np.invert(ind)].tolist()
    print np.asarray(cosipdf_singles).shape
    print np.asarray(cosipdf_multis).shape
    #cosi_vals_singles, cosipdf_singles = obl.compute_cosipdf_from_dataframe(df_mw[df_mw['Nplanets'] == 1],Npoints=400)
    #cosi_vals_multis, cosipdf_multis = obl.compute_cosipdf_from_dataframe(df_mw[df_mw['Nplanets'] > 1],Npoints=400)


    # plot the inclination posteriors
    for pdf in cosipdf_singles: plt.plot(cosi_vals,pdf,color='b',lw=0.6)
    for pdf in cosipdf_multis: plt.plot(cosi_vals,pdf,color='r',lw=0.6)
    plt.plot([np.nan],[np.nan],color='b',label='singles')
    plt.plot([np.nan],[np.nan],color='r',label='multis')
    plt.legend(loc='upper left')
    plt.xlabel(r'$\cos I_{*,k}$',size=20)
    plt.ylabel(r'PDF   $p(\cos I_{*,k}| D)$',size=18)
    plt.savefig('./mw_inc_post.png')
    #plt.show()
    plt.clf()

    
    # compute kappa posteriors
    kappa_vals=np.linspace(0.01,100,100)

    kappa_post_all = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf,cosi_vals,\
                                                           cosi_pdf_function = obl.cosi_pdf_interp,\
                                                           K=1000)
    kappa_post_singles = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf_singles,\
                                                               cosi_vals,cosi_pdf_function = obl.cosi_pdf_interp,\
                                                               K=1000)
    kappa_post_multis = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf_multis,cosi_vals,\
                                                              cosi_pdf_function = obl.cosi_pdf_interp,\
                                                              K=1000)

    # normalize
    kappa_post_all /= trapz(kappa_post_all,x=kappa_vals)
    kappa_post_singles /= trapz(kappa_post_singles,x=kappa_vals)
    kappa_post_multis /= trapz(kappa_post_multis,x=kappa_vals)

    plt.plot(kappa_vals,kappa_post_all,color='k',lw=2.0,label='all')
    plt.plot(kappa_vals,kappa_post_singles,color='b',label='singles')
    plt.plot(kappa_vals,kappa_post_multis,color='r',label='multis')
    plt.legend(loc='upper right')
    plt.xlabel(r'$\kappa$',size=18)
    plt.ylabel(r'PDF   $p(\kappa|\{cos I_{*,k}\})$',size=18)
    plt.xlim(0,100)
    plt.show()

    
