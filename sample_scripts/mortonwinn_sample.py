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
    cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df_mw,Npoints=1000)
    ind = df_mw['Nplanets'] == 1
    cosipdf_singles = np.asarray(cosipdf)[ind].tolist()
    cosipdf_multis = np.asarray(cosipdf)[np.invert(ind)].tolist()


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

    # Compute the cosI posteriors *WITHOUT* approximations
    cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df_mw, Npoints=1000, analytic_approx= False, add_to_dataframe = True)
    obl.compute_inclination_dataframe(df_mw,posterior_list = [cosi_vals,cosipdf])

    
    # Separate into two samples
    ind = df_mw['Nplanets'] == 1
    cosipdf_singles = np.asarray(cosipdf)[ind].tolist()
    cosipdf_multis = np.asarray(cosipdf)[np.invert(ind)].tolist()
    
    for pdf in cosipdf_singles: plt.plot(cosi_vals,pdf,color='b',lw=0.6)
    for pdf in cosipdf_multis: plt.plot(cosi_vals,pdf,color='r',lw=0.6)
    plt.plot([np.nan],[np.nan],color='b',label='singles')
    plt.plot([np.nan],[np.nan],color='r',label='multis')
    plt.legend(loc='upper left')
    plt.xlabel(r'$\cos I_{*,k}$',size=20)
    plt.ylabel(r'PDF   $p(\cos I_{*,k}| D)$',size=18)
    plt.savefig('./mw_inc_post_noapprox.png')
    #plt.show()
    plt.clf()
    
    df_mw.to_csv('./morton2014_processed.csv')

    
    # compute kappa posteriors
    #kappa_vals=np.linspace(0.01,120,420)
    #kappa_vals=np.linspace(0.01,120,30)
    kappa_vals=np.append(np.arange(0.01,30,0.05),np.arange(30,150,0.8))

    # for all targets in the sample
    kappa_loglike_contr = obl.compute_hierachical_likelihood_contributions(kappa_vals,\
                                                                           obl.cosi_pdf_interp,cosi_vals,cosipdf,\
                                                                           k_samples = False)#,K = 1000)

    # add these results to the dataframe
    
    
    #kappa_post_all = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf,cosi_vals,\
    #                                                       cosi_pdf_function = obl.cosi_pdf_interp,\
    #                                                       K=1000)
    #kappa_post_singles = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf_singles,\
    #                                                           cosi_vals,cosi_pdf_function = obl.cosi_pdf_interp,\
    #                                                           K=1000)
    #kappa_post_multis = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf_multis,cosi_vals,\
    #                                                          cosi_pdf_function = obl.cosi_pdf_interp,\
    #                                                          K=1000)

    kappa_post_all = np.exp(kappa_loglike_contr.sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
    kappa_post_singles = np.exp(kappa_loglike_contr[:,ind].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
    kappa_post_multis = np.exp(kappa_loglike_contr[:,np.invert(ind)].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
    
    # normalize
    kappa_post_all /= trapz(kappa_post_all,x=kappa_vals)
    kappa_post_singles /= trapz(kappa_post_singles,x=kappa_vals)
    kappa_post_multis /= trapz(kappa_post_multis,x=kappa_vals)

    # compute probability intervals
    kappa_mid_all,kappa_upp_all,kappa_low_all = obl.measure_interval(kappa_post_all,x=kappa_vals)
    kappa_mid_singles,kappa_upp_singles,kappa_low_singles = obl.measure_interval(kappa_post_singles,x=kappa_vals)
    kappa_mid_multis,kappa_upp_multis,kappa_low_multis = obl.measure_interval(kappa_post_multis,x=kappa_vals)

    
    # Compute the squared Hellinger distance
    hel_dist = obl.hellinger_distance(kappa_post_singles,kappa_post_multis,kappa_vals)
    # and the associated statistical significance
    size1, size2 = len(cosipdf_singles), len(cosipdf_multis)
    indices = np.random.permutation(np.append(np.ones(size1),np.zeros(size2)).astype(bool))
    draws = 5000
    hellinger_list = np.zeros(draws)
    for jj in range(draws):
        indices = np.random.permutation(indices)
        kappa_post_groupa = np.exp(kappa_loglike_contr[:,indices].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
        kappa_post_groupb = np.exp(kappa_loglike_contr[:,np.invert(indices)].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
        kappa_post_groupa /= trapz(kappa_post_groupa,x=kappa_vals)
        kappa_post_groupb /= trapz(kappa_post_groupb,x=kappa_vals)
        hellinger_list[jj]= obl.hellinger_distance(kappa_post_groupa,kappa_post_groupb,kappa_vals)

    significance = (1.0 - hellinger_list[hellinger_list > hel_dist].shape[0] * 1.0 / draws) * 100
    print  hellinger_list[hellinger_list > hel_dist].shape[0] * 1.0 / draws, 1-  hellinger_list[hellinger_list > hel_dist].shape[0] * 1.0 / draws
    
    # plot everything
    plt.plot(kappa_vals,kappa_post_all,color='k',lw=2.0,label='all')
    plt.plot(kappa_vals,kappa_post_singles,color='b',label='singles')
    plt.plot(kappa_vals,kappa_post_multis,color='r',label='multis')


    plt.text(0.1,0.65,r'$\kappa=%.2f^{+%.2f}_{-%.2f}$'\
             % (kappa_mid_all,kappa_upp_all-kappa_mid_all,kappa_mid_all-kappa_low_all),size=20,
             transform = plt.axes().transAxes)
    plt.text(0.02,0.85,r'$\kappa=%.2f^{+%.2f}_{-%.2f}$'\
             % (kappa_mid_singles,kappa_upp_singles-kappa_mid_singles,kappa_mid_singles-kappa_low_singles),size=20,color='b',
             transform = plt.axes().transAxes)
    plt.text(20,max(kappa_post_multis),r'$\kappa=%.1f^{+%.1f}_{-%.1f}$'\
             % (kappa_mid_multis,kappa_upp_multis-kappa_mid_multis,kappa_mid_multis-kappa_low_multis),size=20,color='r')
             
    plt.text(0.6,0.51,r'two-sample statistics:', transform = plt.axes().transAxes,size=16)
    plt.text(0.6,0.45,r'$\delta_{{\rm H}^2}=%.2f$' % hel_dist, transform = plt.axes().transAxes,size=16)
    plt.text(0.6,0.40,r'significance=%.f%%' % significance, transform = plt.axes().transAxes,size=14)
    
    plt.legend(loc='upper right')
    plt.xlabel(r'$\kappa$',size=18)
    plt.ylabel(r'PDF   $p(\kappa|\{cos I_{*,k}\})$',size=18)
    plt.xlim(0,100)
    plt.ylim(0,0.3)
    plt.savefig('./kappa_posterior_m+w.png')
    #plt.show()
    plt.clf()
    


    

