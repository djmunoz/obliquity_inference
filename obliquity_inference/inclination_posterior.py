import numpy as np
import pandas as pd
from string import split
from scipy.special import erf
from scipy.integrate import quad, cumtrapz,trapz
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from sklearn.neighbors import KernelDensity
from significance import hellinger_distance
from hierarchical_inference import update_progress
import matplotlib.pyplot as plt

"""
Bayesian inference routines for the line-of-star inclination of a single star
or of a collection of stars on an object-by-object basis

"""

ALPHA = 0.23
Rsun_in_km = 6.957e5
day_in_seconds = 86400.0

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)

def find_weighted_mode(f,y):
    MODE_THRESHOLD = 0.98
    mode = f.max()
    enclosed = f > MODE_THRESHOLD * mode
    
    try:
        wmode = (trapz(y[enclosed]*f[enclosed],x=y[enclosed])/trapz(f[enclosed],x=y[enclosed]))
    except RuntimeWarning:
        try:
            MODE_THRESHOLD = 0.95
            enclosed = f > MODE_THRESHOLD * mode
            wmode = (trapz(y[enclosed]*f[enclosed],x=y[enclosed])/trapz(f[enclosed],x=y[enclosed]))
        except RuntimeWarning:
            wmode = y[f == f.max()][0]

    if (np.isnan(wmode)): wmode = y[f == f.max()][0]
    return wmode


def measure_interval_from_sample(sample,sigma=1.0,style='enclosed'):

    sorted_sample = np.sort(sample)
    cumu = np.arange(len(sorted_sample)) * 1.0 / len(sorted_sample)
    valmax, valmin = min(sorted_sample[cumu > 0.995]), max(sorted_sample[cumu < 0.005])
    Npoints = 600
    val_arr = np.linspace(0.1 * valmin , 2 * valmax,Npoints)
    val_pdf = gaussian_kde(sorted_sample,bw_method='scott').evaluate(val_arr)

    return measure_interval(val_pdf,x=val_arr,sigma=sigma,style=style)

def measure_interval(xpdf,x=None,sigma=1.0,style='enclosed'):

    if np.any(np.isnan(xpdf)):
        x = x[np.isfinite(xpdf)]
        xpdf = xpdf[np.isfinite(xpdf)]
    
    if (x is None):
        x = np.arange(len(xpdf))
    
    pdf = xpdf[:]/trapz(xpdf,x=x)

    prob_res_min,prob_res_max = 1.0/pdf.shape[0], np.abs(np.diff(pdf)).max()
    prob_res = prob_res_min
    
    
    cumu = cumtrapz(pdf,x=x,initial=0)

    enclosed_prob = round(erf(sigma/np.sqrt(2)),4)
    lower_prob,upper_prob = 0.5 - enclosed_prob/2, 0.5 + enclosed_prob/2
     
    
    if (style == 'enclosed'):
        dist_prob = np.abs(np.subtract.outer(cumu,cumu))
        dist_x = np.abs(np.subtract.outer(x,x))
        
        dist_prob[(dist_prob > (enclosed_prob + prob_res)) | (dist_prob < (enclosed_prob-prob_res))] = 0.0
        while (len(dist_x[dist_prob != 0.0]) == 0):
            prob_res *= 1.05
            dist_prob[(dist_prob > (enclosed_prob + prob_res)) | (dist_prob < (enclosed_prob-prob_res))] = 0.0
            if (prob_res > 0.5 * prob_res_max) | (dist_x[dist_prob != 0.0].shape[0] < 30):
                return measure_interval(pdf,x=x,style = 'percentile_mixed')
            
        xind = np.where((dist_x == dist_x[dist_prob != 0.0].min()) & (dist_prob != 0.0))
        low, upp = x[xind[0][0]], x[xind[1][0]]

        mid = find_weighted_mode(pdf,x)
        
    elif (style == 'percentile'):
              
        mid = x[np.abs(cumu - 0.5) == np.abs(cumu - 0.5).min()][0]
        upp = x[np.abs(cumu - upper_prob) == np.abs(cumu - upper_prob).min()][0]
        low = x[np.abs(cumu - lower_prob) == np.abs(cumu - lower_prob).min()][0]

    elif (style == 'percentile_mixed'):

        mid = find_weighted_mode(pdf,x)
        upp = x[np.abs(cumu - upper_prob) == np.abs(cumu - upper_prob).min()][0]
        low = x[np.abs(cumu - lower_prob) == np.abs(cumu - lower_prob).min()][0]

    elif (style == 'height_from_max'):

        height = pdf.max() * np.exp(-sigma**2*0.5)/np.exp(0)
        
        mid = find_weighted_mode(pdf,x)
        upp = x[x > mid][np.abs(pdf[x > mid] - height) == np.abs(pdf[x > mid] -  height).min()][0]
        low = x[x < mid][np.abs(pdf[x < mid] - height) == np.abs(pdf[x < mid] -  height).min()][0]
        
    return mid, upp, low


def sample_veq_vals(Pval,Perr,Rval,Rerr,N=20000):
    """
    From (double sided) Gaussian distributions in rotational period and stellar radius,
    compute a PDF of the stellar equatorial velocity

    Inputs:

    - Pval (FLOAT): Measured period (expectation value of the period PDF) in DAYS.

    - Perr (FLOAT or FLOAT list): uncertainty in the measured period (standard deviation in the period PDF)

    - Rval (FLOAT): Measured/inferred stellar radius (expectation value of the radius PDF) in SOLAR RADII.

    - Rerr (FLOAT or FLOAT list): One-sided (FLOAT) or two-sided (two-element list) radius uncertainties.
    
    Output:

    - Numpy array of size N containing the Monte-Carlo sampled values of the equatorial velocity
    distribution

    """
    
    try:
       n = len(Perr)
       if (n >1):
           p_vals_upp = norm(Pval,Perr[0]).rvs(N)
           p_vals_upp =  p_vals_upp[p_vals_upp >= Pval] 
           p_vals_low = norm(Pval,np.abs(Perr[1])).rvs(N)
           while (len(p_vals_low[p_vals_low < 0]) > 0):  p_vals_low[p_vals_low < 0] =  norm(Pval,np.abs(Perr[1])).rvs(len(p_vals_low[p_vals_low < 0]))
           p_vals_low =  p_vals_low[p_vals_low < Pval]
           p_vals = np.random.choice(np.append(p_vals_upp,p_vals_low),N)
    except TypeError:
        p_vals = norm(Pval,Perr).rvs(N)
        while (len(p_vals[p_vals < 0]) > 0):
            p_vals[p_vals < 0] =  norm(Pval,Perr).rvs(len(p_vals[p_vals < 0]))

           
    try:
       n = len(Rerr)
       if (n >1):
           try:
               r_vals_upp = norm(Rval,Rerr[0]).rvs(N)
           except ValueError:
               print Rval,Rerr[0]
           r_vals_upp =  r_vals_upp[r_vals_upp >= Rval] 
           r_vals_low = norm(Rval,np.abs(Rerr[1])).rvs(N)
           while (len(r_vals_low[r_vals_low < 0]) > 0):
               r_vals_low[r_vals_low < 0] =  norm(Rval,np.abs(Rerr[1])).rvs(len(r_vals_low[r_vals_low < 0]))
           r_vals_low =  r_vals_low[r_vals_low < Rval]
           r_vals = np.random.choice(np.append(r_vals_upp,r_vals_low),N)
    except TypeError:
        r_vals = norm(Rval,Rerr).rvs(N)
        while (len(r_vals[r_vals < 0]) > 0):
            r_vals[r_vals < 0] =  norm(Rval,Rerr).rvs(len(r_vals[r_vals < 0]))
    
    # starspot distribution
    l_vals = norm(20.0,20.0).rvs(N) * np.pi/180.0 
    peq_vals = p_vals * (1 - ALPHA * np.sin(l_vals)**2)

    v_vals = 2 * np.pi * r_vals / peq_vals * Rsun_in_km/ day_in_seconds

    return v_vals

def compute_equatorial_velocity_single(P,dP,R,dR,percentiles = False, symmetric = False):

    """
    Given measurements (and uncertainties) of stellar rotation period 
    and radii, obtain a derived value (and uncertainty) of the stellar
    equatorial rotation speed.

    """

    veq_vals = np.sort(sample_veq_vals(P,dP,R,dR))

    if (symmetric):
        veq_mid,  veq_upper_err, veq_lower_err = veq_vals.mean(),veq_vals.std(),veq_vals.std()
        return veq_mid,veq_upper_err, veq_lower_err
        
    if (percentiles):
        veq_mid, veq_upp, veq_low = np.percentile(veq_vals, [50, 84, 16],axis= 0).tolist()
        #bins = np.arange(min(veq_vals),max(veq_vals),(veq_upp - veq_low) / 10)
        #hist, bin_edges = np.histogram(veq_vals,bins=bins)
        #veq_mid = bins[hist == hist.max()][0]
    else:
        vcumu = np.arange(len(veq_vals)) * 1.0 / len(veq_vals)
        vmax, vmin = min(veq_vals[vcumu > 0.995]), max(veq_vals[vcumu < 0.005])
        Nvelpoints = 600
        v_arr = np.linspace(0,2*vmax,Nvelpoints)
        vpdf = gaussian_kde(veq_vals,bw_method='scott').evaluate(v_arr)
        veq_mid, veq_upp, veq_low = measure_interval(vpdf,x=v_arr)
        veq_ul95 = veq_vals[vcumu > 0.95][0]
        
    if (np.isnan(veq_mid)):
        plt.hist(veq_vals,bins=30)
        plt.show()
    
    veq_upper_err, veq_lower_err = veq_upp - veq_mid, veq_mid - veq_low


    
    return veq_mid,veq_upper_err, veq_lower_err,veq_ul95


def compute_inclination_value_from_cosi_posterior(cosi_vals,cosi_posterior, normalized = False):

    normalize = trapz(cosi_posterior,x=cosi_vals)
    
    if (normalized != 1.0):
        if (normalize != 0):
            cosi_posterior[:]/=normalize
        else:
            return np.nan,np.nan,np.nan,np.nan
    
            
    mid, upp, low =  measure_interval(cosi_posterior,cosi_vals,sigma=1)
    mode = cosi_vals[cosi_posterior == cosi_posterior.max()][0]
    cum = cumtrapz(cosi_posterior,x=cosi_vals,initial=0)
    lim95 = cosi_vals[(cum >= 0.05)][0]
    
    I = round(np.arccos(mid)*180.0/np.pi,3)
    dI_minus = I - round(np.arccos(upp)*180.0/np.pi,3)
    dI_plus = round(np.arccos(low)*180.0/np.pi,3) - I
    I_ul = round(np.arccos(lim95)*180.0/np.pi,3)

        
    return I,dI_plus,dI_minus,I_ul


def compute_inclination_single(Vsini,dVsini,Veq,dVeq,analytic_approx = True,Npoints=600):
    
    """
    Obtain a posterior probability distribution given the data
    in Vsini and Veq. From the posterior, obtain representative
    values from confidence intervals.
    
    """
    cosi_arr = 1.0-np.logspace(0,-5,Npoints)
    #cosi_arr = np.linspace(0.0,0.99999999,Npoints)

    if (analytic_approx):
        post = np.asarray([posterior_cosi_analytic(c,Vsini,dVsini,Veq,dVeq) for c in cosi_arr])
    else:
        post = np.asarray([posterior_cosi_full(c,Vsini,dVsini,Veq,dVeq) for c in cosi_arr])

        
    I,dI_plus,dI_minus,I_ul =  compute_inclination_value_from_cosi_posterior(cosi_arr,post)

    
    return I,dI_plus,dI_minus,I_ul,post



def compute_equatorial_velocity_dataframe(df,radius_columns = None,period_columns = None,
                                          percentiles=False):

    """
    Compute equatorial velocities -- and uncertainties --  for a collection of targets
    collected in a Pandas dataframe.

    """

    if (radius_columns is None):
        radius_columns = ['R','dR_plus','dR_minus']
    if (period_columns is None):
        period_columns = ['Prot','dProt_plus','dProt_minus']
    
        
    df['Veq'] = pd.Series(np.zeros(df.shape[0]))
    df['dVeq_plus'] = pd.Series(np.zeros(df.shape[0]))
    df['dVeq_minus'] = pd.Series(np.zeros(df.shape[0]))
    df['Veq_ul95'] = pd.Series(np.zeros(df.shape[0]))

    jj = -1
    Nentries = df.shape[0]
    for index,row in df.iterrows():
        jj+=1
        update_progress(jj,Nentries,tag=' (eq. vel.)')
        R0 = float(row[radius_columns[0]])
        if len(radius_columns) == 2:
            dR0 = float(row[radius_columns[1]])
        elif len(radius_columns) == 3:
            dR0 = [float(row[radius_columns[1]]),abs(float(row[radius_columns[2]]))]
        P0 = float(row[period_columns[0]])
        if len(period_columns) == 2:
            dP0 = float(row[period_columns[1]])
        elif len(period_columns) == 3:
            dP0 = [float(row[period_columns[1]]),abs(float(row[period_columns[2]]))]

        Veq, dVeq_plus, dVeq_minus, Veq_ul95 = compute_equatorial_velocity_single(P0,dP0,R0,dR0,percentiles = percentiles)
        df.set_value(index, 'Veq', Veq)
        df.set_value(index, 'dVeq_plus', dVeq_plus)
        df.set_value(index, 'dVeq_minus', dVeq_minus)
        df.set_value(index, 'Veq_ul95', Veq_ul95)
        
    return None


def compute_inclination_dataframe(df, columns = None, posterior_list = None, Npoints=600):

    if (columns is None):
        columns = ['Vsini','dVsini','Veq','dVeq_plus','dVeq_minus']
    
    if ('I' not in df): df['I'] = pd.Series(np.zeros(df.shape[0]))
    if ('dI_plus' not in df): df['dI_plus'] = pd.Series(np.zeros(df.shape[0]))
    if ('dI_minus' not in df): df['dI_minus'] = pd.Series(np.zeros(df.shape[0]))
    if ('I_ul95' not in df): df['I_ul95'] = pd.Series(np.zeros(df.shape[0]))

    kk = 0
    for index,row in df.iterrows():
        if (posterior_list is not None): # if we already have the full posteriors
            post = np.asarray(posterior_list[1][kk]) # assuming data frame and posterior list are index-aligned
            try:
                I,dI_plus,dI_minus,I_ul95 = compute_inclination_value_from_cosi_posterior(posterior_list[0],post)
            except ZeroDivisionError:
                print posterior_list[0]
                print post
                print row[['KOI','Vsini','dVsini','Veq','Prot','dProt_plus','dProt_minus','R','dR_plus','dR_minus']]
                plt.plot(posterior_list[0],post)
                plt.show()
            #if (I_ul95 < 86.0):
            #    plt.plot(posterior_list[0],post)
            #    plt.show()
        else:
            # check if the posterior is in the dataframe already
            if (('cosi_arr' in df) & ('cosi_pdf' in df)):
                I,dI_plus,dI_minus,I_ul95 = compute_inclination_value_from_cosi_posterior(row['cosi_arr'],row['cosi_pdf'])
            else:
                # otherwise recompute them
                Vsini0 = row[columns[0]]
                dVsini0 = row[columns[1]]
                Veq0 = row[columns[2]]
                dVeq0 = np.sqrt(0.5*(row[columns[3]]**2+row[columns[4]]**2))
                I,dI_plus,dI_minus,I_ul95, post = compute_inclination_single(Vsini0,dVsini0,Veq0,dVeq0,Npoints=Npoints)
            
        df.set_value(index, 'I', I)
        df.set_value(index, 'dI_plus', dI_plus)
        df.set_value(index, 'dI_minus', dI_minus)
        df.set_value(index, 'I_ul95', I_ul95)
        kk+=1




def posterior_cosi_full(cosi,Vsini,dVsini,Prot,dProt,R,dR, simple = True):

    veq_vals = np.sort(sample_veq_vals(Prot,dProt,R,dR))
    vcumu = np.arange(len(veq_vals)) * 1.0 / len(veq_vals)
    vmax, vmin = min(veq_vals[vcumu > 0.995]), max(veq_vals[vcumu < 0.005])
    veq_ul95 = veq_vals[vcumu > 0.95][0]
    Nvelpoints = 700
    try:
        Nvelpoints = max(len(cosi),Nvelpoints)
    except TypeError:
        None    
    v_arr = np.linspace(0,max(3*vmax,Vsini + 5 * dVsini),Nvelpoints)
    v_grid, cosi_grid = np.meshgrid(v_arr,cosi)
    vsini_dist = norm.pdf(v_grid,Vsini/ np.sqrt(1.0 - cosi_grid * cosi_grid),dVsini/ np.sqrt(1.0 - cosi_grid * cosi_grid))
    veq_dist = gaussian_kde(veq_vals,bw_method='scott').evaluate(v_arr)
    integrand = vsini_dist * veq_dist[None,:] * v_arr[None,:]
    post = trapz(integrand,x=v_grid, axis=1)

    if (simple):
        return post
    else:
        return post,[v_arr,veq_dist,veq_ul95],veq_vals


def posterior_cosi_analytic(cosi,vsini_val,vsini_err,veq_val,veq_err):

    A = 1.0/np.sqrt(2 * np.pi * (vsini_err**2 / (1 - cosi**2) + veq_err**2)) * np.exp(-(vsini_val - veq_val*np.sqrt(1 - cosi**2))**2/2/(vsini_err**2 + veq_err**2*(1 - cosi**2))) 
    v_bar = (vsini_val * veq_err**2 * np.sqrt(1.0 - cosi**2) + veq_val * vsini_err**2)/(veq_err**2 * (1 - cosi**2) + vsini_err**2)
    sigma_bar = 1.0 / np.sqrt((1 - cosi**2)/ vsini_err**2 + 1.0 / veq_err**2)
    post = A * 0.5 * (1.0 + erf(v_bar/np.sqrt(2)/sigma_bar))

    return post





def  compute_cosipdf_from_dataframe(df, vsini_columns = None, veq_columns = None, period_columns = None,
                                    radius_columns = None,
                                    analytic_approx=True, Npoints = 200,
                                    smooth =  False,
                                    add_to_dataframe = False,recompute_velocity = False):

    if (vsini_columns is None):
        vsini_columns =  ['Vsini','dVsini']
    if (veq_columns is None):
        veq_columns = ['Veq','dVeq_plus','dVeq_minus','Veq_ul95']
    if (radius_columns is None):
        radius_columns = ['R','dR_plus','dR_minus']
    if (period_columns is None):
        period_columns = ['Prot','dProt_plus','dProt_minus']

    #cosi_arr = np.linspace(0.0,0.9999999999,Npoints)
    #cosi_arr = 1.0-np.logspace(0,-5,Npoints)
    cosi_arr = np.linspace(0.0000,0.9999,Npoints)
    
    post_list = []

    if (veq_columns[0] not in df) & (veq_columns[1] not in df) & (veq_columns[2] not in df):
        recompute_velocity = True
        df[veq_columns[0]] = pd.Series(np.empty(df.shape[0])).astype(object)
        df[veq_columns[1]] = pd.Series(np.empty(df.shape[0])).astype(object)
        df[veq_columns[2]] = pd.Series(np.empty(df.shape[0])).astype(object)
        df[veq_columns[3]] = pd.Series(np.empty(df.shape[0])).astype(object)
        
    if (add_to_dataframe):
        if ('cosi_arr' in df): df.drop('cosi_arr',axis=1,inplace=True)
        if ('cosi_pdf' in df): df.drop('cosi_pdf',axis=1,inplace=True)

        if ('cosi_arr' not in df): df['cosi_arr'] = pd.Series(np.empty(df.shape[0])).astype(object)
        if ('cosi_pdf' not in df): df['cosi_pdf'] = pd.Series(np.empty(df.shape[0])).astype(object)
        
    if (analytic_approx) & (recompute_velocity):
        compute_equatorial_velocity_dataframe(df, radius_columns = radius_columns,
                                              period_columns = period_columns)

    dist_list = []

    
    jj = -1
    Nentries = df.shape[0]
    for index,row in df.iterrows():
        jj+=1
        update_progress(jj,Nentries,tag=' (cosI post.)')
        
        vs = float(row[vsini_columns[0]])
        dvs = float(row[vsini_columns[1]])

        if (analytic_approx):
            veq = float(row[veq_columns[0]])
            dveq = float(0.5*(row[veq_columns[1]] + row[veq_columns[2]]))
            post = np.asarray([posterior_cosi_analytic(c,vs,dvs,veq,dveq) for c in cosi_arr])
        else:
            Prot = float(row[period_columns[0]])
            if (period_columns[1] in df) & (period_columns[2] in df):
                dProt = [float(row[period_columns[1]]),np.abs(float(row[period_columns[2]]))]
            elif (split(period_columns[1],'_')[0] in df):
                dProt = float(row[split(period_columns[1],'_')[0]])
            R = float(row[radius_columns[0]])
            dR = [float(row[radius_columns[1]]),np.abs(float(row[radius_columns[2]]))]

            post,veq_data,veq_vals = posterior_cosi_full(cosi_arr,vs,dvs,Prot,dProt,R,dR,simple = False)
            post[post < 1.0e-5 * np.nanmax(post)] = 0.0 # this is necessary to avoid spurious local maxima
                
            if (recompute_velocity):
                veq_dist, v_arr, veq_ul95= veq_data[1], veq_data[0], veq_data[2]
                veq_mid, veq_upp, veq_low = measure_interval(veq_dist,x=v_arr)
                df.set_value(index, veq_columns[0], veq_mid)
                df.set_value(index, veq_columns[1], veq_upp - veq_mid)
                df.set_value(index, veq_columns[2], veq_mid - veq_low)
                df.set_value(index, veq_columns[3], veq_ul95)

        area = trapz(post,x=cosi_arr)
        #if (area != 0.0):
        #    post[:]/=area
        
        #post_analytic = np.asarray([posterior_cosi_analytic(c,vs,dvs,veq_vals.mean(),veq_vals.std()) for c in cosi_arr])
        #post_analytic[:]/=trapz(post_analytic,x=cosi_arr)

        #dist = hellinger_distance(post,post_analytic,cosi_arr)
        #dist_list.append(dist)
        #if (dist > 0.05):
        #    print row
        #    plt.plot(cosi_arr,post,color='b')
        #    plt.plot(cosi_arr,post_analytic,color='r')
        #    plt.show()
        #    plt.plot(v_arr,norm.pdf(v_arr,veq_vals.mean(),veq_vals.std()),color='b')
        #    plt.plot(v_arr,veq_dist,color='green')
        #    plt.hist(veq_vals,bins=np.linspace(veq_vals.min(),veq_vals.max(),40),normed=True,color='blue')
        #    print veq_vals.mean(),veq_vals.std()
        #    plt.show()
        post_list.append(post)

        
        if (add_to_dataframe):
            df.set_value(index, 'cosi_arr', cosi_arr.flatten().tolist())
            df.set_value(index, 'cosi_pdf', (post.flatten()).tolist())

    return cosi_arr, post_list
