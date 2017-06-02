import numpy as np
import pandas as pd
import scipy.special as spec
from scipy.integrate import quad, cumtrapz,trapz
from scipy.stats.distributions import norm
from scipy.stats import gaussian_kde
from hierarchical_inference import update_progress

"""
Bayesian inference routines for the line-of-star inclination of a single star
or of a collection of stars on an object-by-object basis

"""

ALPHA = 0.23
Rsun_in_km = 6.957e5
day_in_seconds = 86400.0


def measure_interval(xpdf,x,sigma=1.0):
    
    pdf = xpdf[:]/trapz(xpdf,x=x)

    prob_res_min,prob_res_max = 1.0/pdf.shape[0], np.abs(np.diff(pdf)).max()
    prob_res = prob_res_min
    
    mode = pdf.max()
    cumu = cumtrapz(pdf,x=x,initial=0)
    dist_prob = np.abs(np.subtract.outer(cumu,cumu))
    dist_x = np.abs(np.subtract.outer(x,x))

    
    enclosed_prob = round(spec.erf(sigma/np.sqrt(2)),4)
    dist_prob[(dist_prob > (enclosed_prob + prob_res)) | (dist_prob < (enclosed_prob-prob_res))] = 0.0
    while (len(dist_x[dist_prob != 0.0]) == 0):
        prob_res *= 1.05
        dist_prob[(dist_prob > (enclosed_prob + prob_res)) | (dist_prob < (enclosed_prob-prob_res))] = 0.0
        if (prob_res > 0.5 * prob_res_max): return np.nan, np.nan,np.nan
        
    xind = np.where((dist_x == dist_x[dist_prob != 0.0].min()) & (dist_prob != 0.0))
    low, upp = x[xind[0][0]], x[xind[1][0]]
    MODE_THRESHOLD = 0.98
    enclosed = pdf > MODE_THRESHOLD * mode

    try:
        mid = (trapz(x[enclosed]*pdf[enclosed],x=x[enclosed])/trapz(pdf[enclosed],x=x[enclosed]))
    except RuntimeWarning:
        try:
            MODE_THRESHOLD = 0.95
            enclosed = pdf > MODE_THRESHOLD * mode
            mid = (trapz(x[enclosed]*pdf[enclosed],x=x[enclosed])/trapz(pdf[enclosed],x=x[enclosed]))
        except RuntimeWarning:
            mid = x[pdf == pdf.max()]
            
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
    
    p_vals = norm(Pval,Perr).rvs(N)
    while (len(p_vals[p_vals < 0]) > 0):
           p_vals[p_vals < 0] =  norm(Pval,Perr).rvs(len(p_vals[p_vals < 0]))
    try:
       n = len(Rerr)
       if (n >1):
           r_vals_upp = norm(Rval,Rerr[0]).rvs(N)
           r_vals_upp =  r_vals_upp[r_vals_upp >= Rval] 
           r_vals_low = norm(Rval,Rerr[1]).rvs(N)
           while (len(r_vals_low[r_vals_low < 0]) > 0):
               r_vals_low[r_vals_low < 0] =  norm(Rval,Rerr[1]).rvs(len(r_vals_low[r_vals_low < 0]))
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

def compute_equatorial_velocity_single(P,dP,R,dR,from_sample=True):

    """
    Given measurements (and uncertainties) of stellar rotation period 
    and radii, obtain a derived value (and uncertainty) of the stellar
    equatorial rotation speed.

    """

    veq_vals = sample_veq_vals(P,dP,R,dR)
    if (from_sample):
        veq_mid, veq_upp, veq_low = np.percentile(veq_vals, [50, 84, 16],axis= 0).tolist()
        #bins = np.arange(min(veq_vals),max(veq_vals),(veq_upp - veq_low) / 10)
        #hist, bin_edges = np.histogram(veq_vals,bins=bins)
        #veq_mid = bins[hist == hist.max()][0]
    else:
        vgrid = np.linspace(max(0,veq_vals.min()),min(400,veq_vals.max()),800)
        vpdf = gaussian_kde(veq_vals,bw_method=0.4).evaluate(vgrid)
        veq_mid, veq_upp, veq_low = measure_interval(vpdf,vgrid)
        
    veq_upper_err, veq_lower_err = veq_upp - veq_mid, veq_mid - veq_low
    
    return veq_mid,veq_upper_err, veq_lower_err


def inclination_value_from_cosi_posterior(cosi_vals,cosi_posterior, normalized = False):

    if (normalized is False):
        normalize = trapz(cosi_posterior,x=cosi_vals)
        cosi_posterior[:]/=normalize
    
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
    cosi_arr = 1.0-np.logspace(0,-4,Npoints)
    #cosi_arr = np.linspace(0.0,0.99999999,Npoints)

    if (analytic_approx):
        post = np.asarray([posterior_cosi_analytic(c,Vsini,dVsini,Veq,dVeq) for c in cosi_arr])
    else:
        post = np.asarray([posterior_cosi_full(c,Vsini,dVsini,Veq,dVeq) for c in cosi_arr])

        
    I,dI_plus,dI_minus,I_ul =  inclination_value_from_cosi_posterior(cosi_arr,post)

    
    return I,dI_plus,dI_minus,I_ul,post



def compute_equatorial_velocity_dataframe(df,columns = None):

    """
    Compute equatorial velocities -- and uncertainties --  for a collection of targets
    collected in a Pandas dataframe.

    """

    if (columns is None):
        columns = ['R','dR_plus','dR_minus','Prot','dProt']

    df['Veq'] = pd.Series(np.zeros(df.shape[0]))
    df['dVeq_plus'] = pd.Series(np.zeros(df.shape[0]))
    df['dVeq_minus'] = pd.Series(np.zeros(df.shape[0]))
    
    for index,row in df.iterrows():
        R0 = float(row[columns[0]])
        dR0 = [float(row[columns[1]]),abs(float(row[columns[2]]))]
        P0 = float(row[columns[3]])
        dP0 = float(row[columns[4]])
        Veq, dVeq_plus, dVeq_minus = compute_equatorial_velocity_single(P0,dP0,R0,dR0)
        df.set_value(index, 'Veq', Veq)
        df.set_value(index, 'dVeq_plus', dVeq_plus)
        df.set_value(index, 'dVeq_minus', dVeq_minus)
        
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
            post = posterior_list[1][kk] # assuming data frame and posterior list are index-aligned
            I,dI_plus,dI_minus,I_ul95 = inclination_value_from_cosi_posterior(posterior_list[0],post)
        else:
            # check if the posterior is in the dataframe already
            if (('cosi_arr' in df) & ('cosi_pdf' in df)):
                I,dI_plus,dI_minus,I_ul95 = inclination_value_from_cosi_posterior(row['cosi_arr'],row['cosi_pdf'])
            else:
                # otherwise recompute them
                Vsini0 = row[columns[0]]
                dVsini0 = row[columns[1]]
                Veq0 = row[columns[2]]
                dVeq0 = np.sqrt(row[columns[3]]+row[columns[4]])
                I,dI_plus,dI_minus,I_ul95, post = compute_inclination_single(Vsini0,dVsini0,Veq0,dVeq0,Npoints=Npoints)
            
        df.set_value(index, 'I', I)
        df.set_value(index, 'dI_plus', dI_plus)
        df.set_value(index, 'dI_minus', dI_minus)
        df.set_value(index, 'I_ul95', I_ul95)
        kk+=1




#def posterior_cosi_full(cosi,vsini_dist,veq_dist):
def posterior_cosi_full(cosi,vsini_dist_grid,veq_dist_grid,varr):

    #def cos_like_integrand(v):
    #    y = v / np.sqrt(1.0 - cosi * cosi)
    #    return vsini_dist(v) * veq_dist(y)
    #
    #post = quad(cos_like_integrand,0.0,np.inf,epsrel=1.e-8,limit=200)[0]

    print cosi
    post = trapz(vsini_dist_grid * veq_dist_grid, x = varr,axis=1)
    
    return post


def posterior_cosi_analytic(cosi,vsini_val,vsini_err,veq_val,veq_err):

    A = 1.0/np.sqrt(2 * np.pi * (vsini_err**2 / (1 - cosi**2) + veq_err**2)) * np.exp(-(vsini_val - veq_val*np.sqrt(1 - cosi**2))**2/2/(vsini_err**2 + veq_err**2*(1 - cosi**2))) 
    v_bar = (vsini_val * veq_err**2 * np.sqrt(1.0 - cosi**2) + veq_val * vsini_err**2)/(veq_err**2 * (1 - cosi**2) + vsini_err**2)
    sigma_bar = 1.0 / np.sqrt((1 - cosi**2)/ vsini_err**2 + 1.0 / veq_err**2)
    post = A * 0.5 * (1.0 + spec.erf(v_bar/np.sqrt(2)/sigma_bar))

    return post





def  compute_cosipdf_from_dataframe(df, columns = None, analytic_approx=True, Npoints = 200,add_to_dataframe = False):
    

    if (columns is None):
        columns = ['Vsini','dVsini','Veq','dVeq_plus','dVeq_minus','Prot','dProt','R','dR_plus','dR_minus']
    
    cosi_arr = np.linspace(0.0,0.9999999999,Npoints)
    
    post_list = []

    if (add_to_dataframe):
        if ('cosi_arr' not in df): df['cosi_arr'] = pd.Series(np.empty(df.shape[0])).astype(object)
        if ('cosi_pdf' not in df): df['cosi_pdf'] = pd.Series(np.empty(df.shape[0])).astype(object)

    jj = -1
    Nentries = df.shape[0]
    for index,row in df.iterrows():
        jj+=1
        update_progress(jj,Nentries,tag=' (cosI post.)')
        
        vs = float(row[columns[0]])
        dvs = float(row[columns[1]])
    
        if (analytic_approx):
            veq = float(row[columns[2]])
            dveq = float(np.sqrt(0.5*(row[columns[3]]**2 + row[columns[4]]**2)))
            post = np.asarray([posterior_cosi_analytic(c,vs,dvs,veq,dveq) for c in cosi_arr])
        else:
            Prot = float(row[columns[5]])
            dProt = float(row[columns[6]])
            R = float(row[columns[7]])
            dR = [float(row[columns[8]]),float(row[columns[9]])]
            veq_vals = sample_veq_vals(Prot,dProt,R,dR)
            veq_dist = gaussian_kde(veq_vals,bw_method='scott').evaluate
            def vsini_dist(x): return norm.pdf(x,vs,dvs)
            Nvelpoints = 400
            varr = np.linspace(0,vs+7*dvs,Nvelpoints)
            vsiniarr, carr = np.meshgrid(varr,cosi_arr)
            veqarr = (vsiniarr/ np.sqrt(1.0 - carr * carr))
            veq_dist_grid = np.zeros([Npoints,Nvelpoints])
            vsin_dist_grid = np.zeros([Npoints,Nvelpoints])
            for kk in range(Npoints):
                veq_dist_grid[kk,:] = veq_dist(veqarr[kk,:].flatten())
                vsin_dist_grid[kk,:] = norm.pdf(vsiniarr[kk,:].flatten())
            print veq_dist_grid.shape, vsin_dist_grid.shape
            post = trapz(veq_dist_grid * vsin_dist_grid,x=vsiniarr, axis=1)
            print post.shape
            
            #post = posterior_cosi_full(carr,vsini_dist(vsiniarr),veq_dist(veqarr),vsiniarr)


        #post[:]/=trapz(post,x=cosi_arr)
        post_list.append(post)

        
        if (add_to_dataframe):
            df.set_value(index, 'cosi_arr', cosi_arr.flatten().tolist())
            df.set_value(index, 'cosi_pdf', post.flatten().tolist())
        
        
    return cosi_arr, post_list
