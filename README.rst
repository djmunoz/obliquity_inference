OBLIQUITY INFERENCE
==================================================
.. sectnum::
   
Overview
--------

Installation
--------

As usual, the installation is very simple

.. code::
   
   git clone https://github.com/djmunoz/obliquity_inference.git

   cd obliquity_inference

   sudo python setup.py build
   
   sudo python setup.py install


Basic Tutorial #1: Using *V sin I*
--------

In order to carry out the Bayesian parameter estimation of the concentration parameter "kappa" (Fabrycky & Winn, 2009), you need to do it in three steps.

How to compute *sin I* from observations of *VsinI* and  *P*:sub:`rot`
~~~~~~~~


First, you import the package
   
.. code:: python
	  
   import obliquity_inference as obl

   
If, for a given star, you have *VsinI* and *P*:sub:`rot` measurements (with errors), you can get a probability distribution function (PDF) for the inclination *cosI*.

First, you need to obtain obtain a PDF for the star's equatorial velocity. You can accomplish this
by running

.. code:: python

   veq_vals = obl.sample_veq_vals(P,dP,R,dR,N=20000)

where P and dP, and R and dR, are the period measurement and the stellar radius measurements with their respective uncertainties. If there is only one value of uncertainty for a given emasurement, it assumed that said measurement is distributed normally with mean and dispersion given by the measurement and its error. If there is an 'upper' and 'lower' uncertainty interval (as it is often the case for the radius of Kepler stars)

To compute the inclination PDF, you have two options:

- Using the full PDF of *V*:sub:`eq`

Following the statistical techniques of Morton & Winn (2014), we can compute the PDF of
*cosI* - for a **given** star - by creating an empirical PDF for *V*:sub:`eq`

.. code:: python

   import numpy as np
   from scipy.stats import gaussian_kde
   def veq_dist(x):
	  return gaussian_kde(veq_vals,bw_method=0.1).evaluate(x)
   
.. code:: python
   
   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([posterior_cosi_full(c,vsini_dist,veq_dist)  for c in cosi_arr])


OR...

- Using an analytic approximation

Alternatively, if *both* :code:`vsini_dist` and :code:`veq_dist` can be well approximated by normal distributions,
you can use the analytic approximation of Munoz & Perets (2017)
  
.. code:: python

   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([obl.posterior_cosi_analytic(c,Vsini0,dVsini0,veq_vals.mean(),veq_vals.std()) for c in cosi_arr])

For example, if you have :code:`Vsini=4.0`, :code:`dVsini=1.16`, :code:`Veq=1.12`, :code:`dVeq_plus=0.42`
and :code:`Veq_minus=0.25`, then we can compute

.. code:: python

   Vsini, dVsini = 4.0, 1.16
   Veq, dVeq_plus, dVeq_minus = 1.12, 0.42, 0.25
   dVeq = np.sqrt(dVeq_plus**2 + dVeq_minus**2)

   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([obl.posterior_cosi_analytic(c,Vsini,dVsini,Veq,dVeq) for c in cosi_arr])

   import matplotlib.pyplot as plt
   plt.plot(cosi_arr, post)
   plt.show()


Computing a set of *cosI* PDFs from a CSV file/dataset
~~~~~~~~~

For a collection of stars, you can either save all the inclination posteriors PDFs, or simply save *V*:sub:`eq` (with 68% confidence intervals)
and recompute the inclination PDF using the analytic approximation.

You need to read-in a table/database of stars. For this, we use dataframe objects in the pandas Python package.

.. code:: python
	  
   import pandas as pd

You need a CSV file containing the following columns: 'Vsini', 'dVsini', 'Veq', 'dVeq_plus' and 'dVeq_minus'

.. code:: python
	  
   columns = ['Vsini','dVsini','Veq','dVeq_plus','dVeq_minus']

   
(where the equatorial velocity values are obtained from the measurements of stellar radius and rotation period -- see above). If your CSV file has slightly different columns names, specify them in the columns keyword below
   
.. code:: python
	  
   cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df, columns=columns, analytic_approx=True)

where :code:`cosivals` is a numpy array of cosine values between 0 and 1, and :code:`cosipdf`  is a *list* of numpy arrays, one array per object,
and each one of the same length as :code:`cosivals`.

Let us create a synthetic random (uniform) sample of stellar orientations and save it into a pandas dataframe:

.. code:: python

   import numpy as np
   import obliquity_inference as obl
   
   # create 100 stars oriented randomly
   np.random.seed(42)
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

   
Thus, you can plot these posteriors

.. code:: python

   import matplotlib.pyplot as plt

   for pdf in cosipdf: plt.plot(cosi_vals,pdf/pdf.sum(),color='b',lw=0.4)

   plt.xlabel(r'$\cos I_{*,k}$',size=20)
   plt.ylabel(r'PDF   $p(\cos I_{*,k}| D)$',size=18)
   plt.ylim(0,0.08)
   plt.show()
   

.. class:: no-web
           
   .. image:: example_figures/posteriors_test.png
      :height: 100px
      :width: 200 px
      :scale: 80 %

   
For the Morton & Winn (2014) sample of 70 Kepler stars, the collection of inclnation PDFs looks like:

.. class:: no-web
           
   .. image:: example_figures/inclination_posteriors_m+w.png
      :height: 100px
      :width: 200 px
      :scale: 80 %

Even by eye, the distribution of PDFs is noticeably different respect to the uniform orientation example.
Morton & Winn (2014) found that, except for 12 KOIs (out of 70), the orientation of the stellar spin is consistent
with alignment with the planetary orbit.

The inclination posteriors computed above can be used to derive representative
values of the inclination angle within confidence intervals. These values can be
computed and added to the catalog dataframe by

.. code:: python

   obl.compute_inclination_dataframe(df_synth,posterior_list = [cosi_vals,cosipdf])
   list(df_syth.columns)

which shows that we have added four more columns to the dataframe: :code:`['Vsini', 'dVsini', 'Veq', 'dVeq_plus', 'dVeq_minus', 'I', 'dI_plus', 'dI_minus', 'I_ul95']`.

You can compare the "measured" inclinations with the original ones in the synthetic sample before noise was added.

.. code:: python
	  
   plt.errorbar(np.arccos(cosi)*180.0/np.pi,df_synth['I'],yerr=[df_synth['dI_minus'],df_synth['dI_plus']],fmt='s')
   plt.plot([0,95],[0,95],'k',ls=':',zorder=0)
   plt.axis([0,95,0,95])
   plt.xlabel(r'$I[^\circ]$ (true)',size=18)
   plt.ylabel(r'$I[^\circ]$ (measured)',size=18)
   plt.axes().set_aspect(1.0)
   plt.show()

.. image:: example_figures/measured_inclinations.png

   
Combining MULTIPLE *cosI* PDFs to perform hierarchical Bayesian inference on the "concentration" parameter
~~~~~~~~

The main goal is to compute a posterior PDF for the concentration parameter kappa. To implement the hierarchical Bayesian inference formalism of Hogg et al (2009) one needs a collection of PDFs for the line-of-sight inclination angle *I* (or more conveniently, PDFs for *cosI*; Morton & Winn, 2014).

Uniform distribution
'''''

Let us use the some uniform distribution of stellar spin orientations from section
3.2 above. We can use all these objects (and their respective inclination PDFs) to
derive a PDF for the values of kappa that are consistent with such sample of inclinations.
(this may take a few minutes, although it contains embarrassingly parallelizable loops)

.. code:: python
	  
   kappa_vals=np.linspace(0.001,6,100)
   pdf_func = obl.cosi_pdf_interp
   # alternatively, you could do
   # pdf_func = obl.cosi_pdf
   # but that is MUCH slower
   kappa_post = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf,cosi_vals,cosi_pdf_function = pdf_func) 

   # estimate the confidence intervals
   c = kappa_post.cumsum()/kappa_post.sum()
   kappa_mid = kappa_vals[kappa_post == kappa_post.max()][0]
   kappa_upp = kappa_vals[np.abs(c - 0.84) == np.abs(c - 0.84).min()][0]
   kappa_low = min(0,kappa_vals[np.abs(c - 0.16) == np.abs(c - 0.16).min()][0])
   
   
If you plot the resulting concentration posterior, 

.. code:: python

   plt.plot(kappa_vals,kappa_post)
   plt.text(0.8,0.9,r'$\kappa=%.2f^{+%.2f}_{-%.2f}$' % (kappa_mid,kappa_upp-kappa_mid,kappa_mid-kappa_low),size=20)
   plt.xlabel(r'$\kappa$',size=18)
   plt.ylabel(r'PDF   $p(\kappa|\{cos I_{*,k}\})$',size=18)
   plt.show()

which should give you

.. class:: no-web
           
   .. image:: example_figures/kappa_posterior_uniform.png
      :height: 100px
      :width: 200 px
      :scale: 80 %

which is a nice Gaussian with a maximum near zero, meaning that the data is consistent
with kappa=0, i.e., uniform spin orientations. The greater the number of targets (in this example we are using :code:`Nstars=100`), the narrower the distribution around zero is (you can check this by setting :code:`Nstars=10` and finding that the kappa distribution is still around zero, but wider).

Real-data
'''''

Morton & Winn (2014)
::::::

The Morton & Winn (2014) data sample can be found in the :code:`data/` directory
of this repository.

First load the :code:`vsini`,  :code:`Prot` and :code:`R` data and check the column names

.. code:: python

   df_mw = pd.read_csv('data/morton2014.csv')
   list(df_mw.columns)

and you should get :code:`['name', 'R', 'dR_plus', 'dR_minus', 'Prot', 'dProt', 'Vsini', 'dVsini', 'Nplanets']`
   
Now compute the equatorial velocities


.. code:: python

   obl.compute_equatorial_velocity_dataframe(df_mw,columns = ['R','dR_plus','dR_minus','Prot','dProt'])
   # you can check that new columns have been added
   list(df_mw.columns)

which will give you  :code:`['name', 'R', 'dR_plus', 'dR_minus', 'Prot', 'dProt', 'Vsini', 'dVsini', 'Nplanets', 'Veq', 'dVeq_plus', 'dVeq_minus']`. And now your dataframe has columns corresponding to equatorial velocity.

Next, you compute the inclination posteriors as in Section 3.3.1 above. We can use the
:code:`Nplanets` column to separate the data into "multis" and "singles"
   
.. code:: python
	  
   cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df_mw,Npoints=1000)
   cosi_vals_singles, cosipdf_singles = obl.compute_cosipdf_from_dataframe(df_mw[df_mw['Nplanets'] == 1],Npoints=1000)
   cosi_vals_multis, cosipdf_multis = obl.compute_cosipdf_from_dataframe(df_mw[df_mw['Nplanets'] > 1],Npoints=1000)

   # plot the inclination posteriors
   for pdf in cosipdf_singles: plt.plot(cosi_vals_singles,pdf,color='b',lw=0.6)
   
   for pdf in cosipdf_multis: plt.plot(cosi_vals_multis,pdf,color='r',lw=0.6)

   plt.plot([np.nan],[np.nan],color='b',label='singles')
   plt.plot([np.nan],[np.nan],color='r',label='multis')
   plt.legend(loc='upper left')
   plt.xlabel(r'$\cos I_{*,k}$',size=20)
   plt.ylabel(r'PDF   $p(\cos I_{*,k}| D)$',size=18)
   plt.show()


.. class:: no-web
           
   .. image:: example_figures/mw_inc_post.png

We can also produce the same plot by computing the actual cosI PDFs without the analytic approximation

.. code:: python
	  
   cosi_vals, cosipdf = obl.compute_cosipdf_from_dataframe(df_mw,Npoints=1000, analytic_approx= False)
   cosi_vals_singles, cosipdf_singles = obl.compute_cosipdf_from_dataframe(df_mw[df_mw['Nplanets'] == 1],\
	                                                                   Npoints=1000, analytic_approx= False)
   cosi_vals_multis, cosipdf_multis = obl.compute_cosipdf_from_dataframe(df_mw[df_mw['Nplanets'] > 1], \
	                                                                 Npoints=1000, analytic_approx= False)

   

	      
Hierarchical inference
::::::::

As we did in Section 3.3.1, we can now use all these imclination posteriors to compute a concentration posterior for
:code:`kappa` using the function :code:`compute_kappa_posterior_from_cosI`. Following Morton & Winn (2014), we can do
this analysis for the whole dataset, or separating between single-candidate stars, and multi-candidate stars

  
.. code:: python

   kappa_vals=np.linspace(0.01,100,300)
	  
   kappa_post_all = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf,cosi_vals,cosi_pdf_function = pdf_func,K=1000)
   kappa_post_singles = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf_singles,cosi_vals_singles,cosi_pdf_function = pdf_func,K=1000)
   kappa_post_multis = obl.compute_kappa_posterior_from_cosI(kappa_vals,cosipdf_multis,cosi_vals_multis,cosi_pdf_function = pdf_func,K=1000)

   from scipy.integrate import trapz, cumtrapz
   kappa_post_all /= trapz(kappa_post_all,x=kappa_vals)
   kappa_post_singles /= trapz(kappa_post_singles,x=kappa_vals)
   kappa_post_multis /= trapz(kappa_post_multis,x=kappa_vals)
   
   c = cumtrapz(kappa_post_all, x = kappa_vals)
   kappa_mid_all = kappa_vals[kappa_post_all == kappa_post_all.max()][0]
   kappa_upp_all = kappa_vals[np.abs(c - 0.84) == np.abs(c - 0.84).min()][0]
   kappa_low_all = kappa_vals[np.abs(c - 0.16) == np.abs(c - 0.16).min()][0]

   c = cumtrapz(kappa_post_singles, x = kappa_vals)
   kappa_mid_singles = kappa_vals[kappa_post_singles == kappa_post_singles.max()][0]
   kappa_upp_singles = kappa_vals[np.abs(c - 0.84) == np.abs(c - 0.84).min()][0]
   kappa_low_singles = kappa_vals[np.abs(c - 0.16) == np.abs(c - 0.16).min()][0]

   c = cumtrapz(kappa_post_multis, x = kappa_vals)
   kappa_mid_multis = kappa_vals[kappa_post_multis == kappa_post_multis.max()][0]
   kappa_upp_multis = kappa_vals[np.abs(c - 0.84) == np.abs(c - 0.84).min()][0]
   kappa_low_multis = kappa_vals[np.abs(c - 0.16) == np.abs(c - 0.16).min()][0]

And we can plot the results
   
.. code:: python

   plt.plot(kappa_vals,kappa_post_all,color='k',lw=2.0,label='all')
   plt.plot(kappa_vals,kappa_post_singles,color='b',label='singles')
   plt.plot(kappa_vals,kappa_post_multis,color='r',label='multis')
   plt.text(10,max(kappa_post_all),r'$\kappa=%.2f^{+%.2f}_{-%.2f}$'\
	  % (kappa_mid_all,kappa_upp_all-kappa_mid_all,kappa_mid_all-kappa_low_all),size=20)
   plt.text(4,max(kappa_post_singles),r'$\kappa=%.2f^{+%.2f}_{-%.2f}$'\
	  % (kappa_mid_singles,kappa_upp_singles-kappa_mid_singles,kappa_mid_singles-kappa_low_singles),size=20,color='b')
   plt.text(20,max(kappa_post_multis),r'$\kappa=%.2f^{+%.2f}_{-%.2f}$'\
	  % (kappa_mid_multis,kappa_upp_multis-kappa_mid_multis,kappa_mid_multis-kappa_low_multis),size=20,color='r')
   plt.legend(loc='upper right')
   plt.xlabel(r'$\kappa$',size=18)
   plt.ylabel(r'PDF   $p(\kappa|\{cos I_{*,k}\})$',size=18)
   plt.xlim(0,100)
   plt.show()
   

.. image:: example_figures/kappa_posterior_m+w.png

Which is consistent with Figure 6 of Morton & Winn (2014).

Statistical significance:
.....

The two-sample separation carried out above could also be the result
of arbitrarily separating a data sample into two smaller subsets. To address
the significance of the difference between the two subsets we need to: (1) come
of with a distance or difference metric; and (2) carry out a Monte Carlo
resampling/subset extraction to compute how likely that given difference metric
is to occur in random subdivisions of the original data.

As a metric of difference, we use the squared Hellinger distance, which is included in the repository. For the two-sample kappa
distributions above:

.. code:: python
 
   print obl.hellinger_distance(kappa_post_singles,kappa_post_multis,kappa_vals)


To get a distribution of "hellinger distances" and therefore assess how likely it is to obtained
the reall metric above out of random chance, we need to repeat the separation of the data sets into two random subsets of
sizes :code:`size1` and :code:`size2` a  total of :code:`draw=1000` times.

.. code:: python

   size1, size2 = len(cosipdf_singles), len(cosipdf_multis)
   print size1, size2

A more efficient way of carrying out this Monte Carlo experiment is to keep all
the contributions of each measurement to loglikelihood in the hierarchical inference calculation,
and sum them together according to random assingments to :code:`dataset1` and :code:`dataset1`.
For this, we use

.. code:: python

   # for all targets in the sample
   kappa_loglike_contr = obl.compute_hierachical_likelihood_contributions(kappa_vals,\
                                                                          obl.cosi_pdf_interp,cosi_vals,cosipdf,\
                                                                          K = 1000)

The calculation presented above is reproduced if

.. code:: python
	  
   ind = df_mw['Nplanets'] == 1
   kappa_post_1 = np.exp(kappa_loglike_contr[:,ind].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
   kappa_post_2 = np.exp(kappa_loglike_contr[:,np.invert(ind)].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
   kappa_post_1 /= trapz(kappa_post_1,x=kappa_vals)
   kappa_post_2 /= trapz(kappa_post_2,x=kappa_vals)
   distance = obl.hellinger_distance(kappa_post_1,kappa_post_2,kappa_vals)

What we want is to produce 1000 random two-sample separations and compute the squared Hellinger distance for each. We can
do that by generating and index array containing a number of :code:`True` entries equal to :code:`size1` and a number of :code:`False`
entries equal to :code:`size2`:
   
.. code:: python

   indices = np.random.permutation(np.append(np.ones(size1),np.zeros(size2)).astype(bool))
   draws = 1000
   hellinger_list = np.zeros(draws)
   for jj in range(draws):
	  indices = np.random.permutation(indices)
	  print indices[indices == True].shape[0], indices[indices == False].shape[0]
	  kappa_post_groupa = np.exp(kappa_loglike_contr[:,indices].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
	  kappa_post_groupb = np.exp(kappa_loglike_contr[:,np.invert(indices)].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
          kappa_post_groupa /= trapz(kappa_post_groupa,x=kappa_vals)
          kappa_post_groupb /= trapz(kappa_post_groupb,x=kappa_vals)
          hellinger_list[jj]= obl.hellinger_distance(kappa_post_groupa,kappa_post_groupb,kappa_vals)

   tvd_list = np.zeros(draws)
   for jj in range(draws):
	  indices = np.random.permutation(indices)
	  print indices[indices == True].shape[0], indices[indices == False].shape[0]
	  kappa_post_groupa = np.exp(kappa_loglike_contr[:,indices].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
	  kappa_post_groupb = np.exp(kappa_loglike_contr[:,np.invert(indices)].sum(axis = 1)) * obl.kappa_prior_function(kappa_vals)
          kappa_post_groupa /= trapz(kappa_post_groupa,x=kappa_vals)
          kappa_post_groupb /= trapz(kappa_post_groupb,x=kappa_vals)
          tvd_list[jj]= obl.total_variation_distance(kappa_post_groupa,kappa_post_groupb,kappa_vals)


	  
Basic Tutorial #2: Using lambda
--------

Coming soon...
