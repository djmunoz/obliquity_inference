OBLIQUITY INFERENCE
==================================================
.. sectnum::
   
Overview
--------

Installation
--------


Basic Tutorial #1: Using *V sin I*
--------

In order to carry out the Bayesian parameter estimation of the concentration parameter "kappa" (Fabrycky & Winn, 2009), you need to do it in three steps.

 **How to compute** *sin I* **from observations of** *VsinI* **and**  *P*:sub:`rot`
~~~~~~~~

First, you import the package
   
.. code:: python
	  
   import obliquity_inference as obl
   import numpy as np
   
If, for a given star, you have *VsinI* and *P*:sub:`rot` measurements (with errors), you can get a probability distribution function (PDF) for the inclination *cosI*.

First, you need to obtain obtain a PDF for the star's equatorial velocity. You can accomplish this
by running

.. code:: python

   veq_vals = sample_veq_vals(P,dP,R,dR,N=20000)

where P and dP, and R and dR, are the period measurement and the stellar radius measurements with their respective uncertainties. If there is only one value of uncertainty for a given emasurement, it assumed that said measurement is distributed normally with mean and dispersion given by the measurement and its error. If there is an 'upper' and 'lower' uncertainty interval (as it is often the case for the radius of Kepler stars)

To compute the inclination PDF, you have two options:

- Using the full PDF of *V*:sub:`eq`

Following the statistical techniques of Morton & Winn (2014), we can compute the PDF of
*cosI* - for a **given** star - by creating an empirical PDF for *V*:sub:`eq`

.. code:: python

   from scipy.stats import gaussian_kde
   def veq_dist(x):
	  return gaussian_kde(veq_vals,bw_method=0.1).evaluate(x)
   
.. code:: python
   
   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([posterior_cosi_full(c,vsini_dist,veq_dist)  for c in cosi_arr])

  
- Using the analytic approximation

Alternatively, if *both* vsini_dist and veq_dist can be well approximated by normal distributions,
you can use the analytic approximation of Munoz & Perets (2017)
  
.. code:: python

   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([posterior_cosi_analytic(c,Vsini0,dVsini0,veq_vals.mean(),veq_vals.std()) for c in cosi_arr])


B. **Computing a set of** *cosI* **PDFs from a CSV file/dataset**

For this, we use dataframe objects in the pandas Python package.

.. code:: python
	  
   import pandas as pd

For a collection of stars, you can either save all the inclination posteriors PDFs, or simply save *V*:sub:`eq` (with 68% confidence intervals) and recompute the inclination PDF using the analytic approximation.

C. **Combining MULTIPLE** *cosI* **PDFs to perform hierarchical Bayesian inference on the "concentration" parameter**

   The main goal is to compute a posterior PDF for the concentration parameter kappa. To implement the hierarchical Bayesian inference formalism of Hogg et al (2009) one needs a collection of PDFs for the line-of-sight inclination angle *I* (or more conveniently, PDFs for *cosI*; Morton & Winn, 2014).

a. Hello

.. code:: python


Basic Tutorial #2: Using lambda
--------
