OBLIQUITY INFERENCE
==================================================

Overview
--------

Installation
--------


Basic Tutorial
--------

1. **How to compute** *sin I* **from data**

First, you import the package
   
.. code:: python
	  
   import obliquity_inference as obl
   import numpy as np
   
If, for a given star, you have *VsinI* and *P*:sub:`rot` measurements (with errors), you can get a probability distribution function (PDF) for the inclination *cosI*.

First, you need to obtain obtain a PDF for the star's equatorial velocity. You can accomplish this
by running

.. code:: python

   veq_vals = sample_veq_vals(P,dP,R,dR)

where P and dP, and R and dR, are respectively the period measurement with its uncertainty, and the stellar radius with its uncertainty. If there is only one value of uncertainty for a given emasurement, it assumed that said measurement is distributed normally with mean and dispersion given by the measurement and its error. If there is an 'upper' and 'lower' uncertainty interval (as it is often the case for the radius of Kepler stars)

To compute the inclination PDF, you have two options:

- Using the full PDF of *Veq*

Following the statistical techniques of Morton & Winn (2014), we can compute the PDF of
*cosI* - for a **given** star - by doing

.. code:: python

   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([posterior_cosi_full(c,Vsini0,dVsini0,veq_vals.mean(),veq_vals.std()) for c in cosi_arr])

  
- Using the analytic approximation

.. code:: python

   cosi_arr = np.linspace(0.0,0.99999999,300)
   post = np.asarray([posterior_cosi_analytic(c,Vsini0,dVsini0,veq_vals.mean(),veq_vals.std()) for c in cosi_arr])


2. **Computing a set of** *cosI* **PDFs from a CSV file/dataset**

For this, we use dataframe objects in the pandas Python package.

.. code:: python
   import pandas as pd


3. **Combining MULTIPLE** *cosI* **PDFs to perform hierarchical Bayesian inference on the "concentration" parameter**
