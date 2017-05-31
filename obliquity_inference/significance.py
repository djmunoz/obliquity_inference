import numpy as np
from scipy.integrate import trapz

def hellinger_distance(sample1,sample2,x):

    delta = 1 - trapz(np.sqrt(sample1[:] * sample2[:]),x=x)
    
    return delta


def total_variation_distance(sample1,sample2,x):

    delta = max(np.abs(np.asarray(sample1)[:]-np.asarray(sample2[:])))
