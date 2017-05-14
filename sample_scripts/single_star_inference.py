import obliquity_inference as obl
import numpy as np
import matplotlib as plt



if __name__ == "__main__":

    # Rotation period
    prot = 20.0
    prot_err = 2.0

    # Stellar radius
    radius =  0.8
    radius_err = [0.12,0.08]

    veq_vals = obl.sample_veq_vals(prot,prot_err,radius,radius_err)

    print veq_vals
