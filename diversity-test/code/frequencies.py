import numpy as np
import matplotllib.pyplot as plt

order = 4
#=============================================================================#
#-----------path to test results file
ss = s + str(order) + str(threshold) + "test_results.pkl"
#-----------load the div_pts dict for testing
with open (ss, "rb") as f:
    results = pickle.load(f)

#-----------the observed rankings, distances, allcounts and diversity dates are
keyobj = ["obr", "d", "ak", "dd"]



