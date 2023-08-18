import numpy as np

# Generate age(X) v. height data with 100 datapoints
np.random.seed(seed = 1) # fix the random number
X_min = 4 # minimum age 
X_max = 30 # maximum age
X_n = 100 # number of data points
X = 5 + 25 * np.random.rand(X_n)
prm_c = [180, 120, 0.2]
T = prm_c[0] - prm_c[1] * np.exp(-prm_c[2] * X) \
    + 4 * np.random.randn(X_n)

np.savez('regression/data_2d.npz', X = X, X_min = X_min, X_max = X_max, X_n = X_n, T = T) # saves data in the file


# Generate age(X1) + weight(X2) v. height data (3d) with 100 datapoints
X0 = X
X0_min = X_min
X0_max = X_max
X1 = 23 * (T / 100)**2 + 2 * np.random.randn(X_n)
X1_min = 40 # minimum weight
X1_max = 75 # maximum weight

np.savez('regression/data_3d.npz', X0 = X0, X1 = X1, X0_min = X0_min, X0_max = X0_max, \
    X1_min = X1_min, X1_max = X1_max, X_n = X_n, T = T) # saves data in the file
