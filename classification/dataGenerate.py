import numpy as np

# 1-dimension
# Generate data with insects of either male or female
# male tends to have a longer body length, while female tends to be shorter
# Each point represents body length (x axis) and sex (y axis)
np.random.seed(seed = 1) # fix the random number
X_min = 0
X_max = 2.5
X_n = 100
X_col = ['red', 'blue']
X = np.zeros(X_n) # input data
T = np.zeros(X_n, dtype = np.uint8) # 1 = male, 0 = female, target data
Dist_s = [0.4, 0.8] # starting number 
Dist_w = [0.8, 1.6] # distribution range
Pi = 0.5 # ratio of female (0)

for i in range(X_n):
    wk = np.random.rand()
    T[i] = 0 * (wk < Pi) + 1 * (wk >= Pi)
    X[i] = np.random.rand() * Dist_w[T[i]] + Dist_s[T[i]]
    
np.savez('data_1d2c.npz', X_min = X_min, X_max = X_max, X_n = X_n, X_col = X_col, \
         X = X, T = T)


# 2-dimension
N = 500 # number of data points
K = 3 # number of classes
T2 = np.zeros((N, 2), dtype = np.uint8)
T3 = np.zeros((N, 3), dtype = np.uint8)
X_col2 = ['red', 'blue']
X_col3 = ['red', 'green', 'blue']

X = np.zeros((N, 2))
X0_range = [-3, 3] # range of X0
X1_range = [-3, 3] # range of X1
MU = np.array([[-.5, -.5], [.5, 1.0], [1, -.5]]) # center of distribution
SIG = np.array([[.7, .7], [.8, .3], [.3, .8]]) # variance of distribution
PI = np.array([0.4, 0.8, 1]) # distribution of classes: 0.4 0.8 1

for n in range(N):
    wk = np.random.rand()
    for k in range(K):
        if wk < PI[k]:
            T3[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn() * SIG[T3[n, :] == 1, k] + MU[T3[n, :] == 1, k])

T2[:, 0] = T3[:, 0]
T2[:, 1] = T3[:, 1] | T3[:, 2]

np.savez('data_2d2c.npz', X0_min = X0_range[0], X0_max = X0_range[1], X1_min = X1_range[0],
         X1_max = X1_range[1], X_n = N, X_col = X_col2, X = X, T = T2)
np.savez('data_2d3c.npz', X0_min = X0_range[0], X0_max = X0_range[1], X1_min = X1_range[0],
         X1_max = X1_range[1], X_n = N, X_col = X_col3, X = X, T = T3)
# 2 class data is stored in T2, 3 class data is stored in T3

