import numpy as np

# 2-dimension
N = 500 # number of data points
K = 3 # number of classes
T3 = np.zeros((N, 3), dtype = np.uint8)

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

TestRatio = 0.5
X_n_training = int(N * TestRatio)
X_train = X[:X_n_training, :]
X_test = X[X_n_training:, :]
T_train = T3[:X_n_training, :]
T_test = T3[X_n_training:, :]

np.savez('data_2d3c.npz', X_train = X_train, T_train = T_train,
         X_test = X_test, T_test = T_test,
         X0_range = X0_range, X1_range = X1_range)