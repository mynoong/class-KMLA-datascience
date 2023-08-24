import numpy as np
from scipy.optimize import minimize

class neural_network:
    def __init__(self):
        pass
    
    def sigmoid(x):
        y = 1/ (1 + np.exp(-x))
        return y
    
    # wv: weight values of 1st and 2nd values concatenated
    # x: input data for classification
    # M: the number of 1st layer nodes
    # K: the number of 2nd layer nodes
    def FNN2(wv, x, M, K):
        N, D = x.shape # N: the number of datapoints, D: dimension
        w = wv[:M * (D + 1)]
        w = w.reshape(M, (D + 1)) # 1st layer weight values
        v = wv[M * (D + 1):]
        v = v.reshape((K, M + 1)) # 2nd layer weight values
        
        # set output matrixes
        b = np.zeros((N, M + 1)) # 1st layer input values
        z = np.zeros((N, M + 1)) # 1st layer output values
        a = np.zeros((N, K)) # 2nd layer input values
        y = np.zeros((N, K)) # 2nd layer output values (final result)
        
        for n in range(N):
            # 1st layer 
            for m in range(M):
                b[n, m] = np.dot(w[m, :], np.r_[x[n, :], 1]) 
                # np.r_[A, B] concatenate matrixes side by side
                z[n, m] = neural_network.sigmoid(b[n, m])
                
            # 2nd layer
            z[n, M] = 1 # dummy node
            tot = 0
            for k in range(K):
                a[n, k] = np.dot(v[k, :], z[n, :])
                tot = tot + np.exp(a[n, k])
            for k in range(K):
                y[n, k] = np.exp(a[n, k]) / tot
                
        return y, a, z, b
    
    def mcee_FNN2(wv, x, t, M, K):
        N, D = x.shape
        y, a, z, b = neural_network.FNN2(wv, x, M, K)
        mcee = - np.dot(np.log(y.reshape(-1)), t.reshape(-1)) / N
        return mcee
        
    def dmcee_FNN2(wv, x, t, M, K):
        N, D = x.shape
        # Turn concatenated wv into W and V matrix for computation
        w = wv[:M * (D + 1)]
        w = w.reshape(M, (D + 1))
        v = wv[M * (D + 1):]
        v = v.reshape((K, M + 1))
        
        # set output vectors
        dwv = np.zeros_like(wv)
        dw = np.zeros((M, D + 1))
        dv = np.zeros((K, M + 1))
        delta1 = np.zeros(M) # 1st layer error
        delta2 = np.zeros(K) # 2nd layer error
        
        # First, get y with initial wv and input x
        y, a, z, b = neural_network.FNN2(wv, x, M, K)
        
        for n in range(N):
            # Second, 2nd layer error
            for k in range(K):
                delta2[k] = (y[n, k] - t[n, k])
            # Third, 1st layer error
            for j in range(M):
                delta1[j] = z[n, j] * (1 - z[n, j]) * np.dot(v[:, j], delta2)
            # Fourth, 2nd layer weight v's dv
            for k in range(K):
                dv[k, :] = dv[k, :] + delta2[k] * z[n, :] / N
            # Fourth, 1st layer weight w's dw
            for j in range(M):
                dw[j, :] = dw[j, :] + delta1[j] * np.r_[x[n, :], 1] / N
        
        # concatenate dw and dv to dwv
        dwv = np.c_[dw.reshape((1, M * (D + 1))), \
                    dv.reshape((1, K * (M + 1)))]
        dwv = dwv.reshape(-1)
        return dwv
    
    def fit_FNN2(wv_init, x_train, t_train, x_test, t_test, M, K, n, alpha):
        # set output matrixes
        wv = wv_init.copy()
        err_train = np.zeros(n)
        err_test = np.zeros(n)
        wv_hist = np.zeros((n, len(wv_init)))
        
        epsilon = 0.001 # learning rate 
        for i in range(n):
            wv = wv - alpha * neural_network.dmcee_FNN2(wv, x_train, t_train, M, K)
            err_train[i] = neural_network.mcee_FNN2(wv, x_train, t_train, M, K)
            err_test[i] = neural_network.mcee_FNN2(wv, x_test, t_test, M, K)
            wv_hist[i, :] = wv
            
        return wv, wv_hist, err_train, err_test