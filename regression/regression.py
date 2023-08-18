import numpy as np

class regression:
    def __init__(self):
        pass
    
    #Mean square error function
    def mse_2d(x, t, w):
        y = w[0] * x + w[1]
        mse = np.mean((y - t)**2)
        return mse

    # Slope of mean square error function
    # where MSE function = np.mean((w_0 * x + w_1 - t)**2)
    def dmse_2d(x, t, w):
        y = w[0] * x + w[1]
        d_w0 = 2 * np.mean((y - t) * x)
        d_w1 = 2 * np.mean(y - t)
        return d_w0, d_w1
    
    def mse_3d(x0, x1, t, w):
        y = w[0] * x0 + w[1] * x1 + w[2]
        mse = np.mean((y - t)**2)
        return mse
    
    def dmse_3d(x0, x1, t, w):
        y = w[0] * x0 + w[1] * x1 + w[2]
        d_w0 = 2 * np.mean((y - t) * x0)
        d_w1 = 2 * np.mean((y - t) * x1)
        d_w2 = 2 * np.mean(y - t)
        return d_w0, d_w1, d_w2
    
    def fit_analytic_3d(x0, x1, t):
        cov_tx0 = np.mean(t * x0) - np.mean(t) * np.mean(x0)
        cov_tx1 = np.mean(t * x1) - np.mean(t) * np.mean(x1)
        cov_x0x1 = np.mean(x0 * x1) - np.mean(x0) * np.mean(x1)
        var_x0 = np.var(x0)
        var_x1 = np.var(x1)
        w0 = (cov_tx1 * cov_x0x1 - var_x1 * cov_tx0) / (cov_x0x1**2 - var_x0 * var_x1)
        w1 = (cov_tx0 * cov_x0x1 - var_x0 * cov_tx1) / (cov_x0x1**2 - var_x0 * var_x1)
        w2 = - w0 * np.mean(x0) - w1 * np.mean(x1) + np.mean(t)
        return np.array([w0, w1, w2])
    
    def __gauss(x, mu, s):
        return np.exp(-(x - mu)**2 / (2 * s**2))
    
    def gauss_model(x, w): # linear model with gauss function as a basis function
        n = len(w) - 1
        mu = np.linspace(5, 30, n)
        s = mu[1] - mu[0]
        y = np.zeros_like(x)
        for i in range(n):
            y = y + w[i] * regression.__gauss(x, mu[i], s)
        y = y + w[n]
        return y
    
    def mse_gauss_2d(x, t, w):
        y = regression.gauss_model(x, w)
        mse = np.mean((y - t)**2)
        return mse
        
    def fit_gauss_analytic_2d(x, t, m):
        mu = np.linspace(5, 30, m)
        s = mu[1] - mu[0]
        n = x.shape[0]
        psi = np.ones((n, m + 1))
        
        for j in range(m):
            psi[:, j] = regression.__gauss(x, mu[j], s)
        psi_T = np.transpose(psi)
    
        b = np.linalg.inv(psi_T.dot(psi))
        c = b.dot(psi_T)
        w = c.dot(t)
        return w
    
    def kfold_gauss_model(x, t, m, k):
        n = x.shape[0]
        mse_train = np.zeros(k)
        mse_test = np.zeros(k)
        for i in range(0, k):
            # fmod gives a list of remainders of numbers, 0 to k-1 , divided by k
            x_train = x[np.fmod(range(n), k) != i]
            t_train = t[np.fmod(range(n), k) != i] 
            x_test = x[np.fmod(range(n), k) == i] 
            t_test = t[np.fmod(range(n), k) == i] 
            
            w = regression.fit_gauss_analytic_2d(x_train, t_train, m)
            mse_train[i] = regression.mse_gauss_2d(x_train, t_train, w)
            mse_test[i] = regression.mse_gauss_2d(x_test, t_test, w)
        
        return mse_train, mse_test