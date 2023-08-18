import numpy as np
from scipy.optimize import minimize

class classification:
    def __init__(self):
        pass
    
    def logistic_1d(w, x):
        y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
        return y
    
    def logistic_2d(w, x0, x1):
        y = 1 / (1 + np.exp(-(w[0] * x0 + w[1] * x1 + w[2])))
        return y
    
    def logistic_2d3c(w, x0, x1):
        K = 3
        w = w.reshape((3, 3))
        n = len(x1)
        y = np.zeros((n, K))
        for i in range(K):
            y[:, i] = np.exp(w[i, 0] * x0 + w[i, 1] * x1 + w[i, 2])
        wk = np.sum(y, axis = 1)
        wk = y.T / wk
        y = wk.T
        return y

    def mcee_logistic_1d2c(w, x, t):
        y = classification.logistic_1d(w, x)
        mcee = 0
        for i in range(len(y)):
            mcee = mcee - (t[i] * np.log(y[i]) + (1 - t[i]) * np.log(1 - y[i]))
        mcee = mcee / len(x)
        return mcee
    
    def mcee_logistic_2d2c(w, x, t):
        y = classification.logistic_2d(w, x[:, 0], x[:, 1])
        mcee = 0
        for i in range(len(y)):
            mcee = mcee - (t[i, 0] * np.log(y[i]) + (1 - t[i, 0]) * np.log(1 - y[i]))
        mcee = mcee / len(x)
        return mcee
    
    def dmcee_logistic_1d2c(w, x, t):
        y = classification.logistic_1d(w, x)
        dmcee = np.zeros(2)
        for i in range(len(y)):
            dmcee[0] = dmcee[0] + (y[i] - t[i]) * x[i]
            dmcee[1] = dmcee[1] + (y[i] - t[i])
        dmcee = dmcee / len(x)
        return dmcee
    
    def dmcee_logistic_2d2c(w, x, t):
        y = classification.logistic_2d(w, x[:, 0], x[:, 1])
        dmcee = np.zeros(3)
        for i in range(len(y)):
            dmcee[0] = dmcee[0] + (y[i] - t[i, 0]) * x[i, 0]
            dmcee[1] = dmcee[1] + (y[i] - t[i, 0]) * x[i, 1]
            dmcee[2] = dmcee[2] + (y[i] - t[i, 0])
        dmcee = dmcee / len(x)
        return dmcee
    
    
    def fit_logistic_1d2c(w_init, x, t):
        # Conjugate Gradient Method (CG) is selected as a gradient descent method here
        result = minimize(classification.mcee_logistic_1d2c, w_init, args = (x, t), \
            jac = classification.dmcee_logistic_1d2c, method = "CG")
        
        return result.x
    
    
    def fit_logistic_2d2c(w_init, x, t):
        result = minimize(classification.mcee_logistic_2d2c, w_init, args = (x, t), \
            jac = classification.dmcee_logistic_2d2c, method = "CG")
        
        return result.x