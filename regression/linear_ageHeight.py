import numpy as np
import matplotlib.pyplot as plt
from regression import regression as reg

# load data points with age on x axis, height on y axis
outfile = np.load('data_1d.npz')
X = outfile['X']
X_min = outfile['X_min']
X_max = outfile['X_max']
X_n = outfile['X_n']
T = outfile['T']


# Gradient Descent on the relation age v. height
def gradientDescent_1d(x, t):
    w_init = [15.0, 175.0] # initial parameter variable
    alpha = 0.0001 # learning rate
    i_max = 500000 # maximum number of iteration
    eps = 0.1 # iteration-finishing condition of MSE function's slope
    w_i = np.zeros([i_max, 2])
    w_i[0, :] = w_init
    
    for i in range(1, i_max):
        dmse = reg.dmse_1d(x, t, w_i[i-1])
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
        
        if max(np.absolute(dmse)) < eps:
            break
    
    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    
    return w0, w1, dmse


def main():
    W0, W1, dMSE = gradientDescent_1d(X, T)
    plt.figure(figsize = (4, 4))
    W = np.array([W0, W1])
    mse = reg.mse_1d(X, T, W)
    
    # plots y = w0 * x + w1 linear graph
    xb = np.linspace(X_min, X_max, 100)
    y = W[0] * xb + W[1]
    plt.plot(xb, y, color = 'black', linewidth = 3)
    
    # plots datapoints
    print("w0 = {0: .3f}, w1 = {1: .3f}".format(W0,W1))
    print("SD = {0: .3f} cm".format(np.sqrt(mse)))
    plt.plot(X, T, marker = 'o', linestyle = 'None', color = 'yellow')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    main()
  