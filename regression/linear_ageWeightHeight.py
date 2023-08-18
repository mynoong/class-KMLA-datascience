import numpy as np
import matplotlib.pyplot as plt
from regression import regression as reg

# load data points with age on x0 axis, weight on x1 axis, and height on y axis
outfile = np.load('regression/data_3d.npz')
X0 = outfile['X0']
X1 = outfile['X1']
X0_min = outfile['X0_min']
X0_max = outfile['X0_max']
X1_min = outfile['X1_min']
X1_max = outfile['X1_max']
X_n = outfile['X_n']
T = outfile['T']


# Gradient Descent on the relation age v. height
def gradientDescent_3d(x0, x1, t):
    w_init = [15.0, 60.0, 175.0] # initial parameter variable
    alpha = 0.0001 # learning rate
    i_max = 500000 # maximum number of iteration
    eps = 0.1 # iteration-finishing condition of MSE function's slope
    w_i = np.zeros([i_max, 3])
    w_i[0, :] = w_init
    
    for i in range(1, i_max):
        dmse = reg.dmse_3d(x0, x1, t, w_i[i-1])
        w_i[i, 0] = w_i[i - 1, 0] - alpha * dmse[0]
        w_i[i, 1] = w_i[i - 1, 1] - alpha * dmse[1]
        w_i[i, 2] = w_i[i - 1, 2] - alpha * dmse[2]
        
        if max(np.absolute(dmse)) < eps:
            break
    
    w0 = w_i[i, 0]
    w1 = w_i[i, 1]
    w2 = w_i[i, 2]
    
    return w0, w1, w2, dmse


def main():
    W0, W1, W2, dMSE = gradientDescent_3d(X0, X1, T)
    plt.figure(figsize = (6, 5))
    ax = plt.subplot(1, 1, 1, projection = '3d')
    
    # prints parameter variables w0, w1, w2, and standard deviation SD
    W = np.array([W0, W1, W2])
    mse = reg.mse_3d(X0, X1, T, W)
    print("w0 = {0: .3f}, w1 = {1: .3f}, w2 = {2: .3f}".format(W0, W1, W2))
    print("SD = {0: .3f} cm".format(np.sqrt(mse)))
    
    # plots y = w0 * x0 + w1 * x1 + w2 plane graph
    px0 = np.linspace(X0_min, X0_max, 5)
    px1 = np.linspace(X1_min, X1_max, 5)
    px0, px1 = np.meshgrid(px0, px1)
    y = W[0] * px0 + W[1] * px1 + W[2]
    ax.plot_surface(px0, px1, y, rstride = 1, cstride = 1, alpha = 0.3, color = 'black')
    
    
    # plots datapoints
    for i in range(len(X0)):
        ax.plot([X0[i], X0[i]], [X1[i], X1[i]], [120, T[i]], color = 'gray')
        ax.plot(X0, X1, T, 'o', color = 'yellow', markersize = 3)
        ax.view_init(elev = 35, azim = -75)
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    main()
  