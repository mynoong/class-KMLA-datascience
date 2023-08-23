import numpy as np
import matplotlib.pyplot as plt
from regression import regression as reg

# load data points with age on x0 axis, weight on x1 axis, and height on y axis
outfile = np.load('data_2d.npz')
X0 = outfile['X0']
X1 = outfile['X1']
X0_min = outfile['X0_min']
X0_max = outfile['X0_max']
X1_min = outfile['X1_min']
X1_max = outfile['X1_max']
X_n = outfile['X_n']
T = outfile['T']


def main():
    plt.figure(figsize = (6, 5))
    ax = plt.subplot(1, 1, 1, projection = '3d')
    
    W = reg.fit_analytic_2d(X0, X1, T)
    
    # prints parameter variables w0, w1, w2, and standard deviation SD
    mse = reg.mse_2d(X0, X1, T, W)
    print("w0 = {0: .3f}, w1 = {1: .3f}, w2 = {2: .3f}".format(W[0], W[1], W[2]))
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