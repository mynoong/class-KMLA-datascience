import numpy as np
import time
import matplotlib.pyplot as plt
from neuralNetwork import neural_network as nn

outfile = np.load('data_2d3c.npz')
X_train = outfile['X_train']
T_train = outfile['T_train']
X_test = outfile['X_test']
T_test = outfile['T_test']
X0_range = outfile['X0_range']
X1_range = outfile['X1_range']


def main():
    # shows the time consumed for calculation
    startTime = time.time()
    
    M = 2 # the dummy node with value 1 is excluded
    K = 3
    np.random.seed(1)
    WV_init = np.random.normal(0, 0.01, M * 3 + K * (M + 1))
    N_step = 1000
    alpha = 1
    WV, WV_hist, Err_train, Err_test = nn.fit_FNN2(WV_init, X_train, \
        T_train, X_test, T_test, M, K, N_step, alpha)
    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))
    print("Mean cross-entropy error = {0:.2f}".format( \
        Err_test[N_step - 1]))
    
    # plots how error changes over time 
    # and boundary lines for datapoints
    plt.figure(1, figsize = (9, 4))
    plt.subplots_adjust(wspace = 0.5)
    
    # First, plots mean cross-entroy error respect to time
    plt.subplot(1, 2, 1)
    plt.plot(Err_train, 'black', label='training')
    plt.plot(Err_test, 'limegreen', label='test')
    plt.legend()
    
    # Second, plots classification boundary line and datapoints
    plt.subplot(1, 2, 2)
    # plot datapoints
    wk, n = T_test.shape
    c = ["black", "gray", "white"]
    for i in range(n):
        plt.plot(X_test[T_test[:, i] == 1, 0], 
                 X_test[T_test[:, i] == 1, 1],
                 linestyle = 'none',
                 marker = 'o', markeredgecolor = 'black',
                 color = c[i], alpha = 0.8)
    
    # plot contour lines
    xn = 60 # plotted contour resolution
    x0 = np.linspace(X0_range[0], X0_range[1], xn)
    x1 = np.linspace(X1_range[0], X1_range[1], xn)
    xx0, xx1 = np.meshgrid(x0, x1)
    x = np.c_[np.reshape(xx0, xn * xn), np.reshape(xx1, xn * xn)]
    y, a, z, b = nn.FNN2(WV, x, M, K)
    plt.figure(1, figsize = (4, 4))
    for ic in range(K):
        f = y[:, ic]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels = [0.8, 0.9],
                           colors = ['limegreen', 'black'])
        cont.clabel(fmt = '%1.1f', fontsize = 9)
    plt.xlim(X0_range)
    plt.ylim(X1_range)
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
        main()