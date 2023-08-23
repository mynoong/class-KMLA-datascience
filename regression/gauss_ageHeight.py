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
    
def main():
    plt.figure(figsize=(4, 4))
    M = 4
    W = reg.fit_gauss_analytic_1d(X, T, M)
    
    # plot gauss model function
    xb = np.linspace(X_min, X_max, 100)
    y = reg.gauss_model(xb, W)
    plt.plot(xb, y, color = 'black')
    
    # plot datapoints
    plt.plot(X, T, marker = 'o', linestyle = 'None', color = 'yellow')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    mse = reg.mse_gauss_1d(X, T, W)
    print('W =' + str(np.round(W, 1)))
    print("SD ={0:.2f} cm".format(np.sqrt(mse)))
    plt.show()
    
    
if __name__ == '__main__':
    main()