import numpy as np
import matplotlib.pyplot as plt
from regression import regression as reg

# load data points with age on x axis, height on y axis
outfile = np.load('regression/data_2d.npz')
X = outfile['X']
X_min = outfile['X_min']
X_max = outfile['X_max']
X_n = outfile['X_n']
T = outfile['T']
    
def main():
    M = range(2, 15)
    K = 20
    # the value in Gauss_test[i][j] means 
    # where M - the number of basis function - is j, the mse for the ith data set;
    # data X and T are splitted into K sets and there is 1th to kth data set.
    # If ith data set is chosen for test, the others are used for training. 
    # Gauss_train[i][j] has the mse from all data, excluding that ith data.
    Gauss_train = np.zeros((K, len(M)))
    Gauss_test = np.zeros((K, len(M)))
    
    for i in range(0, len(M)):
        Gauss_train[:, i], Gauss_test[:, i] = reg.kfold_gauss_model(X, T, M[i], K)
    mean_Gauss_train = np.sqrt(np.mean(Gauss_train, axis = 0))
    mean_Gauss_test = np.sqrt(np.mean(Gauss_test, axis = 0))
    
    M_best = np.argmin(mean_Gauss_test)
    
    # plots mse of test and training data - to determine appropriate value of M
    plt.figure(figsize=(4, 4))
    plt.plot(M, mean_Gauss_train, marker = 'o', linestyle = '-', color = 'black', \
        markeredgecolor = 'black', label = 'training')
    plt.plot(M, mean_Gauss_test, marker = 'o', linestyle = '-', color='yellow', \
        markeredgecolor = 'black', label = 'test')
    plt.legend(loc = 'upper left', fontsize = 10)
    #plt.ylim(0, 20)
    plt.grid(True)
    plt.show()
    
    # plots a fitting lines and data points with the optimal M chosed
    plt.figure(figsize=(4, 4))
    W = reg.fit_gauss_analytic_2d(X, T, M_best) # fitting line
    xb = np.linspace(X_min, X_max, 100)
    y = reg.gauss_model(xb, W)
    plt.plot(xb, y, color = 'black', lw = 3)
    
    plt.plot(X, T, marker='o', linestyle='None',color='yellow') # data points
    plt.xlim([X_min, X_max])
    plt.grid(True)
    mse = reg.mse_gauss_2d(X, T, W)
    print("SD ={0:.2f} cm".format(np.sqrt(mse)))

    plt.show()
    
    
if __name__ == '__main__':
    main()
    
