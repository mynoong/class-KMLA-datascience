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
    W_init = [100, 0, 0]
    W = reg.fit_new_model_1d(X, T, W_init)
    
    # plot new model function
    xb = np.linspace(X_min, X_max, 100)
    y = reg.new_model(xb, W)
    plt.plot(xb, y, color = 'black')
    
    # plot datapoints
    plt.plot(X, T, marker = 'o', linestyle = 'None', color = 'yellow')
    plt.xlim(X_min, X_max)
    plt.grid(True)
    mse = reg.mse_new_model_1d(W, X, T)
    print("SD ={0:.2f} cm".format(np.sqrt(mse)))
    plt.show()
    
    
if __name__ == '__main__':
    main()