import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

outfile = np.load('data_2d3c.npz')
X_train = outfile['X_train']
T_train = outfile['T_train']
X_test = outfile['X_test']
T_test = outfile['T_test']
X0_range = outfile['X0_range']
X1_range = outfile['X1_range']

def main():
    startTime = time.time()
    
    model = Sequential()
    model.add(Dense(2, input_dim = 2, activation = 'sigmoid',
                kernel_initializer = 'uniform'))
    model.add(Dense(3, activation = 'softmax',
                kernel_initializer = 'uniform')) 
    sgd = keras.optimizers.legacy.SGD(learning_rate = 0.1, momentum = 0.0,
                                decay = 0.0, nesterov = False) 
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy',
                    metrics = ['accuracy']) 
    
    history = model.fit(X_train, T_train, epochs = 1000, 
                        batch_size = 50, verbose = 0, 
                        validation_data = (X_test, T_test))
    score = model.evaluate(X_test, T_test, verbose = 0)
    print('cross entropy {0:3.2f}, accuracy {1:3.2f}'
          .format(score[0], score[1]))
    calculation_time = time.time() - startTime
    print("Calculation time:{0:.3f} sec".format(calculation_time))
    
    
    # plots how error changes over time 
    # and boundary lines for datapoints
    plt.figure(1, figsize = (9, 4))
    plt.subplots_adjust(wspace = 0.5)
    
    # First, plots mean cross-entroy error respect to time
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'black', label='training')
    plt.plot(history.history['val_loss'], 'limegreen', label='test')
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
    y = model.predict(x)
    
    K = 3
    for i in range(K):
        f = y[:, i]
        f = f.reshape(xn, xn)
        f = f.T
        cont = plt.contour(xx0, xx1, f, levels = [0.5, 0.9],
                           colors = ['limegreen', 'black'])
        cont.clabel(fmt = '%1.1f', fontsize = 9)
    plt.xlim(X0_range)
    plt.ylim(X1_range)
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':
    main()