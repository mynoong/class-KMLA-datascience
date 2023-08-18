import numpy as np
import matplotlib.pyplot as plt
from classification import classification as cl

outfile = np.load('data_2d2c.npz')
X = outfile['X']
X0_min = outfile['X0_min']
X0_max = outfile['X0_max']
X1_min = outfile['X1_min']
X1_max = outfile['X1_max']
X_col = outfile['X_col']
X_n = outfile['X_n']
T = outfile['T']


plt.figure(1, figsize = (10, 4))
plt.subplots_adjust(wspace = 0.5)
W_init = [0, 0, 0]
W = cl.fit_logistic_2d2c(W_init, X, T)
mcee = cl.mcee_logistic_2d2c(W, X, T)


## first graph (3d)
ax = plt.subplot(1, 2, 1, projection = '3d')
# plot fitting logistic plane in 3d
xb = 100
x0 = np.linspace(X0_min, X0_max, xb)
x1 = np.linspace(X1_min, X1_max, xb)
xx0, xx1 = np.meshgrid(x0, x1)
y = cl.logistic_2d(W, xx0, xx1)
ax.plot_surface(xx0, xx1, y, color = 'blue', edgecolor = 'gray', rstride = 7, \
    cstride = 7, alpha = 0.2)

# plot data points in 3d
for i in range(2):
    ax.plot(X[T[:, i] == 1, 0], X[T[:, i] == 1, 1], 1 - i,
                marker = 'o', color = X_col[i], markeredgecolor = 'black',
                linestyle = 'none', markersize = 5, alpha = 0.8)
ax.view_init(elev = 25, azim = -30)


## second graph (2d)
ax = plt.subplot(1, 2, 2)
# plot boundary line (contour line) in 2d
# use xb, x0, x1, xx0, xx1, y above from 3d
cont = plt.contour(xx0, xx1, y, levels = (0.2, 0.5, 0.8), \
    colors = ['lightgray', 'black','lightgray'])
cont.clabel(fmt = '%1.1f', fontsize = 10)

# plot data points in 2d
wk, K = T.shape
for k in range(K):
    plt.plot(X[T[:, k] == 1, 0], X[T[:, k] == 1, 1], linestyle = 'none', 
             markeredgecolor = 'black', marker = 'o', color = X_col[k], alpha = 0.5)


print("Mean cross-entropy error = {0:.2f}".format(mcee))
plt.grid(True)
plt.show()
