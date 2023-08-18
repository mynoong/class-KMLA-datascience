# classify insect's sex by body length
# vertical boundary line is drawn on the grid classifying 
# female (red dots) and male (blue dots)

import numpy as np
import matplotlib.pyplot as plt
from classification import classification as cl

outfile = np.load('data_1d2c.npz')
X = outfile['X']
X_min = outfile['X_min']
X_max = outfile['X_max']
X_col = outfile['X_col']
X_n = outfile['X_n']
T = outfile['T']

plt.figure(1, figsize = (4, 4))
W_init = [1, -1]
W = cl.fit_logistic_1d2c(W_init, X, T)
mcee = cl.mcee_logistic_1d2c(W, X, T)


# plot data points
for i in range(np.max(T) + 1):
    plt.plot(X[T == i], T[T == i], X_col[i], alpha = 0.5, linestyle = 'none', marker = 'o')

# plot fitting logistic line
xb = np.linspace(X_min, X_max, 100)
y = cl.logistic_1d(W, xb)
plt.plot(xb, y, color = 'gray', linewidth = 3)

# plot boundary line
i = np.min(np.where(y > 0.5)) # returns the smallest index in which y > 0.5
B = (xb[i - 1] + xb[i]) / 2
plt.plot([B, B], [-0.5, 1.5], color = 'black', linestyle = '--')

print("Boundary = {0: .2f} cm".format(B))
print("Mean cross-entropy error = {0:.2f}".format(mcee))

plt.grid(True)
plt.xlim(X_min, X_max)
plt.ylim(-0.5, 1.5)
plt.yticks([0, 1])
plt.show()
