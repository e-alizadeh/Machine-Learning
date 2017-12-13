import numpy as np
import random   # Draw random samples
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


# Define a uniform distribution of X
X = np.random.random(10000)

Y = []  # Initialization of Y = X1 + X2 + ... + Xn
for i in range(1000):
    Y.append( sum(random.sample(X, 1000)) )
    
Y = np.array(Y) # Converting list to np.ndarray

#plt.figure(1)
#plt.hist(X) # Should be a uniform distribution
#plt.show()

#plt.figure(2)
#plt.hist(Y) # Should be a normal distribution(bell curve)
#plt.show()