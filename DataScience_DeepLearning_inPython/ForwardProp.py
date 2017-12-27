# Coding Softmax
import numpy as np
import matplotlib.pyplot as plt

NumSample = 500 # Number of samples per class

X1 = np.random.randn(NumSample, 2) + np.array([0,-2])
X2 = np.random.randn(NumSample, 2) + np.array([2,2])
X3 = np.random.randn(NumSample, 2) + np.array([-2,2])
X = np.vstack([X1, X2, X3])


Ylabels = np.array([0]*NumSample + [1]*NumSample + [2]*NumSample)


plt.scatter(X[:,0], X[:,1], c=Ylabels, s=100, alpha=0.5)
plt.show()

D = 2
M = 3 # Hidden layer size
K = 3 # Number of classes

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)


# Defining forward action of the NN
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    
    A_exp = np.exp(A)
    Ylabels = A_exp / A_exp.sum(axis=1, keepdims=True)
    return Ylabels

def classification_rate(Y_target, Prediction):
    n_correct = 0
    n_total = 0

    for i in xrange(len(Y_target)):
        n_total += 1
        if Y_target[i] == Prediction[i]:
            n_correct += 1

    return float(n_correct) / n_total


P_Y_given_X = forward(X, W1, b1, W2, b2)
Prediction = np.argmax(P_Y_given_X, axis=1)

assert(len(Prediction) == len(Ylabels))

print "Classification rate for randomly chosen weights:", classification_rate(Ylabels, Prediction)


