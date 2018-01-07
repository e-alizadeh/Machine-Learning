# Show the forward path of NN in the E-Commerce data and do Prediction
import numpy as np
from process import get_data

X, Y = get_data()

# Random initialization of weights
M = 5 # hidden units
D = X.shape[1]
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(A):
    A_exp = np.exp(A)
    return A_exp / A_exp.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)   

P_Y_given_X = forward(X, W1, b1, W2, b2)
Predictions = np.argmax(P_Y_given_X, axis=1)   

def Classification_rate(Y, P):
    return np.mean(Y == P)

print "Score:", Classification_rate(Y, Predictions)

