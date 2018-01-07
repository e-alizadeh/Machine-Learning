# NN Back Propagation (PREDICTION + TRAINING)
import numpy as np
import matplotlib.pyplot as plt

# Defining forward action of the NN
def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    
    A_exp = np.exp(A)
    Ylabels = A_exp / A_exp.sum(axis=1, keepdims=True)
    # Z should also be returned to calculate the gradient
    return Ylabels, Z 

# num correct / num total
def classification_rate(Y_target, Prediction):
    n_correct = 0
    n_total = 0

    for i in xrange(len(Y_target)):
        n_total += 1
        if Y_target[i] == Prediction[i]:
            n_correct += 1

    return float(n_correct) / n_total

def derivative_w2(Z, T, Y):
	N, K = T.shape
	M = Z.shape[1]

	# Slow way
	ret1 = np.zeros((M,K))
	for n in range(N):
		for m in xrange(M):
			for k in xrange(K):
				ret1[m,k] += (T[n,k]) 



def cost(T, Y):
	tot = T * np.log(Y)
	return tot.sum()

def main():
	# create the data
	NumSample = 500 # Number of samples per class

	D = 2
	M = 3 # Hidden layer size
	K = 3 # Number of classes

	X1 = np.random.randn(NumSample, 2) + np.array([0,-2])
	X2 = np.random.randn(NumSample, 2) + np.array([2,2])
	X3 = np.random.randn(NumSample, 2) + np.array([-2,2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*NumSample + [1]*NumSample + [2]*NumSample)

	N = len(Y)

	# indicator variable
	T = np.zeros((N,K))
	for i in xrange(N):
		T[i, Y[i]] = 1	# one-hot encoding for targets

	# let's see what it looks like
	plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	plt.show()	

	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M,K)
	b2 = np.random.randn(K)


	learning_rate = 10e-7
	cost = []
	for epoch in xrange(100000):
		output, hidden =forward(X, W1, b1, W2, b2)
		if epoch % 100 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis=1)
			r = classification_rate(Y,P)
			print "cost: ", c, "classification_rate: ", r
			costs.append(c) # append the cost to the cost array

		# Gradient Ascent
		W2 += learning_rate * derivative_w2(hidden, T, output)
		b2 += learning_rate * derivative_b2(T, output) 
		W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
		b1 += learning_rate * derivative_b1(T, output, W2, hidden)

	plt.plot(costs)	
	plt.show()



if __name__ == 'main':
	main()













P_Y_given_X = forward(X, W1, b1, W2, b2)


assert(len(Prediction) == len(Ylabels))

print "Classification rate for randomly chosen weights:", classification_rate(Ylabels, Prediction)


