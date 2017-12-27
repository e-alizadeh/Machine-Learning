# Coding Softmax
import numpy as np

NumSamp = 100
NumClass = 5


A = np.random.randn(NumSamp, NumClass)
A_exp = np.exp(A)

Softmax_Output = A_exp / A_exp.sum(axis=1, keepdims=True)
 # Sum up each row 

 # To check that each row sums up to ONE since they are probabilities. 
Softmax_Output.sum(axis=1)