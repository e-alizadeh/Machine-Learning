import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

M = df.as_matrix()	# Convert DataFrame to NumPy Array (matrix)


# For checking
im = M[0, 1:] # select 0th row & all column except 0th column that is not a pixel

im = im.reshape(28,28)
plt.imshow(255-im, cmap='gray')
plt.show()

M[0,0] # Checking the label


ZEROs = M[ M[:,0] == 0]  # Extracting zeros by checking the label
meanZEROs = ZEROs[:,1:].sum(axis=0) / len(ZEROs)
# ZEROs[:,1:] -> 0th column contains the label, and hence it's excluded.
# np.sum(axis=0) -> summing over rows
# len(ZEROs) -> getting the length of the vector

ONEs = M[ M[:,0] == 1]  
meanONEs = ONEs[:,1:].sum(axis=0) / len(ONEs)

TWOs = M[ M[:,0] == 2]  
THREEs = M[ M[:,0] == 3]  
FOURs = M[ M[:,0] == 4]  
FIVEs = M[ M[:,0] == 5]  
SIXes = M[ M[:,0] == 6]  

SEVENs = M[ M[:,0] == 7]  
meanSEVENs = SEVENs[:,1:].sum(axis=0) / len(SEVENs)


EIGHTs = M[ M[:,0] == 8]  
NINEs = M[ M[:,0] == 9]  
	
	
