import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

M = df.as_matrix()	# Convert DataFrame to NumPy Array (matrix)


SEVENs = M[ M[:,0] == 7]  
meanSEVENs = SEVENs[:,1:].sum(axis=0) / len(SEVENs)
# ZEROs[:,1:] -> 0th column contains the label, and hence it's excluded.
# np.sum(axis=0) -> summing over rows
# len(ZEROs) -> getting the length of the vector

im = meanSEVENs
im = im.reshape(28,28)

# Method 1: using Numpy function
rotated_im = np.rot90(im, k = 3)
# k: number of times the array is rotated by 90 degress.

plt.imshow(255-rotated_im, cmap='gray')
plt.show()
	

# Method 2: using for-loop 
width = 28
height = 28
rotated_im2=np.zeros((width,height)) # initialize rotated_im2 so that can be used in the for-loops

for i in range(width):
    for j in range(height):
        #rotated_im2[i,j] = im[j,width-i-1] # 90 degrees CCW
         rotated_im2[i,j] = im[width-j-1,i] # 90 degrees CW



