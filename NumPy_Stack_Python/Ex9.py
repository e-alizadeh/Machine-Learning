import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################
## Generating XOR dataset from Ex.6 ###########
RandNum = np.random.uniform(-1, 1, (1000, 2))  # (-1,1) 1000 rows with 2 columns

x_less_than_0 = RandNum[ RandNum[:,0] < 0]
x_greater_than_0 = RandNum[ RandNum[:,0] > 0]

# Bottom Left sqaure: -1 < x < 0  and -1 < y < 0 
BottomLeft_Sqaure = x_less_than_0[ x_less_than_0[:,1] < 0] 

# Top Left sqaure: -1 < x < 0  and 0 < y < 1 
TopLeft_Sqaure = x_less_than_0[ x_less_than_0[:,1] > 0]

# Top Right sqaure: 0 < x < 1  and 0 < y < 1
TopRight_Square = x_greater_than_0[ x_greater_than_0[:,1] > 0] 

# Bottom Right sqaure: 0 < x < 1  and -1 < y < 0 
BottomRight_Square = x_greater_than_0[ x_greater_than_0[:,1] < 0] 

plt.scatter(BottomLeft_Sqaure[:,0],BottomLeft_Sqaure[:,1],color='blue', alpha = 0.5)
plt.scatter(TopLeft_Sqaure[:,0],TopLeft_Sqaure[:,1],color='red', alpha = 0.5)
plt.scatter(TopRight_Square[:,0],TopRight_Square[:,1],color='blue', alpha = 0.5)
plt.scatter(BottomRight_Square[:,0],BottomRight_Square[:,1], color='red', alpha = 0.5)

plt.show()
################################################################
# Creating two classes for XOR dataset
class1 = np.append(TopLeft_Sqaure, BottomRight_Square, axis=0)
class2 = np.append(TopRight_Square, BottomLeft_Sqaure, axis=0)

# Combining two classes
data = np.append(class1, class2, axis=0)

# Creating labels for coressponding classes
y = np.append(np.zeros(len(class1),dtype=int), np.ones(len(class2), dtype=int))

d = {'x1': data[:,0], 'x2': data[:,1], 'y':y}

df = pd.DataFrame(data = d)

file_name = 'XOR_dataset.csv'
df.to_csv(file_name, encoding='utf-8', index=False) # No need to store preceding indices of each row