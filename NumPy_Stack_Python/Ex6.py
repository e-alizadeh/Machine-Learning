# Generate and plot XOR dataset
import numpy as np
import matplotlib.pyplot as plt

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