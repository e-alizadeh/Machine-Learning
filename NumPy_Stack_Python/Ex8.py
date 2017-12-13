# Generate and plot spiral dataset
import numpy as np
import matplotlib.pyplot as plt

def getCoordinates(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta) 
    return(x,y)


NumPoints = 200 # number of points
t = np.random.random(NumPoints) # Generate random numbers

mu, sigma = 0, 0.2 # mean and standard deviation
noise = np.random.normal(mu,sigma,2*NumPoints).reshape(2,NumPoints) # Generate noise added to both x and y

# generate theta(t) and r(t)

r0, k = 0.5, 5
r = 5 * t 

theta = (120 * np.pi / 180) * t     # Angle of the first spiral

# generate the first spiral
spiral = np.asarray(getCoordinates(r, theta)) + noise   # np.asarray() converts the input to an array
spirals = spiral

colors = np.full(NumPoints, fill_value = 'darkblue')

for i in range(1,6):
    newAngle = theta + i * np.pi / 3 # 60 degrees difference
    spirals = np.concatenate((spirals, np.asarray(getCoordinates(r, newAngle)) + noise), axis=1)
    if i % 2 == 1:
        colors = np.append(colors, np.full(NumPoints, fill_value = 'darkred'))
    else:
        colors = np.append(colors, np.full(NumPoints, fill_value = 'darkblue'))


plt.scatter(spirals[0,:],spirals[1,:], c = colors, alpha = 0.5)
plt.axis('equal')
plt.show()
