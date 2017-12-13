# Generate and plot donuts (or concentric circles) dataset
import numpy as np
import matplotlib.pyplot as plt

x0 = 0
y0 = 0

# Generate Inner Circle 
Inner_Circ_Rin = 8
Inner_Circ_Rout = 10.2

a = 2*np.pi*np.random.random(1000)
r = np.sqrt(Inner_Circ_Rin**2 + (Inner_Circ_Rout**2 - Inner_Circ_Rin**2)*np.random.random(1000))

x_innerCirc = r * np.cos(a) + x0
y_innerCirc = r * np.sin(a) + y0

# Generate Outer Circle 
Outer_Circ_Rin = 18
Outer_Circ_Rout = 20.2

a = 2*np.pi*np.random.random(1000)
r = np.sqrt(Outer_Circ_Rin**2 + (Outer_Circ_Rout**2 - Outer_Circ_Rin**2)*np.random.random(1000))

x_outerCirc = r * np.cos(a) + x0
y_outerCirc = r * np.sin(a) + y0


plt.scatter(x_innerCirc,y_innerCirc,color='blue')
plt.scatter(x_outerCirc,y_outerCirc,color='red')

plt.axis([-30,30,-30,30])  #[xmin,xmax,ymin,ymax]
plt.show()