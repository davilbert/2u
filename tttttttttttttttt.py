import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from math import exp,sin,cos
from pylab import *


mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
a=0.05
b=0.10
# took the liberty of reducing the max value for th
# as it was giving you values of the order of e42
th=np.linspace(0, 50, 100)
th1=np.linspace(50, 0, 100)
x=a*exp(b*th)*cos(th)
y=a*exp(b*th)*sin(th)
X=-a*exp(-b*th1)*cos(th1)
Y=-a*exp(-b*th1)*sin(th1)
z=np.linspace(-16,16, 100)
Z=np.linspace(16,16, 100)# creating the z array with the same length as th
ax.plot(x, y, z, c='red')
ax.plot(X, Y, Z) # adding z as an argument for the plot
ax.legend()

plt.show()
