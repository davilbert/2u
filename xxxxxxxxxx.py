import numpy as np
import matplotlib
from matplotlib import pylab,mlab,pyplot,cm
plt = pyplot
from mpl_toolkits.mplot3d import Axes3D
from math import exp,sin,cos
from pylab import *

##l = []
##n=8
##print('Add a nucleotids, please')
##for i in range(n):
     ##   l.append(input())
##print(l)
##def trans(l):
##    l2=[]
   ## for i in range(n):
      ##  if l[i]== 'A':
      ##      l2.append('T')
      ##  if l[i]== 'T':
            ##l2.append('A')
     ##   if l[i]== 'G':
       ##     l2.append('C')
      ##  if l[i]== 'C':
        ##    l2.append('G')
   ## return l2
##print(trans(l))


fig_1 = plt.figure()
ax = fig_1.add_subplot(111, projection='3d')

# Spiral parameters
samNum1 = 1000
spConst1 = 5.0
t = np.linspace(0,89*np.pi, samNum1)

T, z = np.meshgrid(t, [0,1])
x = spConst1 * (np.cos(T) + T* np.sin(T))
y = spConst1 * (np.sin(T) - T * np.cos(T))

# Coordinates of involute spiral on xy-plane
coords = np.zeros([samNum1, 3])
coords[:,0] = spConst1 * (np.cos(t) + t * np.sin(t)) # x coord
coords[:,1] = spConst1 * (np.sin(t) - t * np.cos(t)) # y coord

# Paramters for 2D Gaussian surface
amp1 = 200
sigma_x1 = -75.0
sigma_y1 = -75.0
theta1 = np.pi
a1 = np.cos(theta1)**2 / (2 * sigma_x1**2) + np.sin(theta1)**2 / (2 * sigma_y1**2)
b1 = np.sin(2 * theta1) / (4 * sigma_x1**2) - np.sin(2 * theta1) / (4 * sigma_y1**2)
c1 = -np.sin(theta1)**2 / (2 * sigma_x1**2) + np.cos(theta1)**2 / (2 * sigma_y1**2)

# z coords of spiral projected onto Gaussian surface
coords[:,2] = amp1 * np.exp(-(a1 * coords[:,0]**2 - 2 * b1 * coords[:,0]*coords[:,1] + c1 * coords[:,1]**2)) # z coord

z[1,:] = coords[:,2]
ax.plot_surface(x,y,z)
plt.show()
