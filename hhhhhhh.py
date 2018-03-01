import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
#a = int (input())
#mpl.rcParams['legend.fontsize'] = 10


cs = 0.2

cs1 = 0.2


colorsMap='jet'

fig = plt.figure()
##scalarMap.set_array(cs)
fig1 = plt.figure()

def spiral(t):
    z = np.linspace(-i, i, 100)



ax = fig.add_subplot(111, projection='3d')
k = -5.0
for i in np.arange(-15.0,k, 1.0):
    theta2 = np.linspace(-10 * np.pi, 10 * np.pi, 10)
    z2 = np.linspace(-i, i, 10)
    r2 = z2**2 + 1
    x2 = -r2 * np.sin(theta2)
    y2 =- r2 * np.cos(theta2)
    #ax.scatter(x2, y2, z2, c='green', marker='_', s=1, lw=1.0) # lw=2.5
t = 5
for i in np.arange(-5.0,t, 1.0):
    theta1 = np.linspace(-10 * np.pi, 10 * np.pi, 1000)
    z1 = np.linspace(-i, i, 1000)
    r1 = z1**2 + 1
    x1 = -r1 * np.sin(theta1)
    y1 =- r1 * np.cos(theta1)
    ax.scatter(x1, y1, z1, c='red', marker='_', s=1, lw=1.0) # lw=2.5
#plt.colorbar(scalarMap)
p = 15
for i in np.arange(5.0,p, 1.0):
    theta3 = np.linspace(-10 * np.pi, 10 * np.pi, 1000)
    z3 = np.linspace(-i, i, 1000)
    r3 = z3**2 + 1
    x3 = -r3 * np.sin(theta3)
    y3 =- r3 * np.cos(theta3)
    #ax.scatter(x3, y3, z3, c='black', marker='_', s=1,lw=1.0) # lw=2.5
plt.show()
