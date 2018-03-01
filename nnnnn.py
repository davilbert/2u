import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


IMAGE = plt.figure()
ax = IMAGE.add_subplot(111, projection='3d')
samNum1 = 1000
spConst1 = 10.0
r = np.linspace(0,89*np.pi, samNum1)
T, ZZ = np.meshgrid(r, [0,1])
XX = spConst1 * (np.cos(T) + T* np.sin(T))
YY = spConst1 * (np.sin(T) - T * np.cos(T))
amp1 = 2
sigma_x1 = 0.75
sigma_y1 = 0.75
theta11 = np.pi
a11 = np.cos(theta11)**2 / (2 * sigma_x1**2) + np.sin(theta11)**2 / (2 * sigma_y1**2)
b11 = - np.sin(2 * theta11) / (4 * sigma_x1**2) - np.sin(2 * theta11) / (4 * sigma_y1**2)
c11 = np.sin(theta11)**2 / (2 * sigma_x1**2) + np.cos(theta11)**2 / (2 * sigma_y1**2)

coords = np.zeros([samNum1, 3])
coords[:,0] = spConst1 * (np.cos(r) + r * np.sin(r)) # x coord
coords[:,1] = spConst1 * (np.sin(r) - r * np.cos(r)) # y coord
coords[:,2] = amp1 * np.exp(-(a11 * coords[:,0]**2 - 2 * b11 * coords[:,0]*coords[:,1] + c11 * coords[:,1]**2))

ZZ[1,:] = coords[:,2]

#ax.plot_surface(X,Y,Z)
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='black')


spConst2 = -10.0
t11 = np.linspace(0,89*np.pi, samNum1)

T1, Z11 = np.meshgrid(t11, [0,1])
X1 = spConst2 * (np.cos(T1) + T1* np.sin(T1))
Y1 = spConst2 * (np.sin(T1) - T1 * np.cos(T1))


sigma_x = -0.75
sigma_y = -0.75
theta111 = np.pi
a1111 = np.cos(theta111)**2 / (2 * sigma_x**2) + np.sin(theta111)**2 / (2 * sigma_y**2)
b1111 = - np.sin(2 * theta111) / (4 * sigma_x**2) - np.sin(2 * theta111) / (4 * sigma_y**2)
c1111 = np.sin(theta111)**2 / (2 * sigma_x**2) + np.cos(theta111)**2 / (2 * sigma_y**2)

coords = np.zeros([samNum1, 3])
coords[:,0] = spConst2 * (np.cos(t11) + t11 * np.sin(t11)) # x coord
coords[:,1] = spConst2 * (np.sin(t11) - t11 * np.cos(t11)) # y coord
coords[:,2] = amp1 * np.exp(-(a1111 * coords[:,0]**2 - 2 * b1111 * coords[:,0]*coords[:,1] + c1111 * coords[:,1]**2))

Z11[1,:] = coords[:,2]

#ax.plot_surface(X,Y,Z)
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='black')


plt.show()
