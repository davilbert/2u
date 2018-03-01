import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fig_1 = plt.figure()
ax = fig_1.add_subplot(111, projection='3d')
samNum1 = 1000
spConst1 = 5.0
t = np.linspace(0,89*np.pi, samNum1)
T, Z = np.meshgrid(t, [0,1])
X = spConst1 * (np.cos(T) + T* np.sin(T))
Y = spConst1 * (np.sin(T) - T * np.cos(T))
amp1 = 200
sigma_x1 = 75.0
sigma_y1 = 75.0
theta1 = np.pi
a1 = np.cos(theta1)**2 / (2 * sigma_x1**2) + np.sin(theta1)**2 / (2 * sigma_y1**2)
b1 = - np.sin(2 * theta1) / (4 * sigma_x1**2) - np.sin(2 * theta1) / (4 * sigma_y1**2)
c1 = np.sin(theta1)**2 / (2 * sigma_x1**2) + np.cos(theta1)**2 / (2 * sigma_y1**2)

coords = np.zeros([samNum1, 3])
coords[:,0] = spConst1 * (np.cos(t) + t * np.sin(t)) # x coord
coords[:,1] = spConst1 * (np.sin(t) - t * np.cos(t)) # y coord
coords[:,2] = amp1 * np.exp(-(a1 * coords[:,0]**2 - 2 * b1 * coords[:,0]*coords[:,1] + c1 * coords[:,1]**2))

Z[1,:] = coords[:,2]

#ax.plot_surface(X,Y,Z)
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='black')




ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

spConst2 = -5.0
t1 = np.linspace(0,89*np.pi, samNum1)

T1, Z1 = np.meshgrid(t1, [0,1])
X1 = spConst2 * (np.cos(T1) + T1* np.sin(T1))
Y1 = spConst2 * (np.sin(T1) - T1 * np.cos(T1))


sigma_x = -75.0
sigma_y = -75.0
theta = np.pi
a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
b = - np.sin(2 * theta) / (4 * sigma_x**2) - np.sin(2 * theta) / (4 * sigma_y**2)
c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)

coords = np.zeros([samNum1, 3])
coords[:,0] = spConst2 * (np.cos(t1) + t1 * np.sin(t1)) # x coord
coords[:,1] = spConst2 * (np.sin(t1) - t1 * np.cos(t1)) # y coord
coords[:,2] = amp1 * np.exp(-(a * coords[:,0]**2 - 2 * b * coords[:,0]*coords[:,1] + c * coords[:,1]**2))

Z1[1,:] = coords[:,2]

#ax.plot_surface(X,Y,Z)
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='black')

plt.show()


