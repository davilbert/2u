import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


fig_1 = plt.figure()
ax = fig_1.add_subplot(111, projection='3d')

# Spiral parameters
samNum1 = 1000
spConst1 = 5.0
t = np.linspace(0,89*np.pi, 1000)

T, Z = np.meshgrid(t, [0,1])
X = (spConst1 * (np.cos(T) + T* np.sin(T)))/1000
Y = (spConst1 * (np.sin(T) - T * np.cos(T)))/1000


# Coordinates of involute spiral on xy-plane
coords = np.zeros([samNum1, 3])
coords[:,0] = spConst1 * (np.cos(t) + t * np.sin(t)) # x coord
coords[:,1] = spConst1 * (np.sin(t) - t * np.cos(t)) # y coord


# Paramters for 2D Gaussian surface
amp1 = 20
sigma_x1 = 75
sigma_y1 = 75
theta1 = np.pi
a1 = np.cos(theta1)**2 / (2 * sigma_x1**2) + np.sin(theta1)**2 / (2 * sigma_y1**2)
b1 = np.sin(2 * theta1) / (4 * sigma_x1**2) - np.sin(2 * theta1) / (4 * sigma_y1**2)
c1 = -np.sin(theta1)**2 / (2 * sigma_x1**2) + np.cos(theta1)**2 / (2 * sigma_y1**2)

# z coords of spiral projected onto Gaussian surface
coords[:,2] = amp1 * np.exp(-(a1 * coords[:,0]**2 - 2 * b1 * coords[:,0]*coords[:,1] + c1 * coords[:,1]**2)) # z coord

Z[1,:] = coords[:,2]
#ax.plot_surface(X,Y,Z)

# plot 3D spiral
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='k')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

###third spiral


# Spiral parameters
samNum = 1000
spConst = -5.0
t = np.linspace(0,89*np.pi, 1000)

T, Z = np.meshgrid(t, [0,1])
X = (spConst * (np.cos(T) + T* np.sin(T)))/1000
Y = (spConst * (np.sin(T) - T * np.cos(T)))/1000

# Coordinates of involute spiral on xy-plane
coords = np.zeros([samNum, 3])
coords[:,0] = spConst * (np.cos(t) + t * np.sin(t)) # x coord
coords[:,1] = spConst * (np.sin(t) - t * np.cos(t)) # y coord

# Paramters for 2D Gaussian surface
amp = 20
sigma_x = 75
sigma_y = 75
theta = np.pi
a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
b = np.sin(2 * theta) / (4 * sigma_x**2) - np.sin(2 * theta) / (4 * sigma_y**2)
c = -np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)

# z coords of spiral projected onto Gaussian surface
coords[:,2] = amp * np.exp(-(a * coords[:,0]**2 - 2 * b * coords[:,0]*coords[:,1] + c * coords[:,1]**2)) # z coord

Z[1,:]= coords[:,2]
#ax.plot_surface(X,Y,Z)

# plot 3D spiral
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='k')

plt.show()


# Spiral parameters
#samNum = 1000
#spConst = 5.0
#t = np.linspace(0,89*np.pi, samNum)

#T, Z = np.meshgrid(t, [0,1])
#X = spConst * (np.cos(T) + T* np.sin(T))
#Y = spConst * (np.sin(T) - T * np.cos(T))

# Coordinates of involute spiral on xy-plane
#T, Z = np.meshgrid(t, [0,1])
#X = spConst * (np.cos(T) + T* np.sin(T))
#Y = spConst * (np.sin(T) - T * np.cos(T))

# Paramters for 2D Gaussian surface
#amp = 200
#sigma_x = 75.0
#sigma_y = 75.0
#theta = np.pi
#a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
#b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
#c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)

# z coords of spiral projected onto Gaussian surface
#coords[:,2] = amp * np.exp(-(a * coords[:,0]**2 - 2 * b * coords[:,0]*coords[:,1] + c * coords[:,1]**2)) # z coord

#Z[1,:] = coords[:,2]
#ax.plot_surface(X,Y,Z)

# plot 3D spiral
#ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=1, c='k')

#ax.set_xlabel('X axis')
#ax.set_ylabel('Y axis')
#ax.set_zlabel('Z axis')

###second spiral
