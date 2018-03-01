import numpy as np
import matplotlib
from matplotlib import pylab,mlab,pyplot,cm
plt = pyplot
from mpl_toolkits.mplot3d import Axes3D

l = []
n=8
print('Add a nucleotids, please')
for i in range(n):
        l.append(input())
print(l)
def trans(l):
    l2=[]
    for i in range(n):
        if l[i]== 'A':
            l2.append('T')
        if l[i]== 'T':
            l2.append('A')
        if l[i]== 'G':
            l2.append('C')
        if l[i]== 'C':
            l2.append('G')
    return l2
print(trans(l))


IMAGE = plt.figure()
ax = IMAGE.add_subplot(111, projection='3d')
samNum1 = 1000
spConst1 = -5.0
t = np.linspace(0,89*np.pi, samNum1)

T, z = np.meshgrid(t, [0,1])
x = spConst1 * (np.cos(T) + T* np.sin(T))
y = spConst1 * (np.sin(T) - T * np.cos(T))

# Coordinates of involute spiral on xy-plane
coords = np.zeros([samNum1, 3])
coords[:,0] = spConst1 * (np.cos(t) + t * np.sin(t)) # x coord
coords[:,1] = spConst1 * (np.sin(t) - t * np.cos(t))
# Paramters for 2D Gaussian surface
amp = 200
sigma_x = -75.0
sigma_y = -75.0
theta = np.pi
a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
b = np.sin(2 * theta) / (4 * sigma_x**2) - np.sin(2 * theta) / (4 * sigma_y**2)
c = -np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
# z coords of spiral projected onto Gaussian surface
coords[:,2] = amp * np.exp(-(a * coords[:,0]**2 - 2 * b * coords[:,0]*coords[:,1] + c * coords[:,1]**2)) # z coord
z[1,:] = coords[:,2]
ax.plot_surface(x,y,z)
# plot 3D spiral
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='k')
samNum = 1000
spConst = -5.0
coords = np.zeros([samNum, 3])
coords[:,0] = spConst * (np.cos(t) + t * np.sin(t)) # x coord
coords[:,1] = spConst * (np.sin(t) - t * np.cos(t)) # y coord


samNum1 = 1000
spConst1 = 5.0
t = np.linspace(0,89*np.pi, samNum1)

T, Z = np.meshgrid(t, [0,1])
X = spConst1 * (np.cos(T) + T* np.sin(T))
Y = spConst1 * (np.sin(T) - T * np.cos(T))

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

Z[1,:] = coords[:,2]
ax.plot_surface(X,Y,Z)

# plot 3D spiral
ax.scatter(coords[:,0], coords[:,0], coords[:,2], s=1, c='k')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')



if (l[0]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        Z = np.meshgrid(t, [0,1])
        ax.scatter(x,y,Z,c='red',s=1)
        ax.text(0.75, 1, -16, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, -16, "THYMINE", color='blue')
elif (l[0]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        Z = np.meshgrid(t, [0,1])
        ax.scatter(x,y,Z,c='blue',s=1)
        ax.text(0.75, 1, -16, "THYMINE", color='blue')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, -16, "ADENINE", color='red')
elif (l[0]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z1 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z1,c='green',s=1)
        ax.text(0.75, 1, -16, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, -16, "CYTOSINE", color='yellow')
elif (l[0]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z1 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z1,c='yellow',s=1)
        ax.text(0.75, 1, -16, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, 1, -16,  "GUANINE", color='green')
if (l[1]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        z2 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z2,c='red',s=1)
        ax.text(0.75, 1, -12, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, -12, "THYMINE", color='blue')
elif (l[1]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        z2 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z2,c='blue',s=1)
        ax.text(0.75, 1, -12, "THYMINE", color='BLUE')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, -12, "ADENINE", color='red')
elif (l[1]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z2 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z2,c='green',s=1)
        ax.text(0.75, 1, -12, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, -12, "CYTOSINE", color='yellow')
elif (l[1]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z2 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z2,c='yellow',s=1)
        ax.text(0.75, 1, -12, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, -12, "GUANINE", color='green')
if (l[2]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        z3 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z3,c='red',s=1)
        ax.text(0.75, 1, -8, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, -8, "THYMINE", color='blue')
elif (l[2]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        z3 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z3,c='blue',s=1)
        ax.text(0.75, 1, -8, "THYMINE", color='BLUE')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, -8, "ADENIN", color='red')
elif (l[2]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z3 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z3,c='green',s=1)
        ax.text(0.75, 1, -8, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, -8, "CYTOSINE", color='yellow')
elif (l[2]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z3 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z3,c='yellow',s=1)
        ax.text(0.75, 1, -8, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, -8, "GUANINE", color='green')
if (l[3]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        z4 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z4,c='red',s=1)
        ax.text(0.75, 1, -4, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, -4, "THYMINE", color='blue')
elif (l[3]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        z4 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z4,c='blue',s=1)
        ax.text(0.75, 1, -4, "THYMINE", color='blue')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, -4, "ADENIN", color='red')
elif (l[3]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z4 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z4,c='green',s=1)
        ax.text(0.75, 1, -4, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, -4, "CYTOSINE", color='yellow')
elif (l[3]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z4 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z4,c='yellow',s=1)
        ax.text(0.75, 1, -4, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, -4, "GUANINE", color='green')
if (l[4]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        z5 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z5,c='red',s=1)
        ax.text(0.75, 1, 0, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, 0, "THYMINE", color='blue')
elif (l[4]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        z5 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z5,c='blue',s=1)
        ax.text(0.75, 1, 0, "THYMINE", color='BLUE')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, 0, "ADENIN", color='red')
elif (l[4]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z5 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z5,c='green',s=1)
        ax.text(0.75, 1, 0, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, -0, "CYTOSINE", color='yellow')
elif (l[4]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z5 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z5,c='yellow',s=1)
        ax.text(0.75, 1, -0, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, 0, "GUANINE", color='green')
if (l[5]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        z6 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z6,c='red',s=1)
        ax.text(0.75, 1, 4, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, 4, "THYMINE", color='blue')
elif (l[5]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        z6 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z6,c='blue',s=1)
        ax.text(0.75, 1, 4, "THYMINE", color='BLUE')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, 4, "ADENIN", color='red')
elif (l[5]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z6 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z6,c='green',s=1)
        ax.text(0.75, 1, 4, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1 -1, 4, "CYTOSINE", color='yellow')
elif (l[5]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z6 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z6,c='yellow',s=1)
        ax.text(0.75, 1, 4, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, 4, "GUANINE", color='green')
if (l[6]=='A'):
        t = np.linspace(0,89*np.pi, samNum)
        z7 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z7,c='red',s=1)
        ax.text(0.75, 1, 8, "ADENIN", color='red')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, 8, "THYMINE", color='blue')
elif (l[6]=='T'):
        t = np.linspace(0,89*np.pi, samNum)
        z7 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z7,c='blue',s=1)
        ax.text(0.75, 1, 8, "THYMINE", color='BLUE')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, 8, "ADENIN", color='red')
elif (l[6]=='G'):
        t = np.linspace(0,89*np.pi, samNum)
        z7 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z7,c='green',s=1)
        ax.text(0.75, 1, 8, "GUANINE", color='green')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, 8, "CYTOSINE", color='yellow')
elif (l[6]=='C'):
        t = np.linspace(0,89*np.pi, samNum)
        z7 = np.meshgrid(t, [0,1])
        ax.scatter(x,y,z7,c='yellow',s=1)
        ax.text(0.75, 1, 8, "CYTOSINE", color='yellow')
        Z1 = np.meshgrid(t, [0,1])
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, 8, "GUANINE", color='green')
if (l[7]=='A'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='red',s=1)
        ax.text(0.75, 1, 12, "ADENIN", color='red')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-1, -1, 12, "THYMINE", color='blue')
elif (l[7]=='T'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='blue',s=1)
        ax.text(0.75, 1, 12, "THYMINE", color='BLUE')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-1, -1, 12, "ADENINE", color='red')
elif (l[7]=='G'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='green',s=1)
        ax.text(0.75, 1, 16, "GUANINE", color='green')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-1, -1, 12, "CYTOSINE", color='yellow')
elif (l[7]=='C'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='yellow',s=1)
        ax.text(0.75, 1, 16, "CYTOSINE", color='yellow')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-1, -1, 12, "GUANINE", color='green')

plt.show()
