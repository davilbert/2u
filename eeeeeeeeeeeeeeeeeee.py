import numpy as np
import matplotlib
from matplotlib import pylab,mlab,pyplot,cm
plt = pyplot
from mpl_toolkits.mplot3d import Axes3D
from math import exp,sin,cos
from pylab import *

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
a=0.05
b=0.10
z = np.linspace(-16, 16, 100)
theta = np.linspace(0,50, 100)
##r = 1
y = a*exp(b*theta)*cos(theta)   ##- r * np.cos(theta) ## np.linspace(-1, 1, 100)##
x = (a*exp(b*theta)*sin(theta))**1.5   #- (1 - x**2)**0.5 + 1##x * np.sin(theta) ##
THETA = np.linspace(50, 0, 100)
Z = np.linspace(16, -16, 100)
##R = 1
Y =  - (a*exp(b*THETA)*cos(THETA))**1.5  ##np.linspace(1, -1, 100) ##
X =  - a*exp(b*THETA)*sin(THETA) ##R X *  np.sin(theta) ##

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





if (l[0]=='A'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='red',s=1)
        ax.text(0.75, 1, -16, "ADENINE", color='red')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -16, "THYMINE", color='blue')
elif (l[0]=='T'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='blue',s=1)
        ax.text(0.75, 1, -16, "THYMINE", color='blue')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -16, "ADENINE", color='red')
elif (l[0]=='G'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='green',s=1)
        ax.text(0.75, 1, -16, "GUANINE", color='green')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -16, "CYTOSINE", color='yellow')
elif (l[0]=='C'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='yellow',s=1)
        ax.text(0.75, 1, -16, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -16,  "GUANINE", color='green')
if (l[1]=='A'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='red',s=1)
        ax.text(0.75, 1, -12, "ADENINE", color='red')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -12, "THYMINE", color='blue')
elif (l[1]=='T'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='blue',s=1)
        ax.text(0.75, 1, -12, "THYMINE", color='BLUE')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -12, "ADENINE", color='red')
elif (l[1]=='G'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='green',s=1)
        ax.text(0.75, 1, -12, "GUANINE", color='green')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -12, "CYTOSINE", color='yellow')
elif (l[1]=='C'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='yellow',s=1)
        ax.text(0.75, 1, -12, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -12, "GUANINE", color='green')
if (l[2]=='A'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='red',s=1)
        ax.text(0.75, 1, -8, "ADENINE", color='red')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -8, "THYMINE", color='blue')
elif (l[2]=='T'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='blue',s=1)
        ax.text(0.75, 1, -8, "THYMINE", color='BLUE')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -8, "ADENINE", color='red')
elif (l[2]=='G'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='green',s=1)
        ax.text(0.75, 1, -8, "GUANINE", color='green')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -8, "CYTOSINE", color='yellow')
elif (l[2]=='C'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='yellow',s=1)
        ax.text(0.75, 1, -8, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -8, "GUANINE", color='green')
if (l[3]=='A'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='red',s=1)
        ax.text(0.75, 1, -4, "ADENINE", color='red')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -4, "THYMINE", color='blue')
elif (l[3]=='T'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='blue',s=1)
        ax.text(0.75, 1, -4, "THYMINE", color='blue')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -4, "ADENINE", color='red')
elif (l[3]=='G'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='green',s=1)
        ax.text(0.75, 1, -4, "GUANINE", color='green')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -4, "CYTOSINE", color='yellow')
elif (l[3]=='C'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='yellow',s=1)
        ax.text(0.75, 1, -4, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -4, "GUANINE", color='green')
if (l[4]=='A'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='red',s=1)
        ax.text(0.75, 1, 0, "ADENINE", color='red')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 0, "THYMINE", color='blue')
elif (l[4]=='T'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='blue',s=1)
        ax.text(0.75, 1, 0, "THYMINE", color='BLUE')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 0, "ADENINE", color='red')
elif (l[4]=='G'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='green',s=1)
        ax.text(0.75, 1, 0, "GUANINE", color='green')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -0, "CYTOSINE", color='yellow')
elif (l[4]=='C'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='yellow',s=1)
        ax.text(0.75, 1, -0, "CYTOSINE", color='yellow')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 0, "GUANINE", color='green')
if (l[5]=='A'):
        z6 = np.linspace(4, 8, 100)
        ax.scatter(x,y,z6,c='red',s=1)
        ax.text(0.75, 1, 4, "ADENINE", color='red')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 4, "THYMINE", color='blue')
elif (l[5]=='T'):
        z6 = np.linspace(4, 8, 100)
        ax.scatter(x,y,z6,c='blue',s=1)
        ax.text(0.75, 1, 4, "THYMINE", color='BLUE')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 4, "ADENINE", color='red')
elif (l[5]=='G'):
        z6 = np.linspace(4, 8, 100)
        ax.scatter(x,y,z6,c='green',s=1)
        ax.text(0.75, 1, 4, "GUANINE", color='green')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5 -5, 4, "CYTOSINE", color='yellow')
elif (l[5]=='C'):
        z6 = np.linspace(4, 8,100)
        ax.scatter(x,y,z6,c='yellow',s=1)
        ax.text(0.75, 1, 4, "CYTOSINE", color='yellow')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 4, "GUANINE", color='green')
if (l[6]=='A'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='red',s=1)
        ax.text(0.75, 1, 8, "ADENINE", color='red')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 8, "THYMINE", color='blue')
elif (l[6]=='T'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='blue',s=1)
        ax.text(0.75, 1, 8, "THYMINE", color='BLUE')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 8, "ADENINE", color='red')
elif (l[6]=='G'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='green',s=1)
        ax.text(0.75, 1, 8, "GUANINE", color='green')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, 8, "CYTOSINE", color='yellow')
elif (l[6]=='C'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='yellow',s=1)
        ax.text(0.75, 1, 8, "CYTOSINE", color='yellow')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 8, "GUANINE", color='green')
if (l[7]=='A'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='red',s=1)
        ax.text(0.75, 1, 12, "ADENINE", color='red')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 12, "THYMINE", color='blue')
elif (l[7]=='T'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='blue',s=1)
        ax.text(0.75, 1, 12, "THYMINE", color='BLUE')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 12, "ADENINE", color='red')
elif (l[7]=='G'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='green',s=1)
        ax.text(0.75, 1, 12, "GUANINE", color='green')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, 12, "CYTOSINE", color='yellow')
elif (l[7]=='C'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='yellow',s=1)
        ax.text(0.75, 1, 12, "CYTOSINE", color='yellow')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 12, "GUANINE", color='green')

plt.show()
