import numpy as np
import matplotlib
from matplotlib import pylab,mlab,pyplot,cm
plt = pyplot
from mpl_toolkits.mplot3d import Axes3D
from math import exp,sin,cos
from pylab import *

l = []
n=9
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



##data=np.arange(16).reshape((4,4))
##plt.imshow(data)




a=0.05
b=0.10
z = np.linspace(-16, 20, 100)
theta = np.linspace(0,50, 100)
##r = 1
y = a*exp(b*theta)*cos(theta)   ##- r * np.cos(theta) ## np.linspace(-1, 1, 100)##
x = (a*exp(b*theta)*sin(theta))**1.5   #- (1 - x**2)**0.5 + 1##x * np.sin(theta) ##
THETA = np.linspace(50, 0, 100)
Z = np.linspace(20, -16, 100)
##R = 1
Y =  - (a*exp(b*THETA)*cos(THETA))**1.5  ##np.linspace(1, -1, 100) ##
X =  - a*exp(b*THETA)*sin(THETA) ##R X *  np.sin(theta) ##
if (l[0]=='A'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='red',s=1)
        ax.text(0.75, 1, -16, "ADENINE", color='red')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -16, "THYMINE", color='blue')
        z14 = np.linspace(-16, -12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(-16, -12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[0]=='T'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='blue',s=1)
        ax.text(0.75, 1, -16, "THYMINE", color='blue')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -16, "ADENINE", color='red')
        z14 = np.linspace(-16, -12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(-16, -12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[0]=='G'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='green',s=1)
        ax.text(0.75, 1, -16, "GUANINE", color='green')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -16, "CYTOSINE", color='yellow')
        z14 = np.linspace(-16, -12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(-16, -12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[0]=='C'):
        z1 = np.linspace(-16, -12, 100)
        ax.scatter(x,y,z1,c='yellow',s=1)
        ax.text(0.75, 1, -16, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-16, -12, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -16,  "GUANINE", color='green')
        z14 = np.linspace(-16, -12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(-16, -12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[1]=='A'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='red',s=1)
        ax.text(0.75, 1, -12, "ADENINE", color='red')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -12, "THYMINE", color='blue')
        z14 = np.linspace(-12, -8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(-12, -8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[1]=='T'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='blue',s=1)
        ax.text(0.75, 1, -12, "THYMINE", color='BLUE')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -12, "ADENINE", color='red')
        z14 = np.linspace(-12, -8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(-12, -8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[1]=='G'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='green',s=1)
        ax.text(0.75, 1, -12, "GUANINE", color='green')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -12, "CYTOSINE", color='yellow')
        z14 = np.linspace(-12, -8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(-12, -8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[1]=='C'):
        z2 = np.linspace(-12, -8, 100)
        ax.scatter(x,y,z2,c='yellow',s=1)
        ax.text(0.75, 1, -12, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-12, -8, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -12, "GUANINE", color='green')
        z14 = np.linspace(-12, -8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(-12, -8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[2]=='A'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='red',s=1)
        ax.text(0.75, 1, -8, "ADENINE", color='red')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -8, "THYMINE", color='blue')
        z14 = np.linspace(-8, -4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(-8, -4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[2]=='T'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='blue',s=1)
        ax.text(0.75, 1, -8, "THYMINE", color='BLUE')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -8, "ADENINE", color='red')
        z14 = np.linspace(-8, -4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(-8, -4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[2]=='G'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='green',s=1)
        ax.text(0.75, 1, -8, "GUANINE", color='green')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -8, "CYTOSINE", color='yellow')
        z14 = np.linspace(-8, -4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(-8, -4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[2]=='C'):
        z3 = np.linspace(-8, -4, 100)
        ax.scatter(x,y,z3,c='yellow',s=1)
        ax.text(0.75, 1, -8, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-8, -4, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -8, "GUANINE", color='green')
        z14 = np.linspace(-8, -4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(-8, -4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[3]=='A'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='red',s=1)
        ax.text(0.75, 1, -4, "ADENINE", color='red')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, -4, "THYMINE", color='blue')
        z14 = np.linspace(-4, 0, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(-4, 0, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[3]=='T'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='blue',s=1)
        ax.text(0.75, 1, -4, "THYMINE", color='blue')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, -4, "ADENINE", color='red')
        z14 = np.linspace(-4, 0, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(-4, 0, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[3]=='G'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='green',s=1)
        ax.text(0.75, 1, -4, "GUANINE", color='green')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -4, "CYTOSINE", color='yellow')
        z14 = np.linspace(-4, 0, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(-4, 0, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[3]=='C'):
        z4 = np.linspace(-4, 0, 100)
        ax.scatter(x,y,z4,c='yellow',s=1)
        ax.text(0.75, 1, -4, "CYTOSINE", color='yellow')
        Z1 = np.linspace(-4, 0, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, -4, "GUANINE", color='green')
        z14 = np.linspace(-4, 0, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(-4, 0, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[4]=='A'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='red',s=1)
        ax.text(0.75, 1, 0, "ADENINE", color='red')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 0, "THYMINE", color='blue')
        z14 = np.linspace(0, 4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(0, 4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[4]=='T'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='blue',s=1)
        ax.text(0.75, 1, 0, "THYMINE", color='BLUE')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 0, "ADENINE", color='red')
        z14 = np.linspace(0, 4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(0, 4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[4]=='G'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='green',s=1)
        ax.text(0.75, 1, 0, "GUANINE", color='green')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, -0, "CYTOSINE", color='yellow')
        z14 = np.linspace(0, 4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(0, 4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[4]=='C'):
        z5 = np.linspace(0, 4, 100)
        ax.scatter(x,y,z5,c='yellow',s=1)
        ax.text(0.75, 1, -0, "CYTOSINE", color='yellow')
        Z1 = np.linspace(0, 4, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 0, "GUANINE", color='green')
        z14 = np.linspace(0, 4, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(0, 4, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[5]=='A'):
        z6 = np.linspace(4, 8, 100)
        ax.scatter(x,y,z6,c='red',s=1)
        ax.text(0.75, 1, 4, "ADENINE", color='red')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 4, "THYMINE", color='blue')
        z14 = np.linspace(4, 8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(4, 8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[5]=='T'):
        z6 = np.linspace(4, 8, 100)
        ax.scatter(x,y,z6,c='blue',s=1)
        ax.text(0.75, 1, 4, "THYMINE", color='BLUE')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 4, "ADENINE", color='red')
        z14 = np.linspace(4, 8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(4, 8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[5]=='G'):
        z6 = np.linspace(4, 8, 100)
        ax.scatter(x,y,z6,c='green',s=1)
        ax.text(0.75, 1, 4, "GUANINE", color='green')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, 4, "CYTOSINE", color='yellow')
        z14 = np.linspace(4, 8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(4, 8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[5]=='C'):
        z6 = np.linspace(4, 8,100)
        ax.scatter(x,y,z6,c='yellow',s=1)
        ax.text(0.75, 1, 4, "CYTOSINE", color='yellow')
        Z1 = np.linspace(4, 8, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 4, "GUANINE", color='green')
        z14 = np.linspace(4, 8, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(4, 8, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
if (l[6]=='A'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='red',s=1)
        ax.text(0.75, 1, 8, "ADENINE", color='red')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 8, "THYMINE", color='blue')
        z14 = np.linspace(8,12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(8, 12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[6]=='T'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='blue',s=1)
        ax.text(0.75, 1, 8, "THYMINE", color='BLUE')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 8, "ADENINE", color='red')
        z14 = np.linspace(8,12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(8, 12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[6]=='G'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='green',s=1)
        ax.text(0.75, 1, 8, "GUANINE", color='green')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, 8, "CYTOSINE", color='yellow')
        z14 = np.linspace(8,12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(8, 12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[6]=='C'):
        z7 = np.linspace(8, 12, 100)
        ax.scatter(x,y,z7,c='yellow',s=1)
        ax.text(0.75, 1, 8, "CYTOSINE", color='yellow')
        Z1 = np.linspace(8, 12, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 8, "GUANINE", color='green')
        z14 = np.linspace(8,12, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(8, 12, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[7]=='A'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='red',s=1)
        ax.text(0.75, 1, 12, "ADENINE", color='red')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 12, "THYMINE", color='blue')
        z14 = np.linspace(12,16, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(12, 16, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[7]=='T'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='blue',s=1)
        ax.text(0.75, 1, 12, "THYMINE", color='BLUE')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 12, "ADENINE", color='red')
        z14 = np.linspace(12,16, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(12, 16, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[7]=='G'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='green',s=1)
        ax.text(0.75, 1, 12, "GUANINE", color='green')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, 12, "CYTOSINE", color='yellow')
        z14 = np.linspace(12,16, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(12, 16, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[7]=='C'):
        z8 = np.linspace(12, 16, 100)
        ax.scatter(x,y,z8,c='yellow',s=1)
        ax.text(0.75, 1, 12, "CYTOSINE", color='yellow')
        Z1 = np.linspace(12, 16, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 12, "GUANINE", color='green')
        z14 = np.linspace(12,16, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(12, 16, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)
if (l[8]=='A'):
        z8 = np.linspace(16, 20, 100)
        ax.scatter(x,y,z8,c='red',s=1)
        ax.text(0.75, 1, 16, "ADENINE", color='red')
        Z1 = np.linspace(16, 20, 100)
        ax.scatter(X,Y,Z1,c='blue',s=1)
        ax.text(-5, -5, 16, "THYMINE", color='blue')
        z14 = np.linspace(16,20, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='red',s=1)
        z122 = np.linspace(16, 20, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='blue',s=1)
elif (l[8]=='T'):
        z8 = np.linspace(16, 20, 100)
        ax.scatter(x,y,z8,c='blue',s=1)
        ax.text(0.75, 1, 16, "THYMINE", color='BLUE')
        Z1 = np.linspace(16, 20, 100)
        ax.scatter(X,Y,Z1,c='red',s=1)
        ax.text(-5, -5, 16, "ADENINE", color='red')
        z14 = np.linspace(16,20, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='blue',s=1)
        z122 = np.linspace(16, 20, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='red',s=1)
elif (l[8]=='G'):
        z8 = np.linspace(16, 20, 100)
        ax.scatter(x,y,z8,c='green',s=1)
        ax.text(0.75, 1, 16, "GUANINE", color='green')
        Z1 = np.linspace(16, 20, 100)
        ax.scatter(X,Y,Z1,c='yellow',s=1)
        ax.text(-5, -5, 16, "CYTOSINE", color='yellow')
        z14 = np.linspace(16,20, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='green',s=1)
        z122 = np.linspace(16, 20, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='yellow',s=1)
elif (l[8]=='C'):
        z8 = np.linspace(16, 20,100)
        ax.scatter(x,y,z8,c='yellow',s=1)
        ax.text(0.75, 1, 16, "CYTOSINE", color='yellow')
        Z1 = np.linspace(16, 20, 100)
        ax.scatter(X,Y,Z1,c='green',s=1)
        ax.text(-5, -5, 16, "GUANINE", color='green')
        z14 = np.linspace(16,20, 100)
        x14=cos(z14)/4
        y14=sin(z14)/4
        ax.scatter(x14,y14,z14,c='yellow',s=1)
        z122 = np.linspace(16, 20, 100)
        x122=-cos(z122)/4
        y122=-sin(z122)/4
        ax.scatter(x122,y122,z122,c='green',s=1)

plt.show()
