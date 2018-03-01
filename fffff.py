import numpy as np
import matplotlib
from matplotlib import pylab,mlab,pyplot,cm
plt = pyplot
import pyfits as pf
from mpl_toolkits.mplot3d import Axes3D

def spiral(START, FINISH, color):
    z = np.linspace(START, FINISH, 1000)
    theta = np.linspace(-10 * np.pi, 10 * np.pi, 1000)
    r =  1
    x = r * np.sin(theta)
    y = - r * np.cos(theta)
    Cen3D = plt.figure()
    ax = Cen3D.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,c=color,s=1)

spiral(-10,-5,'red')
spiral(-5,-0,'blue')




plt.show()
