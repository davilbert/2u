import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx

mpl.rcParams['legend.fontsize'] = 10

theta = np.linspace(-4 * np.pi, 4 * np.pi, 1000)
z = np.linspace(-2, 2, 1000)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
cs = 1/r

theta1 = np.linspace(4 * np.pi, -4 * np.pi, 1000)
z1 = np.linspace(2, -2, 1000)
r1 = z1**2 + 1
x1 = -r1 * np.sin(theta1)
y1 =- r1 * np.cos(theta1)
cs1 = 1/r1


colorsMap='jet'
cm = plt.get_cmap(colorsMap)
cNorm = mpl.colors.Normalize(vmin=min(cs), vmax=max(cs))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

fig = plt.figure()
scalarMap.set_array(cs)
fig1 = plt.figure()
scalarMap.set_array(cs1)

ax = fig.gca(projection='3d')
ax.scatter(x, y, z, c=scalarMap.to_rgba(cs), marker='_', s=1)
plt.colorbar(scalarMap)
ax1 = fig1.gca(projection='3d')
ax1.scatter(x1, y1, z1, c=scalarMap.to_rgba(cs1), marker='_', s=1)
plt.colorbar(scalarMap)
plt.show()

import random
#r=1=>massiv-random
#r=0=>massivmakesbyhands
class KM:
    count=0
    sumx=0
    def __init__(self,r,sl,t=0):  #t=0 because of python quarrels  witnout this
        self.r=r
        self.sl=sl
        self.t=prosto_hernya(self.r,self.sl)
        KM.count += 1
    @property
    def prosto_kakoito_kapets(self):
        return self.count
        #standart shit, but                                  #all right, but after!!!
    def riba_nevkusnaya(self,t,sl):
        sum=0
        #sum1=0
        for i in range(sl):
            for j in range(sl):
                sum=sum+t[i][i]
            #elif (i+j==2*i): # it is not main diagonal
                 #sum1=sum1+a[i][j]
    #def __str__(self):
        #for i in :
             #print(' '.join([str(j) for j in i]))
        return sum
    def uroboros(self,t):
        for i in range(sl):
            for j in range(sl):
                print(t[i][j], end=' ')
                print()
    def nu_ti_i_petushara(self,other):
        #s=0
        #s1=0
        #for i in range(sl):
            #for j in range(sl):
                #s=s+t[i][j]
        #for i in range(s2):
            #for j in range(s2):
                #s1=s1+b[i][j]
        if(self.sumx>other.sumx):
            print('first M is bigger')
        elif(self.sumx>other.sumx):
            print('second M is bigger')
        elif(self.sumx==other.sumx):
            print('same')
    def ti_cho_gay_BROOOO(self,t,sl):
        d = []
        for i,m in range(sl):
            for j,n in range(sl):
                if (i+j==m+n):
                    d=t[i][j]        # or  d.append(t[i][j])
                    e=t[n][m]     # or  e.append(t[n][m])
                    t[n][m]=d        # or t[n][m].append(d)
                    t[i][j]=e       # or  t[i][j].append(e)
        for f in range(sl):
            for x in range(sl):
                d[f][x]=t[f][x]  # or  d.append(t)  or d[f][x].append(t[f][x])
        return d
    def loshped(self,a):
        for i in a:
            print(' '.join([str(j) for j in i]))
    def shakal(self,t,b,sl,sl2):
        h=[]
        if (sl==sl2):
            for i,m,k in range(sl):
                for j,n,p in range(sl):
                    h[k][p].append(t[i][j]+b[m][n]) # or  h.append(t+b)
        else:
            print('ti tupoi chto li '
                  'matritsi takto raznogo razmera '
                  'nu kapets prosto slov net')
        return h
    def nu_prosto_na_desyatochku(self,t,sl):   # or k=prosto_hernya(r,sl)
        for i in range(sl):
            for j in range(sl):
                sumx=sum+t[i][j]          #sumx=sumx+t[i][j]
        return sumx
def prosto_hernya(r, s1):
    a = []
    g=[]
    if (r == 1):
        sumx=int(0)
        sumy=int(0)
        A = int(input())
        B = int(input())  # в результате работы данная функция
        for i in range(s1):  # должна возвращать список в атрибут класса
            for j in range(s1):
                e=random.randint(A,B)
                g.append(e)
                a.append(g)
                #sumx=sumx+a[i][j]
        #for i in a:
             #print(' '.join([str(j) for j in i]))
    else:
        for i in range(s1):
            for j in range(s1):
                e=int(input())
                g.append(e)
                a.append(g)
                #sumx=sumx+a[i][j]
        for i in a:
             print(' '.join([str(j) for j in i]))
    return a

k=KM(3,1)
l=KM(2,1)
