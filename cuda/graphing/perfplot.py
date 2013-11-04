"""
Plotting the performance of the GPU
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# X -> number of iterations; Y -> array size
with open('test.txt') as f:
  Xlog, Ylog = [int(x) for x in f.readline().split()] # first line
  Xbeg, Xend, Xstep, Xlen = [int(x) for x in f.readline().split()] # second line
  Ybeg, Yend, Ystep, Ylen = [int(x) for x in f.readline().split()] # third line
  flops = []
  for line in f: # read rest of lines
    flops.append([int(x) for x in line.split()])

if (Xlog):
  X = np.zeros(Xlen)
  X[0] = Xbeg
  for i in range(1,Xlen):
    X[i] = X[i-1]*Xstep
else:
  X = np.arange(Xbeg, Xend+1, Xstep)
if (Ylog):
  Y = np.zeros(Xlen)
  Y[0] = Ybeg
  for i in range(1,Ylen):
    Y[i] = Y[i-1]*Ystep
  print Y
else:
  Y = np.arange(Ybeg, Yend+1, Ystep)
X, Y = np.meshgrid(X, Y)

# Z -> number of flops
Z = []
k = 0
for i in xrange(len(X)):
  Z.append([])
  for j in xrange(len(X[0])):
    #Z[i].append(1)
    Z[i] = Z[i] + flops[k]
    k += 1

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,linewidth=0, antialiased=False)
plt.title('GPU Flops')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Array Size')
ax.set_zlabel('Flops')
plt.show()
