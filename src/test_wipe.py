import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from copy import copy
#fig, ax = plt.subplots(figsize = (14, 6))
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 7), constrained_layout=True)


xs = np.linspace(60, 70, 100)
ys = np.linspace(0, 100, 100)


X,Y = np.meshgrid(xs,ys)
s = X.shape
segs = np.empty(((s[0])*(s[1]-1),2,2))
segs[:,0,0] = X[:,:-1].flatten()
segs[:,1,0] = X[:,1:].flatten()
segs[:,0,1] = Y[:,:-1].flatten()
segs[:,1,1] = Y[:,1:].flatten()

lines = LineCollection(segs, linewidth = 1, cmap = plt.cm.Greys_r)
lines.set_array(X[:,:-1].flatten())

artists_list = [copy(lines) for i in range(3)]

for i, ax in enumerate(axes.flat):
    ax.add_collection(artists_list[i])
    ax.set_facecolor('k')
    ax.set_xlim(60, 70)
    ax.set_ylim(0, 100)

plt.show()