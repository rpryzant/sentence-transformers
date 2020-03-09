# python analyze_output.py PC_K_SEARCH.tsv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys

out = open(sys.argv[1])


neither = {}
cluster = {}
both = {}

for l in out:
	parts = l.strip().split()
	
	npc, k, strategy, perf = parts
	npc = int(npc)
	k = int(k)
	perf = float(perf)

	if strategy == 'neither':
		neither[npc, k] = float(perf)
	elif strategy == 'cluster':
		cluster[npc, k] = float(perf)
	elif strategy == 'both':
		both[npc, k] = float(perf)


# NO CLUSTER

z = [0.727,0.727214876,0.7152655359,0.7284281565,0.7339754778,0.7381318628,0.7420014259,0.7438167469,0.7457920793,0.7472631649,0.7484178994,0.750018699,0.7502452064,0.7511343342,0.7518406491,0.7508790835,0.7504608594,0.7509443886,0.7509945449,0.7527338166,0.7532067664,0.7539501733,0.754611266,0.7539467509,0.7551188891,0.7551283232,0.7547701806,0.755556689,0.7551001074,0.7543109587,0.7555731314,0.7554983687,0.7549806184,0.755206495,0.7549156992,0.7547809191,0.754744526,0.7538169766,0.7532628667,0.7533888326,0.7526495286,0.7525812553,0.7522057227,0.7518111431,0.7515425844,0.750926382,0.7501223675,0.7499486714,0.7496674775,0.7491264767,0.7483138962,0.747651285,0.7475052923,0.7468515602,0.7460177454,0.745628432,0.7453235232,0.745143828,0.7443801415,0.7434312616,0.7427133477,0.74180734,0.74130219,0.7408670619,0.7397027048,0.7391060649,0.7378396489,0.7369549247,0.7362053785,0.7352816038,0.7344561549,0.7334348212,0.7331095916]
L = len(z)
z = z + z
x = list(range(L))
x = x + x
y = ([0] * L) + ([20] * L)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)


# CLUSTERING

def get_xyz(d):

	maxv = 0
	maxk = None
	x = []
	y = []
	z = []
	for k, v in d.items():
		x.append(k[0])
		y.append(k[1])
		z.append(v)

		if v > maxv:
			maxv = v
			maxk = k

	return x, y, z


# fig = plt.figure()
# ax = fig.gca(projection='3d')
x, y, z = get_xyz(neither)
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
# plt.show()



x, y, z = get_xyz(both)
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

x, y, z = get_xyz(cluster)
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

plt.show()

# quit()

# quit()

# print maxes

maxv = 0
maxk = None
for k, v in neither.items():
	if v > maxv:
		maxv = v
		maxk = k
print(maxv, maxk)



maxv = 0
maxk = None
for k, v in cluster.items():
	if v > maxv:
		maxv = v
		maxk = k
print(maxv, maxk)


maxv = 0
maxk = None
for k, v in both.items():
	if v > maxv:
		maxv = v
		maxk = k
print(maxv, maxk)


