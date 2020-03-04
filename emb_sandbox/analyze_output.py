import numpy as np

out = open('out.tsv')


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


maxv = 0
maxk = None
x = []
y = []
z = []
for k, v in neither.items():
	x.append(k[0])
	y.append(k[1])
	z.append(v)

	if v > maxv:
		maxv = v
		maxk = k
print(maxv, maxk)

print(x)
print(y)
print(z)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
plt.show()


quit()
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


