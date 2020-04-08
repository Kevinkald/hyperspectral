from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']

I = copy.deepcopy(HICO_original)
[H,W,L] = I.shape

X = np.reshape(I, [H*W,L])
print("shape X", X.shape)

N_CLUSTERS = 9

kmeans = KMeans(n_clusters=N_CLUSTERS).fit(X)
labels = kmeans.labels_	

print(kmeans.cluster_centers_.shape)

imag = np.zeros((500,500))

for i in range(0,500):
	for j in range(0,500):
		imag[i,j] = labels[i*500+j]

#(y,x)
print("south", imag[70,100])
print("-", imag[30,100])
print("north", imag[10,100])
print(imag)
plt.figure(1)

plt.subplot(121)
#(x,y)
#plt.plot(100,70,'ro')
#plt.plot(100,30,'bo')
#plt.plot(100,10,'go')
plt.imshow(imag)


plt.subplot(122)
for i in range(0,N_CLUSTERS):
	plt.plot(hico_wl, kmeans.cluster_centers_[i,:])

plt.xlabel("Wavelenght [nm]")
plt.ylabel("Radiance [$W/m^{-2}\mu m^{-1}sr^{-1}$]")
plt.legend(np.arange(N_CLUSTERS));
plt.show()