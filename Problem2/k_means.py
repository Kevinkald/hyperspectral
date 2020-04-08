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

kmeans = KMeans(n_clusters=8).fit(X)
labels = kmeans.labels_

imag = np.zeros((500,500))

print(imag.shape)

for i in range(0,500):
	for j in range(0,500):
		imag[i,j] = labels[i*500+j]

plt.figure(1)
plt.imshow(imag)
plt.show()