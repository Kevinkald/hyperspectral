from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']
valid_bands_Rrs   = np.loadtxt('../data/valid_bands_Rrs.txt')
land_mask = plt.imread('../data/land_mask.png')[:,:,0] == 0

I = copy.deepcopy(HICO_original)
[H,W,L] = I.shape

def calculate_k_means(image,k, only_valid_bands, mask_land):
	hico_wl = D['hico_wl']
	H,W,L = image.shape
	if (mask_land == 1):
		image[land_mask] = 0
	X = np.reshape(image, [H*W,L])

	if only_valid_bands==1:
		for i in range(L-1,-1,-1):
			# if should remove band
			if valid_bands_Rrs[i] == 0:
				X = np.delete(X, i, 1)
				hico_wl = np.delete(hico_wl, i, 0)

	kmeans = KMeans(n_clusters=k).fit(X)
	
	labels = kmeans.labels_	

	imag = np.zeros((H,W))

	for i in range(0,H):
		for j in range(0,W):
			imag[i,j] = labels[i*W+j]

	return imag, kmeans, hico_wl


if __name__ == "__main__":
	N_CLUSTERS = 8
	mask_land = 1
	imag, kmeans, hico_wl = calculate_k_means(I,N_CLUSTERS,0, mask_land)

	plt.figure(1)

	plt.subplot(121)
	#(x,y)
	#plt.plot(100,70,'ro')
	#plt.plot(100,30,'bo')
	#plt.plot(100,10,'go')
	plt.imshow(imag)


	plt.subplot(122)
	for i in range(0, N_CLUSTERS):
		plt.plot(hico_wl, kmeans.cluster_centers_[i,:])

	plt.xlabel("Wavelenght [nm]")
	plt.ylabel("Radiance [$W/m^{-2}\mu m^{-1}sr^{-1}$]")
	plt.legend(np.arange(N_CLUSTERS));
	plt.show()