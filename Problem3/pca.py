from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import copy

import k_means as km

D = loadmat("../data/HICO.mat")
HICO_original = D['HICO_original']

I = copy.deepcopy(HICO_original)

if __name__ == "__main__":

	[H,W,L] = I.shape
	I = np.reshape(I, [H*W,L])

	pca = PCA(n_components=10)
	X_pca = pca.fit_transform(I)
	X_pca = np.reshape(X_pca, [H,W,10])

	N_CLUSTERS = 9
	MASK_LAND = 0
	I, kmeans, wl = km.calculate_k_means(X_pca, N_CLUSTERS, 0, MASK_LAND)
	print(I.shape)

	plt.figure(1)
	plt.imshow(I)
	plt.show()