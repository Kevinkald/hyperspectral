from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import copy

import k_means as km

D = loadmat("../data/HICO.mat")
HICO_original = D['HICO_original']
land_mask         = plt.imread('../data/land_mask.png')[:,:,0] == 0


I = copy.deepcopy(HICO_original)

if __name__ == "__main__":

	P = 10
	[H,W,L] = I.shape
	I[land_mask] = 0
	I = np.reshape(I, [H*W,L])

	pca = PCA(n_components=P)
	X_pca = pca.fit_transform(I)
	X_pca = np.reshape(X_pca, [H,W,P])

	N_CLUSTERS = 8
	I, kmeans, wl = km.calculate_k_means(X_pca, N_CLUSTERS, 0, 0)
	print(I.shape)

	plt.figure(1)
	plt.imshow(I)
	plt.show()