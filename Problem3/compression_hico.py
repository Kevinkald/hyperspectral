from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import pca as pca

def show_image(X):
    plt.imshow(np.reshape(X.T, [H,W,L]))

def error(X, X_ref):
    return np.mean(np.linalg.norm(X - X_ref, axis=0)) / \
        np.mean(np.linalg.norm(X_ref, axis=0))

D = loadmat("../data/HICO.mat")
HICO_original = D['HICO_original']
HICO_noisy = D['HICO_noisy']
H,W,L      = HICO_original.shape
X          = np.reshape(HICO_noisy, [H*W,L]).T
X_original = np.reshape(HICO_original, [H*W,L]).T
sigma      = np.cov(X)

print(sigma.shape)

N = W*H
X_n = np.zeros((L, N))
for i in range(0, N-1):
	X_n[:,i] = X[:,i] - X[:,i+1]
sigma_n = np.cov(X_n)

P = 10
# Compute MNF reconstruction of X
eigs, V = np.linalg.eig(np.matmul(sigma_n, np.linalg.inv(sigma)))
# sort the eigs and corr V
idx = np.argsort(eigs)
eigs = eigs[idx]
V = V[:,idx]
print("eigs: ", eigs)
A_transpose = np.linalg.inv(V)
Y = np.matmul(A_transpose, X)
# Do transform
X_hat_mnf = np.matmul(V[:,0:P],Y[0:P,:])

# Compute the PCA reconstruction of X
pca = PCA(n_components=P)
X_pca = pca.fit_transform(X.T)
print(X_pca.shape)
X_hat_pca = pca.inverse_transform(X_pca).T
print(X_hat_pca.shape)

# Compute relative error (error), between 0 and 1
error_mnf = error(X_hat_mnf, X_original)
error_pca = error(X_hat_pca, X_original)
error_img = error(X,X_original)

print("error mnf: ", 100*error_mnf)
print("error_pca: ", 100*error_pca)
print(error_img*100)