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

#data = loadmat("../data/task_3_case_1.mat") # Case 1
data = loadmat("../data/task_3_case_2.mat") # Case 2

I_noisy    = data['I_noisy']
I_original = data['I_original']
sigma_n    = data['Sigma_n']
H,W,L      = I_original.shape
X          = np.reshape(I_noisy, [H*W,L]).T
X_original = np.reshape(I_original, [H*W,L]).T
sigma      = np.cov(X)


print(sigma)
print(sigma_n)

# Todo: Compute the MNF reconstruction of X
eigs, V = np.linalg.eig(np.matmul(sigma_n, np.linalg.inv(sigma)))

# sort the eigs and corr V
idx = np.argsort(eigs)
eigs = eigs[idx]
V = V[:,idx]

print("eigs: ", eigs)

A_transpose = np.linalg.inv(V)
Y = np.matmul(A_transpose, X)

P = 2
X_hat_mnf = np.matmul(V[:,0:P],Y[0:P,:])

# Todo: Compute the PCA reconstruction of X
pca = PCA(n_components=P)
X_pca = pca.fit_transform(X.T)
X_hat_pca = pca.inverse_transform(X_pca).T

# Compute relative error (error), between 0 and 1
error_mnf = error(X_hat_mnf, X_original)
error_pca = error(X_hat_pca, X_original)

plt.subplot(221)
show_image(X_original)
plt.title('Original')
plt.subplot(222)
show_image(X)
plt.title('Noisy')
plt.subplot(223)
show_image(X_hat_mnf)
plt.title('MNF, error = %.1f%%' % (100*error_mnf))
plt.subplot(224)
show_image(X_hat_pca)
plt.title('PCA, error = %.1f%%' % (100*error_pca))
plt.tight_layout()
plt.show()
