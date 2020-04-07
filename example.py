from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

M             = loadmat('data/HICO.mat')
HICO_original = M['HICO_original'] # Hyperspectral image cube
HICO_noisy    = M['HICO_noisy']    # HICO_original with added noise
hico_wl       = M['hico_wl']       # Physical wavelength corresponding to band i

land_mask         = plt.imread('data/land_mask.png')[:,:,0] == 0
deep_water_Rrs    = np.loadtxt('data/deep_water_Rrs.txt')
shallow_water_Rrs = np.loadtxt('data/shallow_water_Rrs.txt')
valid_bands_Rrs   = np.loadtxt('data/valid_bands_Rrs.txt')

print(HICO_original.shape)
print(HICO_noisy.shape)
print(hico_wl.shape)
print(land_mask.shape)
print(deep_water_Rrs.shape)
print(shallow_water_Rrs.shape)
print(valid_bands_Rrs.shape)

# To convert a hyperspectral image cube I to matrix form X:
I = HICO_original
[H,W,L] = I.shape
X = np.reshape(I, [H*W,L])
X = X.T

# To convert a matrix X back into a hyperspectral image cube:
I = np.reshape(X.T, [H,W,L])

# To set all land pixels to zero:
I[land_mask] = 0

# Plot a single spectral band
plt.imshow(I[:,:,30])
plt.show()

# Note that quite a few libraries assume a matrix layout where
# each row is a spectral vector, rather than each column as in
# equation 2 of the assignment text. Read the documentation of
# those libraries carefully.
