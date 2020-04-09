from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy
#from estimating_reflectance_surface import atmosphere_correct
#from chlorophyll_estimation import calculate_chlorophyl_image

import estimating_reflectance_surface as est
import chlorophyll_estimation as chl
import k_means as kmean

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']
deep_water_Rrs    = np.loadtxt('../data/deep_water_Rrs.txt')
shallow_water_Rrs = np.loadtxt('../data/shallow_water_Rrs.txt')
valid_bands_Rrs   = np.loadtxt('../data/valid_bands_Rrs.txt')
land_mask         = plt.imread('../data/land_mask.png')[:,:,0] == 0

I = copy.deepcopy(HICO_original)
I_corrected = copy.deepcopy(I)
[H,W,L] = I.shape

deep_water_x = 20
deep_water_y = 20
shallow_water_x = 100
shallow_water_y = 70



# correct image
I_corrected = est.atmosphere_correct(I_corrected)
# Calculate chlorophyll concentration
CL = chl.calculate_chlorophyl_image(I_corrected)
# Mask out land area
CL[land_mask] = 0

# kmeans
N_CLUSTERS = 9
ONLY_VALID_BANDS = 1
imag, kmeans, hico_wl = kmean.calculate_k_means(I_corrected, N_CLUSTERS, ONLY_VALID_BANDS)


plt.figure()
plt.imshow(CL)
plt.show()

plt.figure()
plt.subplot(121)
plt.imshow(imag)
plt.subplot(122)
for i in range(0, N_CLUSTERS):
		plt.plot(hico_wl, kmeans.cluster_centers_[i,:])
plt.xlabel("Wavelenght [nm]")
plt.ylabel("Radiance [$W/m^{-2}\mu m^{-1}sr^{-1}$]")
plt.legend(np.arange(N_CLUSTERS));
plt.show()