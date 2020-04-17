from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy

import pseudo_rgb as psdrgb

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']
deep_water_Rrs    = np.loadtxt('../data/deep_water_Rrs.txt')
shallow_water_Rrs = np.loadtxt('../data/shallow_water_Rrs.txt')
valid_bands_Rrs   = np.loadtxt('../data/valid_bands_Rrs.txt')
land_mask         = plt.imread('../data/land_mask.png')[:,:,0] == 0

I = copy.deepcopy(HICO_original)

[H,W,L] = I.shape

deep_water_x = 20
deep_water_y = 20
shallow_water_x = 100
shallow_water_y = 70

def calculate_ab(lambda_idx):
	# setup A matrix
	rho_deep = deep_water_Rrs[lambda_idx]
	rho_shallow = shallow_water_Rrs[lambda_idx]
	A = np.array([[rho_deep, 1],[rho_shallow, 1]])

	# setup y vector
	L_deep = I[deep_water_y, deep_water_x, lambda_idx]
	L_shallow = I[shallow_water_y, shallow_water_x,  lambda_idx]
	y = np.array([L_deep, L_shallow])
	y = y.reshape(2,1)

	# solve system to yield a and b
	x = np.linalg.inv(A).dot(y)
	return x

def atmosphere_correct(image):
	#calculate all a,b for valid indices
	x = np.zeros((100,2))
	for i in range(0,100):
		if valid_bands_Rrs[i] == 1:
			x[i,:] = calculate_ab(i).T
	# Correct the valid bands
	for i in range(0,H):
		print(i)
		for j in range(0,W):
			# Now we're at a pixel p, time to update the valid channels
			for k in range(0, 100):
				if valid_bands_Rrs[k] == 1:
					# Update the 
					image[i,j,k] = (image[i,j,k]-x[k,1])/x[k,0]
	return image

if __name__ == "__main__":
	
	rgb_original = psdrgb.pseudo_rgb(I)

	I_corrected = atmosphere_correct(I)
	rgb_corrected = psdrgb.pseudo_rgb(I_corrected)

	plt.figure()
	plt.subplot(121)
	plt.imshow(rgb_original)
	plt.subplot(122)
	plt.imshow(rgb_corrected)
	plt.show()