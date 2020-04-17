from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']


def pseudo_rgb(image):

	R_wl = 645
	G_wl = 510
	B_wl = 440

	H, W, L = HICO_original.shape

	pseudo_rgb_image = np.zeros((500,500,3))

	# locate indexes for rgb color
	R_i = np.argmin(np.abs(hico_wl - R_wl))
	G_i = np.argmin(np.abs(hico_wl - G_wl))
	B_i = np.argmin(np.abs(hico_wl - B_wl))

	maximum = 0;
	for i in range(0,H):
		for j in range(0,W):
				if image[i,j,R_i] > maximum:
					maximum = image[i,j,R_i]
				if image[i,j,G_i] > maximum:
					maximum = image[i,j,G_i]
				if image[i,j,B_i] > maximum:
					maximum = image[i,j,B_i]

	# Normalizing
	for i in range(0,H):
		for j in range(0,W):
			for k in range(0,L):
				image[i,j,k] = image[i,j,k]/maximum

	pseudo_rgb_image[:,:,0] = image[:,:,R_i]
	pseudo_rgb_image[:,:,1] = image[:,:,G_i]
	pseudo_rgb_image[:,:,2] = image[:,:,B_i]

	return pseudo_rgb_image

if __name__ == "__main__":
	img = pseudo_rgb(HICO_original)

	plt.figure()
	plt.imshow(img)
	plt.show()
