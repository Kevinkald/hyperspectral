from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']

red = 570.19196
green = 535.824
blue = 432.72

for i in range(0, len(hico_wl)):
	if hico_wl[i] == red:
		red_image_index = i
	if hico_wl[i] == blue:
		blue_image_index = i
	if hico_wl[i] == green:
		green_image_index = i

#print(HICO_original)
#print(hico_wl)
I = copy.deepcopy(HICO_original)
K = copy.deepcopy(HICO_original)
[H,W,L] = I.shape

maximum = 0;
for i in range(0,H):
	for j in range(0,W):
		#for k in range(0,L):
			if I[i,j,red_image_index] > maximum:
				maximum = I[i,j,red_image_index]
			if I[i,j,green_image_index] > maximum:
				maximum = I[i,j,green_image_index]
			if I[i,j,blue_image_index] > maximum:
				maximum = I[i,j,blue_image_index]

# Normalizing
for i in range(0,H):
	for j in range(0,W):
		for k in range(0,L):
			I[i,j,k] = I[i,j,k]/maximum

for i in range(0,H):
	for j in range(0,W):
		tot = (K[i,j,red_image_index]+
			K[i,j,green_image_index]+K[i,j,blue_image_index])
		K[i,j,red_image_index] = K[i,j,red_image_index]/tot;
		K[i,j,green_image_index] = K[i,j,green_image_index]/tot;
		K[i,j,blue_image_index] = K[i,j,blue_image_index]/tot;

rgbArray = np.zeros((500,500,3))
rgbArray[..., 0] = I[:,:,red_image_index]
rgbArray[..., 1] = I[:,:,green_image_index]
rgbArray[..., 2] = I[:,:,blue_image_index]

rgbArray2 = np.zeros((500,500,3))
rgbArray2[..., 0] = K[:,:,red_image_index]
rgbArray2[..., 1] = K[:,:,green_image_index]
rgbArray2[..., 2] = K[:,:,blue_image_index]

# Plot a single spectral band
plt.figure(1)
plt.subplot(221)
plt.imshow(I[:,:,red_image_index])
plt.subplot(222)
plt.imshow(I[:,:,green_image_index])
plt.subplot(223)
plt.imshow(I[:,:,blue_image_index])
plt.subplot(224)
plt.imshow(rgbArray)
plt.show()

plt.figure(2)
plt.subplot(211)
plt.imshow(rgbArray)
plt.subplot(212)
plt.imshow(rgbArray2)
plt.show()