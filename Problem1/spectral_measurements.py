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

# Normalizing the rgb channels
for i in range(0,H):
	for j in range(0,W):
		I[i,j,red_image_index] = I[i,j,red_image_index]/maximum
		I[i,j,green_image_index] = I[i,j,green_image_index]/maximum
		I[i,j,blue_image_index] = I[i,j,blue_image_index]/maximum

# extract rgb image
rgbArray = np.zeros((500,500,3))
rgbArray[..., 0] = I[:,:,red_image_index]
rgbArray[..., 1] = I[:,:,green_image_index]
rgbArray[..., 2] = I[:,:,blue_image_index]

deep_water_x = 20
deep_water_y = 20
shallow_water_x = 100
shallow_water_y = 70
vegetation_x = 400
vegetation_y = 30

deep_water = HICO_original[deep_water_y, deep_water_x, :]
shallow_water = HICO_original[shallow_water_y, shallow_water_x, :]
vegetation = HICO_original[vegetation_y, vegetation_x, :]

plt.figure()
plt.subplot(121)
plt.plot([deep_water_x,shallow_water_x,vegetation_x],[deep_water_y,shallow_water_y,vegetation_y], 'ro')
plt.imshow(rgbArray)

plt.subplot(122)
ax1 = plt.plot(hico_wl, deep_water)
ax2 = plt.plot(hico_wl, shallow_water)
ax3 = plt.plot(hico_wl, vegetation)
plt.xlabel("Wavelenght [nm]")
plt.ylabel("Radiance [$W/m^{-2}\mu m^{-1}sr^{-1}$]")
plt.legend(["Deep water", "Shallow water", "Vegetation"])

plt.show()