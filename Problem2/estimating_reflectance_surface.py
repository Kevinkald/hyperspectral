from sklearn.cluster import KMeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy

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

def calculate_ab(lambda_idx):
	# setup A matrix
	rho_shallow = shallow_water_Rrs[lambda_idx]
	rho_deep = deep_water_Rrs[lambda_idx]
	A = np.array([[rho_deep, 1],[rho_shallow, 1]])

	# setup y vector
	L_shallow = I[shallow_water_y, shallow_water_x, lambda_idx]
	L_deep = I[deep_water_y, deep_water_x, lambda_idx]
	y = np.array([L_deep, L_shallow])
	y = y.reshape(2,1)

	# solve system to yield a and b
	x = np.linalg.inv(A).dot(y)

	return x

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

x = np.zeros((100,2))
#calculate all a,b for valid indices
for i in range(0,100):
	if valid_bands_Rrs[i] == 1:
		temp_x = calculate_ab(i)
		print(temp_x)
		x[i,0] = temp_x[0]
		x[i,1] = temp_x[1]

for i in range(0,H):
	print(i)
	for j in range(0,W):
		# Now we're at a pixel p, time to update the valid channels
		for k in range(0, 100):
			if valid_bands_Rrs[k] == 1:
				# Update the 
				I_corrected[i,j,k] = (I_corrected[i,j,k]-x[k,1])/x[k,0]

#Finding max radiance in RGB bands
maximum = 0;
for i in range(0,H):
	for j in range(0,W):
		#for k in range(0,L):
			if I_corrected[i,j,red_image_index] > maximum:
				maximum = I_corrected[i,j,red_image_index]
			if I_corrected[i,j,green_image_index] > maximum:
				maximum = I_corrected[i,j,green_image_index]
			if I_corrected[i,j,blue_image_index] > maximum:
				maximum = I_corrected[i,j,blue_image_index]

# Normalizing(yes i normalize everything, but i am currently interested in RGB chanels)
for i in range(0,H):
	for j in range(0,W):
		for k in range(0,L):
			I_corrected[i,j,k] = I_corrected[i,j,k]/maximum


rgbArray = np.zeros((500,500,3))
rgbArray[..., 0] = I_corrected[:,:,red_image_index]
rgbArray[..., 1] = I_corrected[:,:,green_image_index]
rgbArray[..., 2] = I_corrected[:,:,blue_image_index]

plt.figure()
plt.imshow(rgbArray)
plt.show()