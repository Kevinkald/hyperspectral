from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']

a = [0.3272, -2.9940, 2.7218, -1.2259, -0.5683]

lambda_green = 553.008
lambda_blue = [444.176, 490.0, 507.18402]
lambda_green_idx = -1
lambda_blue_idx = [-1, -1, -1]

for i in range(0,len(hico_wl)):
	if hico_wl[i] == lambda_green:
		lambda_green_idx = i
	if hico_wl[i] == lambda_blue[0]:
		lambda_blue_idx[0] = i
	if hico_wl[i] == lambda_blue[1]:
		lambda_blue_idx[1] = i
	if hico_wl[i] == lambda_blue[2]:
		lambda_blue_idx[2] = i

intensities = np.zeros((500,500))

H,W,L = HICO_original.shape

I = copy.deepcopy(HICO_original)

for i in range(0,H):
	for j in range(0,W):
		intensity = a[0]
		numerator = max([ I[i,j,lambda_blue_idx[0]], I[i,j,lambda_blue_idx[1]], I[i,j,lambda_blue_idx[2]]])
		denominator = I[i,j,lambda_green_idx]
		for k in range(1,5):
			intensity += a[k]*(math.log(numerator/denominator,10))**k
		intensities[i,j] = intensity

print("low intensity: ", intensities[48,12])
print("high intensity: ", intensities[33,207])
print("medium intensity: ", intensities[133,11])
plt.figure()
plt.imshow(intensities)
plt.show()