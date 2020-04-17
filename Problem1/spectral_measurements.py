from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import copy
import pseudo_rgb as psdrgb

D = loadmat("../data/HICO.mat")
hico_wl = D['hico_wl']
HICO_original = D['HICO_original']

deep_water_x = 20
deep_water_y = 20
shallow_water_x = 100
shallow_water_y = 70
vegetation_x = 400
vegetation_y = 30

deep_water = HICO_original[deep_water_y, deep_water_x, :]
shallow_water = HICO_original[shallow_water_y, shallow_water_x, :]
vegetation = HICO_original[vegetation_y, vegetation_x, :]

rgbArray = psdrgb.pseudo_rgb(HICO_original) 
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