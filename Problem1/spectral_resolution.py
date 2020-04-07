from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

D = loadmat('data/HICO.mat')
hico_wl = D['hico_wl']

print(hico_wl)
for i in range(1, len(hico_wl)):
	print(hico_wl[i]-hico_wl[i-1])