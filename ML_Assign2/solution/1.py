import os
import os.path
import argparse
import h5py
import matplotlib.pyplot as plt


# Load the test data
with h5py.File("Data/data_5.h5", 'r') as hf:
	x = hf['x'][:]
	y = hf['y'][:]


plt.scatter(x[:, 0], x[:, 1], c = y)

plt.show()
