import glob
import numpy as np
import sys
import os

def crop_npy(folder, radius):
    npys = folder + '/*.npy'
    for it in (glob.glob(npys)):
        arr = np.load(it)
        res = arr[100-radius:100+radius, 100-radius:100+radius]
        os.remove(it)
        np.save(it, res)
        
folders = sys.argv
radius = int(sys.argv[-1])

for folder in folders:
     crop_npy(folder, radius)