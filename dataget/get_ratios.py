import sys
import os
import glob
from glob import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt

def get_ratio(folder): # 0 is NDVI, 1 is EVI
    raw_ndvi = folder + '/MOD13Q1*.npy'
    ndvi_glob = glob(raw_ndvi)
    ndvi_glob.sort()
    ndvi_refs = ndvi_glob[:12]
    ndvi_arrs = ndvi_glob[12:]
    
    for it in ndvi_arrs:
        month = it[-10:-6] # -xx-
        date = it[-14:-4]
        it_ref = [k for k in ndvi_refs if month in k]
        ndvi_arr = np.load(it)
        ndvi_ref = np.load(it_ref[0])
        
        res = np.zeros(shape=(ndvi_arr.shape))
        
        for layer in range(len(ndvi_arr)):
            for i in range(len(ndvi_arr[0])):
                for j in range(len(ndvi_arr[0])):
                    if (ndvi_arr[layer][i][j] < 0) or (ndvi_ref[layer][i][j] < 0):
                        res[layer][i][j] = 0
                    else:
                        res[layer][i][j] = round(ndvi_arr[layer][i][j] / ndvi_ref[layer][i][j], 2)
        
        resfile = folder + '/ratio_MOD13Q1_' + date + '.npy'
        np.save(resfile, res)

folders = glob('/home/fun/wildfire_data/' + sys.argv[1])
print(sys.argv[1])
for folder in folders:
    globprof = folder + '/profile.json'
    with open(globprof) as f:
        prof = json.load(f)
        area = prof['info']['acres_burned']
    if int(area) > 10000:
        get_ratio(folder)