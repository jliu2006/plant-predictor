import sys
import os
import glob
from glob import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
import json

def get_ratio(folder): # 0 is NDVI, 1 is EVI
    raw_ndvi = folder + '/MOD13Q1*.npy'
    ref_ndvi = folder + '/r_MOD13Q1*.npy'
    ndvi_arrs = glob(raw_ndvi)
    ndvi_arrs.sort()
    ndvi_refs = glob(ref_ndvi)
    ndvi_refs.sort()
    
    for it in ndvi_arrs:
        #print("CURRENT:", it)
        month = it[-10:-6] # -xx-
        date = it[-14:-4]
        #print("MONTH:", month)
        it_ref = [k for k in ndvi_refs if month in k]
        ndvi_arr = np.load(it)
        ndvi_ref = np.load(it_ref[0])
        
        res = np.zeros(shape=(ndvi_arr.shape))
        
        for layer in range(len(ndvi_ref)):
            for i in range(len(ndvi_ref[0])):
                for j in range(len(ndvi_arr[0])):
                    if (ndvi_arr[layer][i][j] <= 0) or (ndvi_ref[layer][i][j] <= 0):
                        res[layer][i][j] = 0
                    else:
                        res[layer][i][j] = round(ndvi_arr[layer][i][j] / ndvi_ref[layer][i][j], 2)
        
        resfile = folder + '/ratio_MOD13Q1_' + date + '.npy'
        np.save(resfile, res)
        
        
folders = glob('/home/fun/wildfire_data/*')
good = 0
for folder in folders:
    prof = folder + '/profile.json'
    f = open(prof)
    info = json.load(f)
    end = datetime.datetime.strptime(info['end'], '%Y-%m-%d')
    cutoff_date = end.replace(year=2021, month=2, day=1)
    if type(info['info']['acres_burned']) != str:
        if(info['info']['acres_burned'] >= 3000) and (end <= cutoff_date):
            print(folder)
            try:
                get_ratio(folder)
                print("OK")
                good += 1
            except:
                print("BAD")
                continue
                
print("NUM:", good)