import sys
from pyhdf.SD import SD, SDC
import numpy as np
import matplotlib.pyplot as plt
import glob
from glob import glob
import os
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import csv
import h5py
import numpy as np
import json
import datetime
from datetime import timedelta
import functions as f

tiles = [{
    'lat_min': 40,
    'lat_max': 50,
    'name': 'h08v04'
}, 
{
    'lat_min': 30,
    'lat_max': 40,
    'name': 'h08v05'
}]

            
wildfires = glob('/home/fun/wildfire_data/'+ sys.argv[1]) # no forward slash
print(sys.argv)
radius = 25

for wildfire in wildfires:
    print ('================= downloading  ' + wildfire + '==================')
#     f.download_one_fire(wildfire +'/', 'MOD13Q1.006', '32d', 'https://e4ftl01.cr.usgs.gov/MOLT/', tiles ) # vegi index
#     fillvalue = -3000 # JH: MOD13Q1 fill value is -3000
#     f.write_imgs(wildfire, radius, fillvalue, tiles,'')
    
    MOD11_files = glob(wildfire + '/MOD11*.npy')
    if(len(MOD11_files) < 25):
        f.download_one_fire(wildfire +'/', 'MOD11A2.006', '32d', 'https://e4ftl01.cr.usgs.gov/MOLT/', tiles )  # land temporature
#         fillvalue = 0 # JH: MOD13Q1 fill value is 0
#         f.write_imgs(wildfire, radius, fillvalue,tiles, '')
    else:
        print('MOD11A files exisit')
    
# #     download_one_fire(wildfire +'/', 'MCD64A1.006', '1m', 'https://e4ftl01.cr.usgs.gov/MOTA/' ) #burned area
    #download_one_fire(wildfire +'/', 'MOD14A2.006', '32d', 'https://e4ftl01.cr.usgs.gov/MOLT/' ) # firemask
    