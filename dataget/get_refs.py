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
from dateutil.relativedelta import relativedelta
import functions as f

def download_ref(folder, filename, datestr, MOD_ID, urlprefix):
    username_file = open("/home/fun/profile/modis_username.txt", "r")
    password_file = open("/home/fun/profile/modis_password.txt", "r")
    username = username_file.readline()
    password = password_file.readline()

    url = f.generate_modis_url(datestr, MOD_ID, urlprefix)
    url = url + filename
    if len(filename) == 0:
        filename = 'refer_index_' + MOD_ID + '_' + datestr + '.html'
        print("HTML:", filename)
        
    print ('downloading file ', url)
    
    r = requests.get(url, auth = (username, password))

    if r.status_code == 200:
        print ('writing to', folder + filename)
        if os.path.isfile(folder + filename):
            return 
        with open(folder + filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)    
    
def download_ref_html(folder, MOD_ID, dt, urlprefix):
    profile = folder + 'profile.json'
    file = open(profile)
    info = json.load(file)

    start = datetime.datetime.strptime(info['start'], '%Y-%m-%d')
#         begin_date = start - timedelta(days= int(dt[0:len(dt)-1]))
#         final_date = end + timedelta(days= int(dt[0:len(dt)-1]))

    begin_date = start.replace(year=start.year - 1, month=start.month, day=2)
    it_id = 0
    it_date = begin_date 
    while (it_id < 12): # JH: downdload 12 month of reference images , one for each month
        datestr = it_date.strftime('%Y-%m-%d')
    
        url = f.generate_modis_url(datestr, MOD_ID, urlprefix)
        download_ref(folder, "", datestr, MOD_ID, urlprefix)
        
        if os.path.isfile(folder + 'refer_index_' + MOD_ID + '_' + datestr + '.html'):
            it_date = it_date + relativedelta(months=1)
            it_date = it_date.replace(year=it_date.year, month=it_date.month, day=1)  #JH: always start with first day of month and try
            it_id = it_id + 1
        else:
            it_date = it_date + timedelta(days=1)

def download_ref_wrapped(folder, MOD_ID, dt, urlprefix, tiles):
    globprof = folder + 'profile.json'
    print("PROFILE:", globprof)
    with open(globprof) as file:
        prof = json.load(file)
        area = prof['info']['acres_burned']
    if int(area) < 10000:
        return
    
    download_ref_html(folder, MOD_ID, dt, urlprefix)
    f.download_hdf(folder, MOD_ID, urlprefix, tiles, 'refer_index_')
    
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


radius = 25
fillvalue = -3000
print(sys.argv[1])
wildfires = glob('/home/fun/wildfire_data/' + sys.argv[1])

for wildfire in wildfires:
    download_ref_wrapped(wildfire + '/', 'MOD13Q1.006', '32d', 'https://e4ftl01.cr.usgs.gov/MOLT/', tiles)
    f.write_imgs(wildfire, radius, fillvalue, tiles, 'r_')