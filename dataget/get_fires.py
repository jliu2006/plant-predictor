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

def download_file(folder, filename, datestr, MOD_ID, urlprefix):
     
    username_file = open("/home/fun/profile/modis_username.txt", "r")
    password_file = open("/home/fun/profile/modis_password.txt", "r")
    username = username_file.readline()
    password = password_file.readline()
    
    url = generate_modis_url(datestr, MOD_ID, urlprefix)
    url = url + filename
    if len(filename) == 0:
        filename = 'fires_index_' + MOD_ID + '_' + datestr + '.html'
    
#     print ('downloading file ', url)
    
    r = requests.get(url, auth = (username, password))
    if r.status_code == 200:
        print ('writing to', folder + filename)
        if os.path.isfile(folder + filename):
            return 
        with open(folder + filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)
#    else:
#         print ('download error ', r.status_code)
        
def parse_html(html_file, MOD_ID):
    '''
    parse html to get file list
    '''

    with open(html_file, 'r') as input:
        soup = BeautifulSoup(input, "html.parser").find_all(lambda t: t.name == "a" and t.text.startswith(MOD_ID) and t.text.endswith('hdf'))
        filelist = []
        for it in soup:
            filelist.append(it["href"])
        return filelist

def generate_modis_url(datestr, MOD_ID , urlprefix):
    '''
    compose url using date  'YYYY.MM.DD'
    '''
    url = urlprefix + MOD_ID +'/'+ datestr[0:4] + '.' + datestr[5:7] + '.'+ datestr[8:10] + '/'
    
    return url

def download_one_fire(folder, MOD_ID, dt, urlprefix):
    download_html(folder, MOD_ID, dt, urlprefix)
    download_hdf(folder, MOD_ID, urlprefix)

def download_html(folder, MOD_ID, dt, urlprefix):
    profile = folder + 'profile.json'
    f = open(profile)
    info = json.load(f)
    
    if dt[-1] == 'd':
        start = datetime.datetime.strptime(info['start'], '%Y-%m-%d')
        end = datetime.datetime.strptime(info['end'], '%Y-%m-%d')
#         begin_date = start - timedelta(days= int(dt[0:len(dt)-1]))
#         final_date = end + timedelta(days= int(dt[0:len(dt)-1]))
        begin_date = end - timedelta(days=60) # 6 timesteps would include all of fire + roughly 1 month of regeneration
        final_date = end + timedelta(days=150) # 
        
        it_date = begin_date 
        while (it_date < final_date):
            datestr = it_date.strftime('%Y-%m-%d')
            print(datestr)
            url = generate_modis_url(datestr, MOD_ID, urlprefix)
            download_file(folder, "", datestr, MOD_ID, urlprefix)

            if os.path.isfile(folder + 'fires_index_' + MOD_ID + '_' +datestr + '.html'):
                it_date = it_date + timedelta(days= 16)
            else:
                it_date = it_date + timedelta(days=1)
                
    elif dt[-1] == 'm':
        start = datetime.datetime.strptime(info['start'][0:7]+'-01', '%Y-%m-%d')
        end = datetime.datetime.strptime(info['end'][0:7]+'-01', '%Y-%m-%d')
        
        #begin_date = start - relativedelta(months = 1) 
        begin_date = start  # TODO: if fire starts on 1st day of month, need to addjust month
        final_date = end + relativedelta(months=1)

        it_date = begin_date
        while (it_date <= final_date):
            datestr = it_date.strftime('%Y-%m-%d')
            print(datestr)
            url = generate_modis_url(datestr, MOD_ID, urlprefix)
            print(url)
            download_file(folder, "", datestr, MOD_ID, urlprefix)
            
            it_date = it_date + relativedelta(months=1)
        
def download_hdf(folder, MOD_ID, urlprefix):
    globdirc = folder + '*.html'
    globprof = folder + 'profile.json'
    with open(globprof) as f:
        prof = json.load(f)
        lat = prof['info']['latitude']
        area = prof['info']['acres_burned']
       
    if int(area) < 1000:
        return
    
    if (h08v05_info['lat_min'] < lat <= h08v05_info['lat_max']): 
        for filename in glob(globdirc):
            datestr = filename[-15:-5]
            print ('parsing %s' % filename)
            filelist = parse_html(filename, MOD_ID.split(".")[0])
            for hdf in filelist:
                if 'h08v05' in hdf:
                    print('downloading ', hdf, 'to ', folder)
                    download_file(folder, hdf, datestr, MOD_ID, urlprefix)
    else:
        for filename in glob(globdirc):
            datestr = filename[-15:-5]
            print ('parsing %s' % filename)
            filelist = parse_html(filename, MOD_ID.split(".")[0])
            for hdf in filelist:
                if 'h08v04' in hdf:
                    print('downloading ', hdf, 'to ', folder)
                    download_file(folder, hdf, datestr, MOD_ID, urlprefix)
                
h08v04_info = dict([
    ('lat_min', 40),
    ('lat_max', 50),
])

h08v05_info = dict([           
    ('lat_min', 30),
    ('lat_max', 40),
])

def get_coords(lat, lon, MOD):
    lon_index = 0
    lat_index = 0
    if (h08v05_info['lat_min'] < lat <= h08v05_info['lat_max']): 
        MOD_lat = MOD + '_h08v05_lat'
        MOD_lon = MOD + '_h08v05_lon'
    else:
        MOD_lat = MOD + '_h08v04_lat'
        MOD_lon = MOD + '_h08v04_lon'
    while coords[MOD_lat][lat_index][0] > lat:
        lat_index += 1
    while coords[MOD_lon][lat_index][lon_index] < lon:
        lon_index += 1
    # print(coords[MOD_lat][lat_index][lon_index], coords[MOD_lon][lat_index][lon_index])
    return lat_index, lon_index

def get_subimg(lat, lon, radius, hdf_link, MOD):
    lat_index, lon_index = get_coords(lat, lon, MOD)
    hdf = SD(hdf_link)
    ndvi = hdf.select(0).get()
    # print(lat_index, lon_index)
    subimg = ndvi[lat_index-radius:lat_index+radius, lon_index-radius:lon_index+radius]
    return subimg

def load_lon_lat(txt):
    arr = np.loadtxt(txt, str)
    arr = np.char.replace(arr, ',', '')
    arr = arr.astype(float)
    length = int((arr.shape[0]) ** 0.5)
    arr = np.reshape(arr, (length, length))
    
    return arr
   
def write_imgs(folder, radius):
    profile = folder + '/profile.json'
    with open(profile) as f:
        prof = json.load(f)
        lat = prof['info']['latitude']
        lon = prof['info']['longitude']
    if (h08v05_info['lat_min'] < lat <= h08v05_info['lat_max']): 
        tile = 'MOD*h08v05*'
    else:
        tile = 'MOD*h08v04*'
    mod_link = folder + '/' + tile
    mods = glob(mod_link)
    for mod in mods: 
        date = datetime.datetime(int(mod[-36:-32]), 1, 1) + datetime.timedelta(days = int(mod[-32:-29])-1)
        subimg = get_subimg(lat, lon, radius, mod, mod[-45:-38])
        file = folder + '/' + mod[-45:-38] + '_' + date.strftime('%Y-%m-%d') + '.npy'
        np.save(file, subimg)
        if os.path.isfile(mod):
            os.remove(mod)
            
coords = dict()
lat = glob('/home/fun/wildfire_coords/*lat.txt')
lon = glob('/home/fun/wildfire_coords/*lon.txt')
for txt in lat:
    name = txt[-22:-4]
    arr = load_lon_lat(txt)
    coords[name] = arr
for txt in lon:
    name = txt[-22:-4]
    arr = load_lon_lat(txt)
    coords[name] = arr
            
wildfires = glob('/home/fun/wildfire_data/'+ sys.argv[1]) # no forward slash
print(sys.argv)
radius = 100
for wildfire in wildfires:
    print ('================= downloading  ' + wildfire + '==================')
#         download_one_fire(wildfire +'/', MOD_ID, dt, urlprefix ) 
    download_one_fire(wildfire +'/', 'MOD13Q1.006', '16d', 'https://e4ftl01.cr.usgs.gov/MOLT/' ) # vegi index
    download_one_fire(wildfire +'/', 'MOD11A2.006', '16d', 'https://e4ftl01.cr.usgs.gov/MOLT/' )  # land temporature
#         download_one_fire(wildfire +'/', 'MCD64A1.006', '1m', 'https://e4ftl01.cr.usgs.gov/MOTA/' ) #burned area
    download_one_fire(wildfire +'/', 'MOD14A2.006', '16d', 'https://e4ftl01.cr.usgs.gov/MOLT/' ) # firemask
    write_imgs(wildfire, radius)