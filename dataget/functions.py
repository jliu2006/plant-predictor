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
    
def download_file(folder, filename, datestr, MOD_ID, urlprefix, typeindex):
     
    username_file = open("/home/fun/profile/modis_username.txt", "r")
    password_file = open("/home/fun/profile/modis_password.txt", "r")
    username = username_file.readline()
    password = password_file.readline()
    
    url = generate_modis_url(datestr, MOD_ID, urlprefix)
    url = url + filename
    if len(filename) == 0:
        filename = typeindex + MOD_ID + '_' + datestr + '.html'
    
    print ('downloading file ', url)
    
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
    if type(datestr) != str:
        datestr = datestr.strftime('%Y-%m-%d')
    url = urlprefix + MOD_ID +'/'+ datestr[0:4] + '.' + datestr[5:7] + '.'+ datestr[8:10] + '/'
    
    return url

def download_one_fire(folder, MOD_ID, dt, urlprefix, tiles):
    globprof = folder + 'profile.json'
    
    with open(globprof) as f:
        prof = json.load(f)
        area = prof['info']['acres_burned']
        print ('area:' ,area)
        
        end = datetime.datetime.strptime(prof['end'], '%Y-%m-%d')
        cutoff_date = end.replace(year=2021, month=2, day=1)
    try: 
        if (int(area) >= 10000 and end <= cutoff_date): #JH: if burned area is too small or the end date is too recent
            download_html(folder, MOD_ID, dt, urlprefix)
            download_hdf(folder, MOD_ID, urlprefix, tiles, 'fires_index_')
    except:
        return

def download_html(folder, MOD_ID, dt, urlprefix):
    profile = folder + 'profile.json'
    f = open(profile)
    info = json.load(f)
    
    it_id = 0
    frame = 27  #JH: download fixed number of series
    
    if dt[-1] == 'd':
        start = datetime.datetime.strptime(info['start'], '%Y-%m-%d')
        end = datetime.datetime.strptime(info['end'], '%Y-%m-%d')

        begin_date = end - timedelta(days=90) # 
        final_date = end + timedelta(days=800) # 
        it_date = begin_date
        while (it_id < frame and it_date < final_date):
            try:
                datestr = it_date.strftime('%Y-%m-%d')
                print(datestr)
                url = generate_modis_url(datestr, MOD_ID, urlprefix)
                download_file(folder, "", datestr, MOD_ID, urlprefix, 'fires_index_')

                if os.path.isfile(folder + 'fires_index_' + MOD_ID + '_' + datestr + '.html'):
                    it_date = it_date + timedelta(days=32)
                    it_id = it_id +1
                else:
                    it_date = it_date + timedelta(days=1)
            except:
                continue
                
    elif dt[-1] == 'm':
        start = datetime.datetime.strptime(info['start'][0:7]+'-01', '%Y-%m-%d')
        end = datetime.datetime.strptime(info['end'][0:7]+'-01', '%Y-%m-%d')
        
        begin_date = end - timedelta(days=90) # 
        
        final_date = end + timedelta(days=800) # 

        it_date = begin_date
        while (it_id < frame and it_date < final_date):
            datestr = it_date.strftime('%Y-%m-%d')
            print(datestr)
            url = generate_modis_url(datestr, MOD_ID, urlprefix)
            print(url)
            download_file(folder, "", datestr, MOD_ID, urlprefix, 'fires_index_')
            it_id = it_id +1
            it_date = it_date + relativedelta(months=1)
        
def download_hdf(folder, MOD_ID, urlprefix, tiles, typeindex):
    """
    profiles = list of  dic {
         'lat_min': 40,
         'lat_max': 50,
         'name': ''
    }
    """
    
    globdirc = folder + typeindex + '*.html'
    globprof = folder + 'profile.json'
    with open(globprof) as f:
        prof = json.load(f)
        lat = prof['info']['latitude']
        
    tile = None
    for it in tiles:
         if it['lat_min'] < lat <= it['lat_max']: 
                tile = it
                break
        
    if tile is None:
        return
        
    for filename in glob(globdirc):
        datestr = filename[-15:-5]
        print ('parsing %s' % filename)
        filelist = parse_html(filename, MOD_ID.split(".")[0])
        for hdf in filelist:
            if tile['name'] in hdf:
                print('downloading ', hdf, 'to ', folder)
                download_file(folder, hdf, datestr, MOD_ID, urlprefix, typeindex)
                # break

def get_coords(lat, lon, MOD, tiles):   
    lon_index = 0
    lat_index = 0
    
    tile = None
    for it in tiles:
        if it['lat_min'] < lat <= it['lat_max']: 
            MOD_lat = MOD + '_' + it['name'] + '_lat'
            MOD_lon = MOD + '_' + it['name'] + '_lon'
            tile = it
            break
    if tile is None:
        print ("tile not found")
        return
            
            
    while coords[MOD_lat][lat_index][0] > lat:
        lat_index += 1
    while coords[MOD_lon][lat_index][lon_index] < lon:
        lon_index += 1
    # print(coords[MOD_lat][lat_index][lon_index], coords[MOD_lon][lat_index][lon_index])
    return lat_index, lon_index

def get_subimg(lat, lon, radius, hdf_link, MOD, tiles):
    lat_index, lon_index = get_coords(lat, lon, MOD, tiles)
    hdf = SD(hdf_link)
    ndvi = hdf.select(0).get()
    evi = hdf.select(1).get()
    # print(lat_index, lon_index)
    ndvi_img = ndvi[lat_index-radius:lat_index+radius, lon_index-radius:lon_index+radius]
    evi_img = evi[lat_index-radius:lat_index+radius, lon_index-radius:lon_index+radius]
    subimg = np.zeros(shape=(2, radius*2, radius*2)) # batch, time, chan, h, w
    subimg[0] = ndvi_img
    subimg[1] = evi_img
    return subimg

def load_lon_lat(txt):
    arr = np.loadtxt(txt, str)
    arr = np.char.replace(arr, ',', '')
    arr = arr.astype(float)
    length = int((arr.shape[0]) ** 0.5)
    arr = np.reshape(arr, (length, length))
    
    return arr

def load_coords():
    global coords
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

def write_imgs(folder, radius, fillvalue, tiles, typeindex):
    profile = folder + '/profile.json'
    with open(profile) as f:
        prof = json.load(f)
        lat = prof['info']['latitude']
        lon = prof['info']['longitude']
    
    tile = None
    for it in tiles:
        if it['lat_min'] < lat <= it['lat_max']: 
            tile = it
            break
         
    if tile is None:
        print ("tile not found")
        return
        
    mod_link = folder + '/' + 'MOD*' + tile['name'] + '*'
    mods = glob(mod_link)
    print("ALL FILES:", mods)
    for mod in mods:
        date = datetime.datetime(int(mod[-36:-32]), 1, 1) + datetime.timedelta(days = int(mod[-32:-29])-1)
        try:  # may not be sufficient to crop
            subimg = get_subimg(lat, lon, radius, mod, mod[-45:-38], tiles)
            avg = np.mean(subimg)
            subimg[subimg == fillvalue] = avg  #JH: replacing fill value with average value
            file = folder + '/' + typeindex + mod[-45:-38] + '_' + date.strftime('%Y-%m-%d') + '.npy'
            np.save(file, subimg)
        except:
            continue
        if os.path.isfile(mod):
            print("removing...", mod)
            os.remove(mod)
            
load_coords()


