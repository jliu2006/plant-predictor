#rain precipitation from TRMM/GPM has been normalized into HDF5 by IMERG
#example
#wget https://arthurhouhttps.pps.eosdis.nasa.gov/gpmallversions/V06/2012/10/21/imerg/3B-HHR.MS.MRG.3IMERG.20121021-S170000-E172959.1020.V06B.HDF5
#

# from ipynb.fs.full.utils import *
from glob import glob
import os
import sys
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import csv
import h5py
import numpy as np
from datetime import timedelta
import json
import datetime
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
diameter = 5


def date_to_nth_day(date, fmt='%Y%m%d'):
    date = datetime.datetime.strptime(date, fmt)
    new_year_day = datetime.datetime(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1

def parse_html(html_file):
    '''
    parse html to get file list
    '''       
    with open(html_file, 'r') as input:
        soup = BeautifulSoup(input, "html.parser").find_all(lambda t: t.name == "a" and t.text.startswith('3B'))
        filelist = []
        for it in soup:
            filelist.append(it["href"])
        return filelist
    

                
                
def download_filelist(folder, url):
#     print ('downloading to ', folder)
    
    username_file = open("/home/fun/profile/imerg_username.txt", "r")
    password_file = open("/home/fun/profile/imerg_password.txt", "r")
    username = username_file.readline()
    password = password_file.readline()
    
    filename = folder + 'index.html'
    
    r = requests.get(url, auth = (username, password))

    if r.status_code == 200:
#         print ('writing to', filename)
        with open(filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)
                
        file_list = parse_html(filename)
        
        return file_list
        
def download_file(folder, url, filename):
     
    username_file = open("/home/fun/profile/imerg_username.txt", "r")
    password_file = open("/home/fun/profile/imerg_password.txt", "r")
    username = username_file.readline()
    password = password_file.readline()
    
#     print ('downloading file ', url + filename)
    
    r = requests.get(url + filename, auth = (username, password))
    if r.status_code == 200:
        print ('writing to', folder + filename)
        with open(folder + filename, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)
    else:
        print ('download error ', r.status_code)
    
     
def generate_imerg_url(datestr):
    '''
    compose url using date  'YYYY-MM-DD'
    '''
    #url = 'https://arthurhouhttps.pps.eosdis.nasa.gov/gpmallversions/V06/' + datestr[0:4] + '/' + datestr[4:6] + '/' + datestr[6:8] + '/imerg/'
    #url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.06/'+ datestr[0:4] + '/' + str(date_to_nth_day(datestr)).zfill(3) + '/'
    url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.06/' + datestr[0:4] + '/' + str(datestr)[5:7] + '/'
    print (url)
    return url

    
def get_target_file(filelist, timestr):
    '''
    return file that matches timestamp
    '''
    
    yyyy = str(timestr)[0:4]
    mm = str(timestr)[5:7]
    dd = str(timestr)[8:10]
    key = yyyy + mm + dd
    print (key)
    for it in filelist:
        if key in it:
            return it
    return None



def get_precip(nc_file, lon, lat): #read nc4 file , get precipitation for a give lon, lat
    dataset = nc.Dataset(nc_file)
    lon = float(lon)
    lat = float(lat)
    lons = dataset.variables['lon'][:].toflex().tolist()  #3600
    lats = dataset.variables['lat'][:].toflex().tolist() #1800
    precip = dataset.variables['precipitationCal_cnt_cond'][:, :,:]  #lon, lat, data
    
    lon_idx = -1
    for i in range(0, len(lons)):
        if abs(lons[i][0] - lon) <= 0.1:
            lon_idx = i
            break

    lat_idx = -1
    for i in range(0, len(lats)):
        if abs(lats[i][0] - lat) <= 0.1:
            lat_idx = i
            break
#     print('preciptation:', precip[0, lon_idx, lat_idx])
    return precip[0, lon_idx, lat_idx]


def get_monthly_precip_imerg(folder):
    json_profile = folder + '/profile.json'
    json_file = open(json_profile)
    info = json.load(json_file)
    lat = info['info']['latitude']
    lon = info['info']['longitude']
    
                                     
    start = datetime.datetime.strptime(info['start'], '%Y-%m-%d')
    end = datetime.datetime.strptime(info['end'], '%Y-%m-%d')
    begin_date = end - timedelta(days = 30)
    
    
    frame = 25  #JH: download fixed number of series
    day_per_frame = 4 #JH: pick 4 days for each frame, 8 days apart from each other 
    precip_daily = np.zeros((frame, day_per_frame))
    precip_monthly = np.zeros((frame, 1))
    it_date = begin_date
    for i in range(frame):
        for j in range(day_per_frame):
            url = generate_imerg_url(str(it_date))
            filelist = download_filelist(folder, url)
            if filelist is None:
                continue
            filename = get_target_file(filelist, str(it_date))  # without path
            if filename is None:
                continue
            download_file(folder, url, filename)
            precip_daily[i,j] = (get_precip(folder+filename, lon, lat) + 
            get_precip(folder+filename, lon+1, lat+1) + 
            get_precip(folder+filename, lon+1, lat-1) + 
            get_precip(folder+filename, lon-1, lat+1) + 
            get_precip(folder+filename, lon-1, lat-1) )

            precip_monthly[i] = precip_monthly[i] + precip_daily[i,j]
            
            it_date = it_date + timedelta(days= 8) #JH too time consuming to get daily, we now get precip every 8 days
            print('precip : ', i , j, precip_daily[i,j])
            if os.path.isfile(folder+filename):
                print("removing...", folder+filename)
                os.remove(folder+filename)
        
    file = folder + '/monthly_percip'+'.npy'
    np.save(file, precip_monthly)
        

wildfires = glob('/home/fun/wildfire_data/'+ sys.argv[1]) # no forward slash
print(sys.argv)


for wildfire in wildfires:
    globprof = wildfire + '/profile.json'
    with open(globprof) as f:
        prof = json.load(f)
        area = prof['info']['acres_burned']
        end = datetime.datetime.strptime(prof['end'], '%Y-%m-%d')
        cutoff_date = end.replace(year=2021, month=2, day=1)
    if (int(area) < 3000 or end > cutoff_date):
        print(wildfire +' burned area is less than 3000 or the end date is too recent')
        
    else:
        if os.path.isfile(wildfire +'/monthly_percip.npy'):
            print("monthly_percip.npy File exists")
        else:
            print ('================= downloading imerg precipitation into ' + wildfire + '==================')
            get_monthly_precip_imerg(wildfire + '/')  


  
    