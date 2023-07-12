import requests
import urllib
import pandas as pd
import glob
import time
import numpy as np
import json
import sys
# USGS Elevation Point Query Service
#url = r'https://nationalmap.gov/epqs/pqs.php?'
#new 2023:
url = r'https://epqs.nationalmap.gov/v1/json?'


def elevation_function(lon, lat):
                
        # define rest query params
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters'
        }
        
        #print (params)
        # format query string and return query value
        result = requests.get((url + urllib.parse.urlencode(params)))
        #elevations.append(result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
        #new 2023:
       
        try:
            return result.json()['value']
        except:
            return 0

    

coordnates_path = '/home/fun/wildfire_coords'

modtypes = ['MOD13Q1'] 
keys = ['h08v04']


def get_elevations(h08v04_info, suffix):
    elevations = {}
    for x in range(h08v04_info['l_lon'] * 1000, h08v04_info['r_lon'] * 1000, 10):
        for y in range(h08v04_info['l_lat'] * 1000, h08v04_info['u_lat']* 1000, 10):
             v = elevation_function(x * 0.001, y*0.001)
             elevations[str(x) + '_' + str(y)] = v
        
               
    json_file = "h08v04_info_elevations_" + suffix + ".json"
    try:
        with open(json_file, 'w') as jsonfile:
            json.dump(elevations, jsonfile)
            
    except IOError:
        print("I/O error")


def find_elevation():
    for mod in modtypes:
        for key in keys:
            coords_files = glob.glob(coordnates_path + '/*' + mod + '*' + key +'*')
            lats = []
            lons = []
            elevations = []
            for it in coords_files:
                if 'lat' in it:
                    with open(it, 'r') as f:
                        lats = f.readlines()
                if 'lon' in it:
                    with open(it, 'r') as f:
                        lons = f.readlines()

            for i in range(0, len(lats)):
                if (lats[i] == '\n'):
                    elevations.append('\n')
                else:
                    lon = float(lons[i].strip(', \n'))
                    lat = float(lats[i].strip(', \n'))

                    if lon < h08v04_info['l_lon'] or lon > h08v04_info['r_lon'] \
                        or lat < h08v04_info['l_lat'] or lat > h08v04_info['u_lat']:
                        elevations.append('0\n')
                    else:
                        v = elevation_function(lon, lat)
                        print (v)
                        elevations.append(str(v) + '\n')
                        if (i % 10000 ) == 0 and i > 0:
                            time.sleep(1)



            filename = coordnates_path + '/' + mod + "_" + key + '_elevation.txt'
            file = open(filename,'w')
            file.writelines(elevations)
            file.close()    


            
            
h08v04_info = dict([
    ('l_lon', int(sys.argv[1])),
    ('l_lat', int(sys.argv[2])),
    ('r_lon', int(sys.argv[3])),
    ('u_lat', int(sys.argv[4]))
])

            
            
get_elevations(h08v04_info, sys.argv[5])
