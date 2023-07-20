import os
import json
import glob
import datetime
from glob import glob
        
folders = glob('/home/fun/wildfire_data/*')
for folder in folders:
    prof = folder + '/profile.json'
    f = open(prof)
    info = json.load(f)
    end = datetime.datetime.strptime(info['end'], '%Y-%m-%d')
    cutoff_date = end.replace(year=2021, month=2, day=1)
    if type(info['info']['acres_burned']) != str:
        if(info['info']['acres_burned'] >= 10000) and (end <= cutoff_date):
            print(folder)
            refs = folder + '/r_*'
            gref = glob(refs)
            for it in gref:
                date = it[-14:-4]
                match_ref = folder + '/MOD13Q1_' + date + '.npy'
                if os.path.isfile(match_ref):
                    print("removing..." + match_ref)
                    os.remove(match_ref)