import os
import glob
# dont run while data downloading is in progress!!!
htmls = glob.glob('/home/fun/wildfire_data/*/*.html')
hdfs = glob.glob('/home/fun/wildfire_data/*/*.hdf')

for it in htmls:
    os.remove(it)
for it in hdfs:
    os.remove(it)