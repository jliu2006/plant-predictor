import os
import glob
# dont run while data downloading is in progress!!!
htmls = glob.glob('/home/fun/wildfire_data/*/*.html')
for it in htmls:
    os.remove(it)