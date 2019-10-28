import urllib 
import lzma
from tqdm import tqdm #from tqdm import tqdm_notebook as tqdm
import pandas as pd 
import json
import requests
import traceback
import subprocess 
import os


#url constants
PUSHSHIFT_URL = 'https://files.pushshift.io/'
REDDIT = 'reddit'
SUBMISSIONS = 'submissions'
COMMENTS = 'comments'
RS = 'RS_20'
RS_V2 = 'RS_v2_20'
RC = 'RC_20'
XZ = '.xz'
ZST = '.zst'
BZ = '.bz2'
TMP = '/tmp/'
JSON = '.json'

START_YEAR = 05
END_YEAR = 19



def get_submissions():

    def _get_file(year, month, url, how=XZ, rs=RS):
        try:
            file_ = '{}{}-{}'.format(rs, str(year).zfill(2), str(month).zfill(2))
            compressed_file = "{}{}".format(file_, how)
            decompressed_file = "{}{}".format(file_, JSON)
            endpoint = '{}{}'.format(url, compressed_file)

            print("File: {}".format(file_))
            print("Compressed file: {}".format(compressed_file))
            print("Decompressed File: {}".format(decompressed_file))
            print("Endpoint: {}".format(endpoint))

            if how == XZ: decomp_command = "xz --decompress"
            if how == BZ: decomp_command = "bzip2 -d"
            if how == ZST: decomp_command = "zstd --decompress"

            # os.system("curl https://files.pushshift.io/reddit/submissions/RS_v2_2008-01.xz | xz --decompress > /tmp/ohyea.json")
            os.system("curl {} | {} > {}{}".format(endpoint, decomp_command, TMP, decompressed_file))

        except Exception as e: 
            traceback.print_exc()
            print("ERROR: {}".format(e)) 
    
    url = '{}{}/{}/'.format(PUSHSHIFT_URL, REDDIT, SUBMISSIONS) 
  
    for year in tqdm(range(START_YEAR, END_YEAR+1)): 
        for month in tqdm(range (1, 13)): 

            if (year == 5) and (month <= 5): continue #the first file is 2005-06
            
            if year <= 10: #pre-2011 everything is xz encoded
                _get_file(year, month, url, XZ, RS_V2)
            elif year <= 14: #2011-2014 inclusive is bs2 encoded
                _get_file(year, month, url, BZ)
            elif year <= 16: #2015-2016 inclusive is zst encoded
                _get_file(year, month, url, ZST)
            elif (year <= 17) and (month <= 11): # 2017-01 through 2017-11 inclusive are bz2 encoded 
                _get_file(year, month, url, BZ)
            elif ((year == 2017) and (month == 12)) or ((year <= 18) and (month <= 10)):
                # 2017-12 through 2018-10 inclusive are xz encoded
                _get_file(year, month, url, XZ)
            else: # from 2018-11 to current is zst encoded 
                _get_file(year, month, url, ZST) 
            print('\n\n\n')


if __name__ == "__main__":
    get_submissions() 
