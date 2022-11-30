import csv
import os
from datetime import datetime as dt_obj
from datetime import timedelta

import drms
import numpy as np
from astropy.io import fits
import shutil

import pandas as pd


# 将Time从字符串转换为可以作为文件名
def time_name(tstr):

    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])

    name = '{}_{}_{}_{}_{}_TAI'.format(year, month, day, hour, minute)
    return name

# 将Time从字符串转换为datetime对象
def parse_tai_string(tstr):

    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return dt_obj(year, month, day, hour, minute)

def read_fits(path):

    hdu = fits.open(path)
    img = hdu[0].data
    img = np.array(img, dtype=np.float32)
    hdu.close()
    return img



def save_fit(img, name, path):

    if os.path.exists(path + name + '.fits'):
        os.remove(path + name + '.fits')

    grey = fits.PrimaryHDU(img)
    greyHDU = fits.HDUList([grey])
    greyHDU.writeto(path + name + '.fits')

def data_real_time():

    c = drms.Client()
    keys, segments = c.query('hmi.sharp_cea_720s[$]',key='HARPNUM, T_REC', seg='Br')
    return list(keys["HARPNUM"])[-1],list(keys["T_REC"])[-1]

def data_laoder(harp_num, start_time, end_time, time_gap="1h", out_path="./data/"):

    c = drms.Client()
    keys, segments = c.query(
        'hmi.sharp_cea_720s[' + str(harp_num) + '][' + start_time + '-' + end_time + "@" + time_gap + ']',
        key='HARPNUM, T_REC', seg='Br')


    for index, time in enumerate(keys["T_REC"]):

        file_name = time_name(time)

        save_path = out_path + "/" + end_time.replace(":", "_").replace(".", "_") + "/"

        image_url = 'http://jsoc.stanford.edu' + segments.Br[index]
        photosphere_image = fits.open(image_url)  # download the data

        image = np.array(photosphere_image[1].data)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_fit(image, file_name, save_path)
    print("data_lengh:{}".format(len(keys)))
    return save_path


def check_data(path,time_length):

    use_label = True

    for root, dirs, filenames in os.walk(path,topdown=False):
        if len(filenames) < 6 and ("TAI" in root):
            shutil.rmtree(root)
            use_label = False
            print("数据不能收用长度")
        elif ("TAI" in root):
            for i in range(len(filenames)):
                image = read_fits(os.path.join(root,filenames[i]))
                if str(np.sum(image)) == "nan":
                    shutil.rmtree(root)
                    use_label = False
                    print("数据不能收用数值")

    return use_label
    

def down_loader_data(time_length,time_gap,out_path):

    harp_num,real_time = data_real_time()
    end_time = str(real_time)
    start_time = (parse_tai_string(end_time)-timedelta(hours=time_length+1)).strftime('%Y.%m.%d_%H:%M_TAI')

    print("--Download_real_time_data:")
    print("------------[{}][{}-{}]------------".format(harp_num, start_time, end_time))

    data_laoder(harp_num, start_time, end_time, time_gap=time_gap, out_path=out_path)
    use_label = check_data(out_path,time_length)

    return use_label,start_time, end_time,harp_num
