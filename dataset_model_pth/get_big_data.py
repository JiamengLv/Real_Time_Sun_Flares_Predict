import os
from glob import glob
import numpy as np
import skimage.transform as transform
import torch
from astropy.io import fits
from torch.autograd import Variable
from torch.utils import data
import ast
import pandas as pd
from PIL import Image
from torchvision import transforms as T
import os
from torch.utils import data # 获取迭代数据
from torch.autograd import Variable # 获取变量
from datetime import datetime
from datetime import datetime


def read_fits(path):
    
    hdu = fits.open(path)
    img = hdu[0].data
    img = np.array(img, dtype=np.float32)
    hdu.close()
    return img


def norm(img):

    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalization
    img -= np.mean(img)  # take the mean
    img /= np.std(img)  # standardization
    img = np.array(img, dtype='float32')
    return img

class DATASET_actions_fits():

    def __init__(self, datapath='', demo ="train",fineSize=128):

        super(DATASET_actions_fits, self).__init__()
        
        self.datapath = datapath

        self.list = sorted(os.listdir(datapath), key=lambda x: datetime.strptime(x, "%Y_%m_%d_%H_%M_TAI.fits"))

        self.len = int(len(self.list)/6)


    def __getitem__(self, iter):

        path = self.datapath
        data_list = self.list[iter*6:(iter+1)*6]
        # data_list = self.list[iter:iter+6]


        imgA = read_fits(os.path.join(path, data_list[0]))
        h, w = imgA.shape
        image = np.zeros(shape = (6,1,h, w))
        image1 = np.zeros(shape = (6,1,h, w))

        # 对数据路径进行排序
        for index in range(6):
            # print(data_list[index])
            imgA = read_fits(os.path.join(path, data_list[index]))
            image[index] = transform.resize(norm(imgA), (h, w))
            image1[index] = transform.resize(imgA, (h, w))

        return image, image1, self.list[iter],data_list

    
    def __len__(self):

        return self.len


if __name__ == "__main__":

    import numpy as np
    
    dataPath = "/home/lab30201/sdd/SUN_flares/DATA/DEMO/new_sun_flares"
    train_dataset = DATASET_fits(datacsv="/home/lab30201/sdd/SUN_flares/DATA/DEMO/train_test.csv",demo="train")
    train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    label_=[]
    
    for i, (x, y) in enumerate(train_loader):

        a = {}
        a["label"] = y

        label_.append(a)

