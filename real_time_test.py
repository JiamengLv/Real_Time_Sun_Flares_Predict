import os
import argparse
import torch
import random
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
from dataset_model_pth.get_big_data import DATASET_actions_fits
import torch.nn as nn
from torch.autograd import Variable
from itertools import chain
from dataset_model_pth.model import EncoderCNN,DecoderRNN,FC_regression,BBFC_regression
import pandas as pd
from glob import glob

from torch import autograd
import matplotlib.pyplot as plt
import imageio
import cv2
from astropy.io import fits 
from down_loader_data import down_loader_data
import warnings
import shutil
from analy import calculate_class_acc,draw_CAM

warnings.filterwarnings("ignore")


def record_put_file(str):
    """ 记录文件 """
    file = open(r"./sun_flares_record.txt","a+") 
    file.write(str+"\n")

def put_file(str):
    """ 记录文件 """
    file = open(r"./sun_flares.txt","a+") 
    file.write(str+"\n")

def save_fit(img, name, path):
    if os.path.exists(path + name + '.fits'):
        os.remove(path + name + '.fits')
    grey = fits.PrimaryHDU(img)
    greyHDU = fits.HDUList([grey])
    greyHDU.writeto(path + name + '.fits')

def RemoveDir(filepath):
    ''' 
    如果文件夹不存在就创建，如果文件存在就清空！
    
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


######################## 超参数的设计 ##################################################

parser = argparse.ArgumentParser(description="real test class model")

parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument('--batchSize', type=int, default=1, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument("--inputch", type=int, default=1)
parser.add_argument("--outputch",type=int,default=1)
parser.add_argument('--Is_BBFC', default=False, type=bool)

parser.add_argument("--data_path", type=str, default="./realTime_data/")
parser.add_argument('--EM_pth', default='/home/dell460/ljm/Sun_flares/class_config/result/bbfc/gap_1hprior_6pre_24/checkpoints/EM_950.pth', help='path to pre-trained EM')
parser.add_argument('--RM_pth', default='/home/dell460/ljm/Sun_flares/class_config/result/bbfc/gap_1hprior_6pre_24/checkpoints/RM_950.pth', help='path to pre-trained RM')
parser.add_argument('--Light_RM_pth', default='/home/dell460/ljm/Sun_flares/class_config/result/bbfc/gap_1hprior_6pre_24/checkpoints/Light_RM_950.pth', help='path to pre-trained LightRM')

parser.add_argument("--samples", type=int, default=100,help="the number of samples")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loss_type', default='mse')

parser.add_argument('--save_path', default='./real_out/', help='folder to loss')

parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument("--gpus", default="1", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()

epochs = opt.epochs
batchSize = opt.batchSize

cuda = opt.cuda
samples = opt.samples

inputch = opt.inputch
outputch = opt.outputch

#################################### 设计随机种子 ####################################

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    print("=> use  gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

torch.backends.cudnn.benchmark = True


######################## 定义网络模型 ######################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

# input_channel
encoder_model = EncoderCNN(inputch)

# CNN_embed_dim = 256 , h_RNN = 256 (out)
rnn_model = DecoderRNN()

# h_RNN = 256 
if opt.Is_BBFC:
    classifier_model = BBFC_regression(256,1,128)
else:
    classifier_model = FC_regression(256,1,128)
    classifier_model.apply(weights_init)    

# 加载网络参数
if (opt.EM_pth != ''):
    print('Warning! Loading pre-trained weights.')
    encoder_model.load_state_dict(torch.load(opt.EM_pth))
    rnn_model.load_state_dict(torch.load(opt.RM_pth))
    classifier_model.load_state_dict(torch.load(opt.Light_RM_pth))
else:
    encoder_model.apply(weights_init)
    rnn_model.apply(weights_init)
    if not opt.Is_BBFC:
        classifier_model.apply(weights_init)

if (opt.loss_type == 'bce'):
    criterion = nn.BCELoss()
elif (opt.loss_type == 'cce'):
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.L1Loss()

if cuda:
    encoder_model.cuda()
    classifier_model.cuda()
    rnn_model.cuda()
    criterion.cuda()


def single_test(path,data):

    test_dataset = DATASET_actions_fits(path)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    loaderB = iter(test_loader)

    out = []
    count = 0

    while True:
        try:

            input_image, no_norm_image,_,data_list= loaderB.next()
            count+=1
            if cuda:
                input_data=input_image.cuda()
        except StopIteration:

            f=open(save_path+"/"+"out.txt","w")
            f.write(str(out))
            f.close()
            break

        input_data = input_data.to(torch.float32)
        save_path = opt.save_path + path.split("/")[-1] +"/"

        samples = 100

        with torch.no_grad():
            pred = [classifier_model(rnn_model(encoder_model(input_data))).detach().cpu().numpy().item() for per in range(samples)]

        out.append(np.array(pred).mean())

        class_out,class_acc = calculate_class_acc(pred)

        need_new_data = True

        if class_out == "N":

            need_new_data = False

        return need_new_data

def Multiple_test(path):

    #############################  加载数据 #################################################

    test_dataset = DATASET_actions_fits(path)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    loaderB = iter(test_loader)

    out = []
    count = 0

    while True:
        try:

            input_image, no_norm_image,_,data_list= loaderB.next()
            count+=1
            if cuda:
                input_data=input_image.cuda()
        except StopIteration:

            f=open(save_path+"/"+"out.txt","w")
            f.write(str(out))
            f.close()
            break

        input_data = input_data.to(torch.float32)
        save_path = opt.save_path + path.split("/")[-1] +"/"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        draw_CAM(count,encoder_model, rnn_model, classifier_model, input_data,save_path,data_list)

        samples = 100

        with torch.no_grad():
            pred = [classifier_model(rnn_model(encoder_model(input_data))).detach().cpu().numpy().item() for per in range(samples)]

        out.append(np.array(pred).mean())

        import matplotlib.pyplot as plt
         
        plt.hist(pred)
        plt.title("{}_mean{}_std{}".format(count,str(np.array(pred).mean())[0:5],str(np.array(pred).std())[0:4]))
        plt.savefig(save_path+"{}.jpg".format(count))
        plt.close()

        # 计算各类耀斑的发生概率
        class_out,class_acc = calculate_class_acc(pred)


if __name__=="__main__":

    path = opt.data_path

    for data in os.listdir(path):

        datapath = os.path.join(path,data)

        # 对实验结果进行分析 ---》 如果发生耀斑的话我们将研究24小时之前的数据热力图 、 更细致的热力图

        # 1.数据结果分析 ===》 保存耀斑发生，以及各类耀斑发生的概率
        if single_test(datapath, data):

            record_put_file("occurrence of solar flares--------------".format(data))
            put_file("occurrence of solar flares--------------".format(data))

            put_file("If you want to see the figure, please cilck here https://github.com/JiamengLv/Real_Time_Sun_Flares_Predict/tree/master/historyTime_out/{}".format(data))

            # RemoveDir(path)
            # 2.数据重新下载
            # print("downloader_new_data")
            use_label,start_time, end_time,harp_num =  down_loader_data(24,"1h",path)  

            # 3. 分析24小时 --》 耀斑发生以及各类耀斑的发生概率
            Multiple_test(datapath)
