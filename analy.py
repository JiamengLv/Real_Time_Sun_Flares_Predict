import ast
from collections import defaultdict

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import autograd
import skimage.transform as transform

from datetime import datetime

def calculate_class_acc(sample_out):
    """
    对每个样本得到结果 进行进一步的处理 并放在一个字典中
    slare_num, start_time, end_time, label, out
    1. label_class: 得到样本本身标签所属的类别
    2. class_acc: 该样本属于各个类别的概率
    3. out_class: 预测样本属于哪个类别
    """

    sample = defaultdict()
    label = ["N", "C", "M", "X"]

    sample["slare_num"] = sample_out["slare_num"]
    sample["start_time"] = sample_out["start_time"]
    sample["end_time"] = sample_out["end_time"]
    sample["light_label"] = sample_out["light_label"]
    sample["light_out"] = ast.literal_eval(sample_out["light_out"])

    if sample["light_label"] == 0:
        sample["label_class"] = "N"
    else:
        i = [index for index in range(0, 3) if 1 <= (sample["light_label"] / (10 ** (index))) < 10][0]
        sample["label_class"] = label[i + 1]

    class_acc = {}
    for index in range(4):
        if index == 0:
            class_acc[label[index]] = len([i for i in sample["light_out"] if i < 10 ** (index)]) / len(
                sample["light_out"])
        else:
            class_acc[label[index]] = len(
                [i for i in sample["light_out"] if 1 < i / (10 ** (index - 1)) < 10]) / len(sample["light_out"])
    print(class_acc)

    # 设置不同的条件确定最终类别
    sample["out_class"] = max(class_acc, key=class_acc.get)
    sample["class_acc"] = class_acc

    return sample


def cal_second_class(index, all_samples, path):
    """
    计算二分类的结果
    """
    all_label = ["N", "C", "M", "X"]
    p_label = all_label[index:]
    n_label = all_label[:index]

    p_samples = all_samples[[per in p_label for per in all_samples["label_class"]]]
    n_samples = all_samples[[per in n_label for per in all_samples["label_class"]]]

    TP = len([per for per in p_samples["out_class"] if per in p_label])
    TN = len([per for per in n_samples["out_class"] if per in n_label])
    FP = len([per for per in n_samples["out_class"] if per in p_label])
    FN = len([per for per in p_samples["out_class"] if per in n_label])

    Acc = (TP + TN) / (TP + TN + FP + FN)
    TSS = TP / (TP + FN) - (FP / (FP + TN))

    Tp_rate = TP / (TP + FN)
    TN_rate = TN / (TN + FP)

    try:
        Precision = TP / (TP + FP)
    except:
        Precision = 0

    with open(path + "cal_second_class.txt", "a+") as f:
        f.write("---------label>{}------------\n".format(all_label[index]))
        f.write("TP:{},FN:{}\n".format(TP, FN))
        f.write("FP:{},TN:{}\n".format(FP, TN))
        f.write("Acc:{},TSS:{}\n".format(Acc, TSS))
        f.write("Tp_rate:{},Tn_rate:{},Precision:{}\n".format(Tp_rate, TN_rate, Precision))


def cal_every_class_Acc(index, class_label, all_sample):
    """
    计算每一类数据，对于其他类的分类结果 --- 混淆矩阵
    """
    label = class_label

    a = all_sample["label_class"]
    b = all_sample["out_class"]
    n_acc = len(b[[per == "N" for per in b]]) / len(a)
    c_acc = len(b[[per == "C" for per in b]]) / len(a)
    m_acc = len(b[[per == "M" for per in b]]) / len(a)
    x_acc = len(b[[per == "X" for per in b]]) / len(a)

    print("label:", label, "n_acc:%s,c_acc:%s,m_acc:%s,x_acc:%s" % (n_acc, c_acc, m_acc, x_acc))

    return [n_acc, c_acc, m_acc, x_acc]


def class_acc(csv_path, loss_path, label):
    out = pd.read_csv(csv_path)

    # 1. 对得到的结果，进行进一步的预处理
    pre_out = []
    for i in range(len(out)):
        pre_out.append(calculate_class_acc(out.loc[i]))
    pre_out = pd.DataFrame(pre_out)

    # 2.计算如果按二分类来说的结果
    for index, class_label in enumerate(["C", "M", "X"]):
        cal_second_class(index + 1, pre_out, loss_path)

    # 3.计算混淆矩阵，以及某一类别的输出结果
    out_acc = []
    for index, class_label in enumerate(["N", "C", "M", "X"]):
        choice_sample = pre_out[[per == class_label for per in pre_out["label_class"]]]
        out_acc.append(cal_every_class_Acc(index, class_label, choice_sample))

    # 4.绘制混淆矩阵
    plt.imshow(np.array(out_acc))
    class_labels = ["N", "C", "M", "X"]
    for i in range(4):
        plt.text(i, - 0.6, class_labels[i], fontdict={'fontsize': 14})
        for j in range(4):
            plt.text(i - 0.25, j, '%.2f' % np.array(out_acc)[j][i], fontdict={'fontsize': 14})
            plt.text(-1, j, class_labels[j], fontdict={'fontsize': 14})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(loss_path + "/test_mlti_class.jpg".replace("test", label))


def plot_features(csv_path, loss_path, label):
    out = pd.read_csv(csv_path)


def draw_CAM(label, encoder_model, rnn_model, classifier_model, input_data, loss_path,data_list):
    # 获取模型输出的feature/score
    input_data = input_data.to(torch.float32)

    layers_names = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2", "fc3"]
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}

    for i, (name, module) in enumerate(encoder_model._modules.items()):
        layers_dict[layers_names[i]] = module

    cnn_embed_features = []
    cnn_embed_seq = []

    for t in range(input_data.size(1)):

        # CNNs
        x = layers_dict["conv1"](input_data[:, t, :, :, :])
        x = layers_dict["conv2"](x)
        x = layers_dict["conv3"](x)
        x = layers_dict["conv4"](x)
        cnn_embed_features.append(x)

        # 拉平 encoder 的输出
        x = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)  # flatten the output ,of conv

        # FC layers
        x = F.relu(layers_dict["fc1"](x))
        x = F.relu(layers_dict["fc2"](x))
        x = F.dropout(x, p=0.3)
        x = layers_dict["fc3"](x)
        cnn_embed_seq.append(x)

    cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

    rnn_out = rnn_model(cnn_embed_seq)
    out = classifier_model(rnn_out)

    heamp_frames = []
    input_frames = []

    for index, features in enumerate(cnn_embed_features):

        features_grad = autograd.grad(out[0][0], features, allow_unused=True, retain_graph=True)

        # grads, weight 此处batch size默认为1，所以去掉了第0维（batch size维）
        grads = features_grad[0]  # 获取 grad - attention
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))[0].cpu().detach().numpy()   # 使用平均池化--》每个通道的weight

        # 选择几个网络最关注的区域
        # pooled_grads = np.where(pooled_grads>pooled_grads.max()*0.95,pooled_grads,0)
        # pooled_grads = pooled_grads/pooled_grads.sum()

        import heapq

        list_aa = range(features.shape[0])

        # list_aa = list(map(list(pooled_grads).index, heapq.nlargest(25, list(pooled_grads))))

        features = features[0].cpu().detach().numpy()

        for i in list_aa:
            features[i, ...] *= pooled_grads[i, ...]

        heatmap = features
        heatmap = np.mean(heatmap, axis=0)/np.sum(heatmap)
        heatmap = transform.resize(heatmap, (input_data.size(-2),input_data.size(-1)))
        heatmap = np.where(heatmap>heatmap.max()*0.05,heatmap,0)

        heamp_frames.append(heatmap)

    #### 对heamp_frames进行处理 #########################

    for i in range(6):

        if i == 0:
            heatmap[i] = heatmap[i]*0.4 + heatmap[i+1]*0.6
            continue

        if i == 5:
            heatmap[i] = heatmap[i]*0.4 + heatmap[i-1]*0.6
            continue

        heatmap[i] = heatmap[i]*0.2 + heatmap[i-1]*0.6 + heatmap[i+1]*0.2 


    for i in range(6):
       

        plt.figure(figsize=(15,4))

        plt.subplot(133)

        plt.imshow(heamp_frames[i]*input_data[0][i][0].cpu().detach().numpy(),cmap="gray")
        plt.xticks([])
        plt.yticks([])


        plt.subplot(132)
        plt.imshow(heamp_frames[i],cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.title("Time:{}".format(data_list[i][0].split(".")[0]))

        plt.subplot(131)
        plt.imshow(input_data[0][i][0].cpu().detach().numpy(),cmap="gray")
        plt.xticks([])
        plt.yticks([])

        plt.savefig(loss_path+"/input_heatmap_{}_{}.jpg".format(label,i))

        plt.close()

def calculate_class_acc(out_list):
    """
    计算一个样本的各类概率
    :param out_list:
    :return:
    """

    label = ["N","C","M","X"]
    class_acc = {}

    for index in range(4):
        if index == 0:
            class_acc[label[index]] = len([i for i  in out_list if i < 10**(index)])/len(out_list)
        else:
            class_acc[label[index]] = len([i for i  in out_list if 10**(index-1) < i < 10**(index)])/len(out_list)

    class_out = max(class_acc,key=class_acc.get)

    print("out_label:{},out_mean:{},out_acc{}".format(class_out,np.array(out_list).mean(),class_acc))
 
    record_put_file("out_label:{},out_mean:{},out_acc{}".format(class_out,np.array(out_list).mean(),class_acc))
    put_file("\n\nout_label:{},out_mean:{},out_acc{}".format(class_out,np.array(out_list).mean(),class_acc))

    return class_out,class_acc

def record_put_file(str):
    """ 记录文件 """
    file = open(r"./sun_flares_record.txt","a+") 
    file.write(str+"\n")

def put_file(str):
    """ 记录文件 """
    file = open(r"./sun_flares.txt","a+") 
    file.write(str+"\n")