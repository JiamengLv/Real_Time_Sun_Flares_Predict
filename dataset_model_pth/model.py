import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils import data
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions import Normal

from torch import autograd
import matplotlib.pyplot as plt
import imageio


## ------------------------ Encoder part ---------------------- ##

def conv2D_output_size(img_size, padding, kernel_size, stride):

    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


class EncoderCNN(nn.Module):

    def __init__(self, input_channel, fc_hidden1=512, fc_hidden2=512, CNN_embed_dim=256):
        super(EncoderCNN, self).__init__()

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.gap_h,self.gap_w = 2, 2              # the output of GAP
        self.CNN_embed_dim = CNN_embed_dim        # the output of conv2d

        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4,
                      padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(self.ch4 * self.gap_h * self.gap_w,
                             self.fc_hidden1)  # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

    def forward(self, x_3d):

        cnn_embed_seq = []

        # conv_encoding
        for t in range(x_3d.size(1)):

            cnn_embed_patch = []

            conv_list = [self.conv1,self.conv2,self.conv3,self.conv4]
            # CNNs
            x = x_3d[:, t, :, :, :]
            for layer_conv in conv_list:
                cnn_embed_patch.append(x)
                x = layer_conv(x)
            cnn_embed_patch.append(x)
            #### 可视化每一个时刻提取的数据 ##########################################

            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(40,5))
            # plt.suptitle("time{}".format(t))
            # for i in range(5):
            #     plt.subplot(1, 5, i + 1)
            #     plt.imshow(np.sum(cnn_embed_patch[i][0].cpu().numpy(), axis=0))
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.title("layer{}".format(i))
            #
            # plt.show()

            # 通过gap，然后拉平 encoder 的输出 
            x = torch.nn.functional.adaptive_avg_pool2d(x, (self.gap_h , self.gap_w))
            x = x.view(x.size(0), -1)

            # # FC layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # cnn_embed_seq: shape=(time_step, batch, input_size) - (batch, time_step, input_size)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

## ------------------------ Lstm part ---------------------- ##

class DecoderRNN(nn.Module):

    def __init__(self, CNN_embed_dim=256, h_RNN_layers=1, h_RNN=256):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )


    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        x = RNN_out[:, -1, :]  # choose RNN_out at the last time step

        return x

## ------------------------ regression part ---------------------- ##

class FC_regression(nn.Module):

    def __init__(self, input_size, ouput_size, hidden_units, is_drop=True, drop=0.3):
        super().__init__()
        self.hidden1 = nn.Linear(input_size,int(hidden_units/2))
        self.hidden2 = nn.Linear(int(hidden_units/2),int(hidden_units/4))
        self.out = nn.Linear(int(hidden_units/4), ouput_size)
        self.is_drop = is_drop
        self.drop = drop

    def forward(self, x):

        x = torch.relu(self.hidden1(x))
        if self.is_drop:
        	x = F.dropout(x, p=self.drop)  

        x = self.hidden2(x)
        if self.is_drop:
        	x = F.dropout(x, p=self.drop)  

        x = self.out(x)

        return x

class Linear_BBB(nn.Module):
    """
        Layer of our BNN.
    """
    def __init__(self, input_features, output_features, prior_var=1.):
        """
            Initialization of our layer : our prior is a normal distribution
            centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer    定义w均值和方差为可学习的参数
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))

        #initialize mu and rho parameters for the layer's bias     定义b均值和方差为可学习的参数
        self.b_mu =  nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))

        #initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases  定义先验分布
        self.prior = torch.distributions.Normal(0,prior_var)

    def forward(self, input):
        """
          Optimization process
        """
        # sample weights

        w_epsilon = Normal(0,1).sample(self.w_mu.shape)

        try:
            self.w = self.w_mu + torch.log(1+torch.exp(self.w_rho)) * w_epsilon
        except:
            self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho))* w_epsilon.cuda()

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        try:
            self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon
        except:
            self.b = self.b_mu + torch.log(1+torch.exp(self.b_rho)) * b_epsilon.cuda()

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()

        return F.linear(input, self.w, self.b)

class BBFC_regression(nn.Module):

    def __init__(self, input_size, ouput_size, hidden_units, noise_tol=.1,  prior_var=1.):

        # initialize the network like you would with a standard multilayer perceptron, but using the BBB layer
        super().__init__()

        self.hidden = Linear_BBB(input_size, hidden_units,prior_var=prior_var)
        self.out = Linear_BBB(hidden_units, ouput_size,prior_var=prior_var)

        self.noise_tol = noise_tol # we will use the noise tolerance to calculate our likelihood

    def forward(self, x):

        x = torch.relu(self.hidden(x))
        x = self.out(x)

        return x

    def log_prior(self):

        # calculate the log prior over all the layers
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):

        # calculate the log posterior over all the layers
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target , samples):

        # we calculate the negative elbo, which will be our loss  function
        #initialize tensors
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)

        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            try:
                log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum() # calculate the log likelihood
            except:
                log_likes[i] = Normal(outputs[i].cuda(), self.noise_tol).log_prob(target.reshape(-1)).sum()
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()

        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss

################################################  Grad_cam ####################################################################################


def draw_CAM(label,count ,encoder_model, rnn_model, classifier_model, input_data,im,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    input_data = input_data.to(torch.float32)

    layers_names = ["conv1", "conv2", "conv3", "conv4", "fc1", "fc2", "fc3"]
    layers_dict = {layers_names[i]: None for i in range(len(layers_names))}

    for i, (name, module) in enumerate(encoder_model._modules.items()):
        layers_dict[layers_names[i]] = module

    cnn_embed_features = []
    cnn_embed_seq = []

    for t in range(input_data.size(1)):

        x = layers_dict["conv1"](input_data[:, t, :, :, :])
        x = layers_dict["conv2"](x)
        x = layers_dict["conv3"](x)
        x = layers_dict["conv4"](x)
        cnn_embed_features.append(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (2,2))
        x = x.view(x.size(0), -1)  # flatten the output ,of conv

        x = F.relu(layers_dict["fc1"](x))
        x = F.relu(layers_dict["fc2"](x))
        # x = F.dropout(x, p=0.5)
        x = layers_dict["fc3"](x)
        cnn_embed_seq.append(x)

    cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

    rnn_out = rnn_model(cnn_embed_seq)

    out = classifier_model(rnn_out)

    heamp_frames, input_frames = [], []

    all_heamp = np.zeros(shape=(10,10))
    
    for index, features in enumerate(cnn_embed_features):

        out1 = out.max(1, keepdim=True)[1]
        features_grad = autograd.grad(out[0][0], features, allow_unused=True,retain_graph=True)[0]
        grads = features_grad  
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
        pooled_grads = pooled_grads[0]
        features = features[0]
     
        for i in range(features.shape[0]):
            features[i, ...] *= pooled_grads[i, ...]
        
        heatmap1 = features.detach().cpu().numpy()
        heatmap1 = abs(np.mean(heatmap1, axis=0))

        heatmap1 = np.maximum(heatmap1, heatmap1.max()*0.01)
        heatmap1 /= np.max(heatmap1)
        
        h,w = input_data.shape[-2],input_data.shape[-1]
        heatmap = cv2.resize(heatmap1, (w,h))               
        heatmap = np.uint8(50 * heatmap)                
        map_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  

        dst,superimposed_img = cv2.threshold(heatmap, heatmap.max()*0.05, 255, cv2.THRESH_BINARY)

        b_img = superimposed_img*im.detach().cpu().numpy()[0][index][0] 

        if not os.path.exists(save_path +"/heat_map/{}/".format(count)):
            os.makedirs(save_path +"/heat_map/{}/".format(count))

        if not os.path.exists(save_path +"/im/{}/".format(count)):
            os.makedirs(save_path +"/im/{}/".format(count))

        save_fit(heatmap,"{}".format(index),save_path +"/heat_map/{}/".format(count))
        save_fit(im.detach().cpu().numpy()[0][index][0],"{}".format(index),save_path +"/im/{}/".format(count))


        # 研究关注区域与磁场的关系

        plt.imsave("./{}/{}_{}.jpg".format(save_path, out.detach().cpu().numpy()[0],index),superimposed_img*im.detach().cpu().numpy()[0][index][0],cmap="gray")

        plt.imsave("./{}/b{}_{}.jpg".format(save_path,out.detach().cpu().numpy()[0],index),im.detach().cpu().numpy()[0][index][0],cmap="gray")

        plt.imsave("./{}/heatmap{}_{}.jpg".format(save_path,out.detach().cpu().numpy()[0],index),heatmap,cmap="gray")