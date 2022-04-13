#-*- encoding:utf8 -*-

import os
import time
import sys

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


#from basic_module import BasicModule
from models.BasicModule import BasicModule

sys.path.append("../")
from utils.config import DefaultConfig
configs = DefaultConfig()


class global_attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Parameter(t.rand(50))
        self.linear1 = nn.Linear(50, 50)
        
        self.LeakyRelu=nn.LeakyReLU(negative_slope=0.01)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, features, mid_feature, msa_features):
        features = features.squeeze(1)
        s = t.mean(features, dim=1).unsqueeze(1).repeat(1, 500, 1)
        y = t.mul(s, features) #128*500*49 ，msa_features： 128*500
        y = t.concat((y, msa_features.unsqueeze(-1)), dim=-1) # 128*500*50
        y = self.LeakyRelu(self.linear1(y))
        
        x = y * self.q1
        attention = self.softmax(x.sum(dim=-1)).unsqueeze(-1).repeat(1, 1, 49)
        output = t.mul(attention, features).sum(dim=1)
        output = t.concat((output, mid_feature), dim=-1)
        
        return output
    

class PPIModel(BasicModule):
    def __init__(self,class_nums,window_size,ratio=None):
        super(PPIModel,self).__init__()
        global configs
        configs.kernels = [13, 15, 17]
        self.dropout = configs.dropout = 0.2

        seq_dim = configs.seq_dim*configs.max_sequence_length
        
        self.seq_layers = nn.Sequential()
        self.seq_layers.add_module("seq_embedding_layer",
        nn.Linear(seq_dim,seq_dim))
        self.seq_layers.add_module("seq_embedding_ReLU",
        nn.ReLU())
        
        self.atten = global_attention()

        seq_dim = configs.seq_dim
        dssp_dim = configs.dssp_dim
        pssm_dim = configs.pssm_dim
        local_dim = (window_size*2+1)*(pssm_dim+dssp_dim+seq_dim)
        if ratio:
            configs.cnn_chanel = (local_dim*int(ratio[0]))//(int(ratio[1])*3)
        input_dim = 1923
                            
        self.DNN1 = nn.Sequential()
        print('input_dim:'+str(input_dim))
        self.DNN1.add_module("DNN_layer1",
                            nn.Linear(input_dim,1024))
        self.DNN1.add_module("ReLU1",
                            nn.ReLU())
        
        self.DNN2 = nn.Sequential()
        self.DNN2.add_module("DNN_layer2",
                            nn.Linear(1024,64))
        self.DNN2.add_module("ReLU2",
                            nn.ReLU())
                            
        self.DNN3 = nn.Sequential()
        self.DNN3.add_module("DNN_layer3",
                            nn.Linear(64,8))
        self.DNN3.add_module("ReLU2",
                            nn.ReLU())                    
        

        self.outLayer = nn.Sequential(
            nn.Linear(8, class_nums),
            nn.Sigmoid())
            
            
        in_channel = 1
        hidden_channels = 1
        self.conv1 = nn.Sequential()
        self.conv1.add_module("conv1",
                              nn.Conv1d(in_channel, hidden_channels,
                                        kernel_size=3)
                              )
        self.conv1.add_module("ReLU", nn.PReLU())
        self.conv1.add_module("pooling1", nn.MaxPool1d(kernel_size=3, stride=1))
        
        # in_channel = 1
        # hidden_channels = 1
        self.conv2 = nn.Sequential()
        self.conv2.add_module("conv2",
                              nn.Conv1d(in_channel, hidden_channels,
                                        kernel_size=5)
                              )
        self.conv2.add_module("ReLU", nn.PReLU())
        self.conv2.add_module("pooling2", nn.MaxPool1d(kernel_size=3, stride=1))
        
        # in_channel = 1
        # hidden_channels = 1
        self.conv3 = nn.Sequential()
        self.conv3.add_module("conv3",
                              nn.Conv1d(in_channel, hidden_channels,
                                        kernel_size=7)
                              )
        self.conv3.add_module("ReLU", nn.PReLU())
        self.conv3.add_module("pooling1", nn.MaxPool1d(kernel_size=3, stride=1))
        
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)
        self.dropout_3 = nn.Dropout(p=0.1)
        
        self.W_a = nn.Parameter(t.rand(98, 98))
        self.v_a = nn.Parameter(t.rand(98))
        
        
    
    def forward(self,seq,dssp,pssm,local_features, msa_features, middle_features):
        
        
        shapes = seq.data.shape
        features = seq.view(shapes[0],shapes[1]*shapes[2]*shapes[3])
        features = self.seq_layers(features)
        features = features.view(shapes[0],shapes[1],shapes[2],shapes[3])

        features = t.cat((features,dssp,pssm),3)
        # features = self.multi_CNN(features)
        # print('local_features:'+str(local_features.shape))
        # features = self.vgg19(features)
        features = self.atten(features, middle_features, msa_features)
        # local_features = self.local_atten(local_features, middle_features, msa_features)
        
        msa_features = msa_features.unsqueeze(1)
        msa_features1 = self.conv1(msa_features).squeeze(1)
        msa_features2 = self.conv2(msa_features).squeeze(1)
        msa_features3 = self.conv3(msa_features).squeeze(1)
        
        msa_features = t.cat((msa_features1, msa_features2, msa_features3), dim=-1)
        
        features = t.cat((features, local_features, msa_features), 1)
        # features = t.cat((local_features, msa_features), 1)
        # features = t.cat((local_features, features), 1)
        
        features = self.dropout_1(self.DNN1(features))

        features = self.dropout_2(self.DNN2(features))
        features = self.dropout_3(self.DNN3(features))
        features = self.outLayer(features)

        return features
