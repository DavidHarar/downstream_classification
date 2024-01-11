import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm1d(out_chanels)
        
    def forward(self, x):
        orig_len = x.shape[2]
        unpadded_results = F.relu(self.bn(self.conv(x)))
        unpadded_results_len = unpadded_results.shape[2]
        left_pad,right_pad = int(np.floor((orig_len-unpadded_results_len)/2)) , int(np.ceil((orig_len-unpadded_results_len)/2))

        return F.pad(unpadded_results, (left_pad,right_pad))

class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1,
        red_3,
        out_3,
        red_5,
        out_5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        
        self.branch1 = ConvBlock(in_channels, out_1, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3, kernel_size=1, padding=0),
            ConvBlock(red_3, out_3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5, kernel_size=1),
            ConvBlock(red_5, out_5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)
    


class DownstreamInception(nn.Module):
    def __init__(self, dropout, scale, num_inputs=12, num_classes=1):
        super(DownstreamInception, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=num_inputs, 
            out_chanels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )

        self.conv2 = ConvBlock(in_channels=64, 
                               out_chanels=128*scale, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

                        # in_channels, out_1,red_3,out_3,red_5,out_5,
        self.inception3a = InceptionBlock(128*scale, 
                                          64*scale, 
                                          96, 
                                          128*scale, 
                                          16, 
                                          32*scale, 
                                          32*scale)
        self.inception3b = InceptionBlock(256*scale, 
                                          128*scale, 
                                          128, 
                                          192*scale, 
                                          32, 
                                          96*scale, 
                                          64*scale)
        self.inception4a = InceptionBlock(480*scale, 
                                          192*scale, 
                                          96, 
                                          208*scale, 
                                          16, 
                                          48*scale, 
                                          64*scale)
        self.inception4b = InceptionBlock(512*scale, 
                                          32, 
                                          112, 
                                          32, 
                                          24, 
                                          64, 
                                          32)
        
        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(8480, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        out = self.conv1(X)
        # print('conv1:', out.shape)
        out = self.maxpool(out)
        # print('maxpool:', out.shape)
        out = self.conv2(out)
        # print('conv2:', out.shape)
        out = self.maxpool(out)
        # print('maxpool:', out.shape)
        out = self.inception3a(out)
        # print('inception 3a:', out.shape)
        out = self.inception3b(out)
        # print('inception 3b:', out.shape)
        out = self.maxpool(out)
        # print('maxpool:', out.shape)
        out = self.inception4a(out)
        # print('inception 4a:', out.shape)
        out = self.inception4b(out)
        # print('inception 4b:', out.shape)
        out = self.avgpool(out)
        # print('avgpool:', out.shape)

        out = out.reshape(out.shape[0], -1)
        # print('reshape:', out.shape)
        out = self.dropout(out)
        # print('dropout:', out.shape)
        out = self.fc(out)
        # print('linear:', out.shape)
        out = self.sigmoid(out)

        return out



# ------------------------
# RESNET
# ------------------------

class DownstreamInceptionResnetBlock(nn.Module):
    def __init__(
        self,num_inputs,scale,
    ):
        super(DownstreamInceptionResnetBlock, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=num_inputs, 
            out_chanels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )

        self.conv2 = ConvBlock(in_channels=64, 
                        out_chanels=128*scale, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

                        # in_channels, out_1,red_3,out_3,red_5,out_5,
        self.inception3a = InceptionBlock(128*scale, 
                                          64*scale, 
                                          96, 
                                          128*scale, 
                                          16, 
                                          32*scale, 
                                          32*scale)
        self.inception3b = InceptionBlock(256*scale, 
                                          128*scale, 
                                          128, 
                                          192*scale, 
                                          32, 
                                          96*scale, 
                                          64*scale)
        self.inception4a = InceptionBlock(480*scale, 
                                          192*scale, 
                                          96, 
                                          208*scale, 
                                          16, 
                                          48*scale, 
                                          64*scale)
        self.inception4b = InceptionBlock(512*scale, 
                                          32, 
                                          112, 
                                          32, 
                                          24, 
                                          64, 
                                          32)

        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=1)
        self.fc_resnet = nn.Linear(8480, 5400)
        self.fc_out = nn.Linear(5400, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        # input = X
        out = self.conv1(X)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool(out)
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc_resnet(out)
        out = out.reshape(out.shape[0], 12, 450)
        out = self.relu(out+X) 

        return out


class DownstreamInceptionResnetFinal(nn.Module):
    def __init__(
        self,num_inputs,scale,dropout,

    ):
        super(DownstreamInceptionResnetFinal, self).__init__()
        
        self.conv1 = ConvBlock(
            in_channels=num_inputs, 
            out_chanels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )

        self.conv2 = ConvBlock(in_channels=64, 
                        out_chanels=128*scale, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

                        # in_channels, out_1,red_3,out_3,red_5,out_5,
        self.inception3a = InceptionBlock(128*scale, 
                                          64*scale, 
                                          96, 
                                          128*scale, 
                                          16, 
                                          32*scale, 
                                          32*scale)
        self.inception3b = InceptionBlock(256*scale, 
                                          128*scale, 
                                          128, 
                                          192*scale, 
                                          32, 
                                          96*scale, 
                                          64*scale)
        self.inception4a = InceptionBlock(480*scale, 
                                          192*scale, 
                                          96, 
                                          208*scale, 
                                          16, 
                                          48*scale, 
                                          64*scale)
        self.inception4b = InceptionBlock(512*scale, 
                                          32, 
                                          112, 
                                          32, 
                                          24, 
                                          64, 
                                          32)

        self.avgpool = nn.AvgPool1d(kernel_size=5, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(8480, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        # print('DownstreamInceptionResnetFinal')
        out = self.conv1(X)
        # print('conv1: ', out.shape)
        out = self.maxpool(out)
        out = self.conv2(out)
        # print('conv2: ', out.shape)
        out = self.maxpool(out)
        out = self.inception3a(out)
        # print('inc3a: ', out.shape)
        out = self.inception3b(out)
        # print('inc3b: ', out.shape)
        out = self.maxpool(out)
        out = self.inception4a(out)
        # print('inc4a: ', out.shape)
        out = self.inception4b(out)
        # print('inc4b: ', out.shape)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        # print('reshaped: ', out.shape)
        out = self.fc_out(out)
        out = self.sigmoid(out)
        return out



class DownstreamInceptionResnet(nn.Module):
    def __init__(self, dropout, scale, depth, num_inputs=12, num_classes=1):
        super(DownstreamInceptionResnet, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleDict() # a collection that will hold your layers

        self.layers['input'] = DownstreamInceptionResnetBlock(num_inputs,scale)
        
        if self.depth>2:
            for i in range(1, self.depth-1):
                self.layers['hidden_'+str(i)] = DownstreamInceptionResnetBlock(num_inputs,scale)
        
        self.layers['output'] = DownstreamInceptionResnetFinal(num_inputs,scale,dropout)
            
    def forward(self, x):
        for layer in self.layers:
            x = self.layers[layer](x)
        return x

