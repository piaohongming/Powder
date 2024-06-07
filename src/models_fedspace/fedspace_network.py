import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models_Cprompt.vision_transformer import VisionTransformer
import numpy as np
import math


class fedspace_network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(fedspace_network, self).__init__()
        self.feature = feature_extractor
        #self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.fc = nn.Linear(768, numclass, bias=True)

    def forward(self, input):
        #x = self.feature(input)
        x, _, _ = self.feature(input)
        x = x[:,0,:]
        #print(x.shape)
        x = self.fc(x)
        return x
    
    def Incremental_learning(self, numclass, pretrain_model=None):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

        if pretrain_model is not None:
            self.fc.weight.data[out_feature:] = pretrain_model.model.fc.weight.data[out_feature: numclass]
            self.fc.bias.data[out_feature:] = pretrain_model.model.fc.bias.data[out_feature: numclass]


    def feature_extractor(self, inputs):
        feature, _, _ = self.feature(inputs)
        return feature[:,0,:]
    
    def predict(self, fea_input):
        return self.fc(fea_input)
    
    