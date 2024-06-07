import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models_Cprompt.vision_transformer import VisionTransformer
import numpy as np
import math
from models_Cprompt.vit_coda_p import DualPrompt, L2P, CodaPrompt, CodaPrompt_weight, CodaPrompt_2d_v2

class cfed_network_hard(nn.Module):
    def __init__(self, numclass, feature_extractor, class_distribution, class_distribution_real=None, class_distribution_proportion=None, args=None):
        super(cfed_network_hard, self).__init__()
        self.args = args
        if "sharedencoder" in self.args.method:
            self.feature = nn.ModuleList()
            for i in range(int(self.args.numclass/self.args.class_per_task) + 1):
                self.feature.append(copy.deepcopy(feature_extractor))
        elif "sharedprompt" in self.args.method:
            self.feature = feature_extractor
            self.prompt = nn.ParameterList()
            for i in range(int(self.args.numclass/self.args.class_per_task) + 1):
                self.prompt.append(nn.Parameter(torch.FloatTensor(8, 768), requires_grad=True))
        elif "sharedcodap" in self.args.method:
            self.feature = feature_extractor
            self.global_prompt = nn.Parameter(torch.FloatTensor(8, 768), requires_grad=True)
            self.prompt = CodaPrompt(768, self.args.task_size, self.args.prompt_param, device=self.args.device, clients_local=self.args.local_clients, num_clients=self.args.num_clients, args=self.args)
        else:
            self.feature = feature_extractor
        self.numclass = numclass
        #self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.fc = nn.Linear(768, numclass, bias=True)
        self.client_fc = nn.Linear(768, self.args.class_per_task, bias=True)
        self.class_distribution = class_distribution
        self.class_distribution_real = class_distribution_real
        self.class_distribution_proportion = class_distribution_proportion
        self.task_id = 0
        self.client_index = -1
        self.client_class_min_output = []
        self.global_class_min_output = [] #This is for evaluation of global model
        self.current_class = []
        self.trained_task_id = None
        self.learned_class = None
        self.unlearned_class = None
        self.ep_g = 0

    
    def set_global_class_min_output(self, global_class_output, global_class_output_now):
        self.global_class_min_output = []
        for i in range(self.numclass):
            if i in global_class_output:
                continue
            else:
                self.global_class_min_output.append(i)

    def set_client_class_min_output(self, client_class_output):
        self.client_class_min_output = []
        self.client_class_max_output = client_class_output
        for i in range(self.numclass):
            if i in client_class_output:
               continue
            else:
                self.client_class_min_output.append(i) 
    
    def forward(self, input):
        #x = self.feature(input)
        if self.client_index == -1:
            x, _, _, _ = self.feature(input)
            feature = x[:,0,:]
            x = self.fc(feature)
            x = self.weit_weight(x, feature)
            
        else:
            if "sharedfc" in self.args.method:
                x, _, _, _ = self.feature(input)
                feature = x[:,0,:]
                #print(x.shape)
                x = self.fc(feature)
                x = self.weit_weight(x, feature)

                x[:,self.client_class_min_output] = -float('inf')
            elif "sharedencoder"  in self.args.method:
                x, _, _, _ = self.feature(input)
                feature = x[:,0,:]
                #print(x.shape)
                x = self.fc(feature)
                x[:,self.client_class_min_output] = -float('inf')
            elif "sharedprompt" in self.args.method:
                x = self.feature.forward_sharedprompt(input, self.prompt, -1)
                x = x[:,0,:]
                x = self.fc(x)
                x[:,self.client_class_min_output] = -float('inf')  
            elif "sharedcodap" in self.args.method:
                with torch.no_grad():
                    q, _, _, _ = self.feature(input)
                    q = q[:,0,:]
                x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, -1, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                x = x[:,0,:]
                x = self.fc(x)
                x[:,self.client_class_min_output] = -float('inf')
            else:
                x, _, _, _ = self.feature(input)
                feature = x[:,0,:]
                #print(x.shape)
                x = self.fc(feature)

                x[:,self.client_class_min_output] = -float('inf')
        return x
    
    def Incremental_learning(self, task_id):
        self.task_id = task_id

    def feature_extractor(self, inputs):
        if "sharedprompt" in self.args.method:
            feature = self.feature.forward_sharedprompt(inputs, self.prompt, -1)
            x = feature[:,0,:]
        elif "sharedcodap" in self.args.method:
            with torch.no_grad():
                q, _, _, _ = self.feature(inputs)
                q = q[:,0,:]
            x = self.feature.forward_sharedcodap(inputs, self.prompt, self.global_prompt, -1, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
            x = x[:,0,:]
        else:
            feature, _, _, _ = self.feature(inputs)
            x = feature[:,0,:]
        return x

    def set_learned_unlearned_class(self, learned_class):
        self.learned_class = learned_class
        self.unlearned_class = sorted(list(set(list(range(self.numclass))) - set(learned_class)))
    
    def weit_weight(self, output, features):
        
        output = output + self.client_fc(features).repeat(1, int(self.args.numclass/self.args.class_per_task))

        return output
    
    
    