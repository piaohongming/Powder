from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
# import models
from models_DWL.vision_transformer_dwl import VisionTransformer
# from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
import scipy as sp
import scipy.linalg as linalg
# from utils.visualization import tsne_eval, confusion_matrix_vis, pca_eval
from torch.optim import Optimizer
import contextlib
import os
import copy
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable, Function
import random
from torch.utils.data import DataLoader
import torch.optim as optim
from Fed_utils import * 
from models_DWL.vit_coda_p_dwl import vit_pt_imnet_dwl
import re
torch.nn.BCEWithLogitsLoss


def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3  # scale up
        distance_negative = (anchor - negative).pow(2).sum(1) * 1.0e1  # scale up
        # print(distance_positive)
        losses = torch.relu((distance_positive - distance_negative).sum() + self.margin)
        return losses.mean()
class positive_loss(torch.nn.Module):
    def __init__(self, margin=0.0):
        super(positive_loss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive):
        distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3 # scale up
        losses = torch.relu((distance_positive).sum() + self.margin)
        return losses.mean()
    





class DWL:

    def __init__(self, num_class, prompt_flag, prompt_param, task_size, batch_size, device, epochs, learning_rate, train_dataset, model_g):
        super(DWL, self).__init__()
        self.prompt_param = prompt_param
        self.model = copy.deepcopy(model_g)
        self.task_id_old = -1
        self.task_size = task_size
        self.numclass = 0
        self.current_class = None
        self.last_class = None
        self.train_loader = None
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.old_model = None
        self.old_round_model = None
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        #self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        self.triplet_loss = TripletLoss(margin=1)
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none').cuda(self.device)
        
    def beforeTrain(self, task_id_new, group):
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            self.numclass = self.task_size * (task_id_new + 1)
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
                self.current_class = random.sample([x for x in range(self.numclass - self.task_size, self.numclass)], 6)
                # print(self.current_class)
            else:
                self.last_class = None

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, False)

    
    def _get_train_and_test_dataloader(self, train_classes, mix):
        if mix:
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.learned_classes)
        else:
            self.train_dataset.getTrainData(train_classes, [], [])

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  num_workers=8,
                                  pin_memory=True)

        return train_loader
    
    def train(self, ep_g, model_old, model_g):
        self.model = model_to_device(self.model, False, self.device)
        self.model.train()
        model_save= copy.deepcopy(self.model)
        model_save_weight = model_save.state_dict()
        model_save_weight_name = model_save_weight.keys()
        model_save_weight_len = len(model_save_weight)
        params_to_opt = list(self.model.prompt.parameters()) + list(self.model.fc.parameters())
        #params_to_opt = list(self.model.parameters())
        opt = optim.SGD(params_to_opt, lr=self.learning_rate, weight_decay=0.00001) 
        #opt = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_opt), lr=self.learning_rate,
                                         #weight_decay=0.00001)

        self.old_model = model_old #old task
        self.old_round_model = model_g #old round
        if self.old_model != None:
            print('load old model')
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()
        
        
            dis_loss_list = []
            grad_list = []
            dis_loss = self.compute_dis_loss()
            dis_loss_list.append(dis_loss)
            for epoch in range(self.epochs):
                #model_try1 = copy.deepcopy(self.model)

                
                '''
                if (epoch + ep_g * 20) % 200 == 100:
                    if self.numclass==self.task_size:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                    else:
                        for p in opt.param_groups:
                            p['lr'] =self.learning_rate / 5
                elif (epoch + ep_g * 20) % 200 == 150:
                    if self.numclass>self.task_size:
                        for p in opt.param_groups:
                            p['lr'] =self.learning_rate / 25
                    else:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
                elif (epoch + ep_g * 20) % 200 == 180:
                    if self.numclass==self.task_size:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001)
                    else:
                        for p in opt.param_groups:
                            p['lr'] =self.learning_rate / 125
                '''
                grad = None
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    
                    images, target = images.cuda(self.device), target.cuda(self.device)
                    loss_value, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss = self._compute_loss(indexs, images, target)
                    if step % 10 == 0 and epoch % 10 == 0:
                        print('{}/{} {}/{} {} {} {} {} {}'.format(step, len(self.train_loader), epoch, self.epochs, loss_value, normal_loss, t_c2loss, prompt_loss, output_disloss))
                    
                    #dis_loss = dis_loss + output_disloss.item()
                    opt.zero_grad()
                    if grad is None:
                        grad = torch.autograd.grad(loss_value, params_to_opt, retain_graph=True, allow_unused=True)
                    else:
                        grad = grad + torch.autograd.grad(loss_value, params_to_opt, retain_graph=True, allow_unused=True)
                    
                    loss_value.backward()
                    opt.step()

                dis_loss = self.compute_dis_loss()

                if len(dis_loss_list) < 3:
                    dis_loss_list.append(dis_loss)
                    grad_list.append(grad)
                else:
                    dis_loss_list = [dis_loss_list[1], dis_loss_list[2], dis_loss]
                    grad_list = [grad_list[1], grad]
                if len(dis_loss_list) == 3:
                    if dis_loss_list[2] > dis_loss_list[1] and dis_loss_list[1] > dis_loss_list[0]:
                        self.model.load_state_dict(model_save_weight)
                        with torch.no_grad():
                            
                            for i in range(model_save_weight_len):
                                if 'prompt' in model_save_weight_name[i] or 'fc' in model_save_weight_name[i]:
                                    
                                    match = re.search('(\.[0-9]+)\.', self.weight_name[i])
                                    if match is None:
                                        weight_name = 'self.model.' + self.weight_name[i]
                                    else:
                                        num = match.group(1)
                                        weight_name = 'self.model.' + self.weight_name[i].replace(num, f'[{num[1:]}]')
                                    
                                    #weight_name = 'self.model.' + [i]
                                    eval(f'{weight_name}.set_({weight_name}.data - self.learning_rate * grad[i])')
                #model_try_2 = copy.deepcopy(model_try1)
                        model_save= copy.deepcopy(self.model)
                        model_save_weight = model_save.state_dict()
                        model_save_weight_name = model_save_weight.keys()
                        model_save_weight_len = len(model_save_weight)
                        dis_loss = self.compute_dis_loss()
                        dis_loss_list = [dis_loss]
                        grad_list = []
        else:
            for epoch in range(self.epochs):
                loss_cur_sum, loss_mmd_sum = [], []
                '''
                if (epoch + ep_g * 20) % 200 == 100:
                    if self.numclass==self.task_size:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 5, weight_decay=0.00001)
                    else:
                        for p in opt.param_groups:
                            p['lr'] =self.learning_rate / 5
                elif (epoch + ep_g * 20) % 200 == 150:
                    if self.numclass>self.task_size:
                        for p in opt.param_groups:
                            p['lr'] =self.learning_rate / 25
                    else:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
                elif (epoch + ep_g * 20) % 200 == 180:
                    if self.numclass==self.task_size:
                        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001)
                    else:
                        for p in opt.param_groups:
                            p['lr'] =self.learning_rate / 125
                '''
                for step, (indexs, images, target) in enumerate(self.train_loader):
                        
                    images, target = images.cuda(self.device), target.cuda(self.device)
                    loss_value, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss = self._compute_loss(indexs, images, target)
                    if step % 10 == 0 and epoch % 10 == 0:
                        print('{}/{} {}/{} {} {} {} {} {}'.format(step, len(self.train_loader), epoch, self.epochs, loss_value, normal_loss, t_c2loss, prompt_loss, output_disloss))
                        
                    opt.zero_grad()
                    loss_value.backward()
                    opt.step()
            
            


        return len(self.train_loader)
    
    def compute_dis_loss(self):
        self.model.eval()
        avg_output_disloss = 0
        for step, (indexs, images, target) in enumerate(self.train_loader):
            images, target = images.cuda(self.device), target.cuda(self.device)
            with torch.no_grad():
                logits, prompt_loss, prelogits_current, prompt_client_current = self.model(images, train=True)
                logits_previous_task, _, prelogits_previous_task, prompt_client_previous = self.old_model(images, train=True)
    
            logits_previous_task = torch.sigmoid(logits_previous_task)
            logits_previous_task[..., (self.numclass - self.task_size) : ] = torch.sigmoid(logits[..., (self.numclass - self.task_size) :])
            #logits_previous_task[..., (self.numclass - self.task_size) : ] = target[..., (self.numclass - self.task_size) :]
            output_disloss = F.binary_cross_entropy_with_logits(logits, logits_previous_task)
            avg_output_disloss = avg_output_disloss + output_disloss.item()
        
        avg_output_disloss = avg_output_disloss / len(self.train_loader)
        return avg_output_disloss


    def _compute_loss(self, indexs, imgs, label):
        target = get_one_hot(label, self.numclass, self.device)
        #fedmoonLoss = torch.zeros((1,), requires_grad=True).cuda()
        t_c2loss = torch.zeros((1,), requires_grad=True).cuda(self.device)
        output_disloss = torch.zeros((1,), requires_grad=True).cuda(self.device)
        prelogits_disloss = torch.zeros((1,), requires_grad=True).cuda(self.device)
        logits, prompt_loss, prelogits_current, prompt_client_current = self.model(imgs, train=True)
        if self.old_model == None:
            with torch.no_grad():
                logits_global, _, prelogits_global, prompt_client_global = self.old_round_model(imgs, train=True)
            t_c2loss = positive_loss()(prompt_client_current[0].view(self.batch_size, -1), prompt_client_global[0].view(self.batch_size, -1))
            prelogits_disloss = positive_loss()(prelogits_current, prelogits_global)
            #t_c2loss = positive_loss()(prelogits_current, prelogits_global)
        else:
            with torch.no_grad():
                logits_previous_task, _, prelogits_previous_task, prompt_client_previous = self.old_model(imgs, train=True)
                _, _, prelogits_global, prompt_client_global = self.old_round_model(imgs, train=True)
            t_c2loss =self.triplet_loss(prompt_client_current[0].view(self.batch_size, -1), prompt_client_global[0].view(self.batch_size, -1), prompt_client_previous[0].view(self.batch_size, -1))
            prelogits_disloss = self.triplet_loss(prelogits_current, prelogits_global, prelogits_previous_task)
            logits_previous_task = torch.sigmoid(logits_previous_task)
            logits_previous_task[..., (self.numclass - self.task_size) : ] = torch.sigmoid(logits[..., (self.numclass - self.task_size) :])
            #logits_previous_task[..., (self.numclass - self.task_size) : ] = target[..., (self.numclass - self.task_size) :]
            output_disloss = F.binary_cross_entropy_with_logits(logits, logits_previous_task)
            #t_c2loss =self.triplet_loss(prelogits_current, prelogits_global, prelogits_previous_task)

        #logits[:,:(self.numclass - self.task_size)] = -float('inf')

        #dw_cls = self.dw_k[-1 * torch.ones(target.size()).long()]
        #print("hhhhhhh")
        #print(logits.shape)
        #print(target.shape)
        #total_loss = self.criterion(logits, target)
        normal_loss = torch.mean(F.binary_cross_entropy_with_logits(logits, target, reduction='none'))
        total_loss = normal_loss + 0 * t_c2loss + 0 * prompt_loss + 0 * prelogits_disloss + 1 * output_disloss
        

        return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss
    
    def criterion(self, logits, targets):
        loss_supervised = (self.criterion_fn(logits, targets)).mean()
        return loss_supervised

        
    




