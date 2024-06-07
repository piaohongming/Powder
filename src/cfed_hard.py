from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
# import models
from models_Cprompt.vision_transformer import VisionTransformer
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
from models_FCeD.cfed_network import cfed_network
from models_FCeD.cfed_network_hard import cfed_network_hard
from PIL import Image
import dataloaders
from dataloaders.utils import *
from utils.schedulers import CosineSchedule

class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()

    def forward(self, output, old_target, temperature, frac):
        T = temperature
        alpha = frac
        outputs_S = F.log_softmax(output / T, dim=1)
        outputs_T = F.softmax(old_target / T, dim=1)
        l_old = outputs_T.mul(outputs_S)
        l_old = -1.0 * torch.sum(l_old) / outputs_S.shape[0]

        return l_old * alpha

def get_one_hot(target, num_class, device):
    if isinstance(device, int):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    else:
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class CFeD_hard:
    def __init__(self, batch_size, task_size, epochs, learning_rate, train_set, device, numclass, feature_extractor, dataset):
        super(CFeD_hard, self).__init__()
        self.memory_size = 500
        self.criterion = F.cross_entropy
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.task_id_old = -1
        self.task_size = task_size
        self.review = None
        self.numclass = numclass
        self.current_class = None
        self.last_class = None
        self.last_class_real = None
        self.last_class_proportion = None
        self.learned_classes = []
        self.train_loader = None
        self.batchsize = batch_size
        self.train_dataset = train_set
        self.model = None
        self.device = device
        self.dataset = dataset
        self.real_task_id = -1
        if dataset == 'MNIST':
            self.transform = transforms.Compose([#transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1,), (0.2752,))])
        else:
            self.transform = dataloaders.utils.get_transform(dataset=dataset, phase='train', aug=True, resize_imnet=True)
        
        self.exemplar_set = []
        self.client_learned_global_task_id = []

    


    def beforeTrain(self, task_id_new, group, client_index, global_task_id_real, class_real=None):
        try:
            if "sharedcodap" in self.model.args.method:
                self.model.module.prompt.client_index = client_index
            self.model.module.client_index = client_index
        except:
            if "sharedcodap" in self.model.args.method:
                self.model.prompt.client_index = client_index
            self.model.client_index = client_index
        
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            if group != 0:
                self.review = False
                if self.current_class != None:
                    self.last_class = self.current_class
                    self.last_class_proportion = self.current_class_proportion
                    self.last_class_real = self.current_class_real
                    self.learned_classes += self.last_class
                self.current_class = self.model.class_distribution[client_index][task_id_new]
                #TODO:same task
                classes_list = []
                for i in self.current_class:
                    classes_list.append(class_real[i])

                self.current_class = classes_list
                self.current_class_real = self.model.class_distribution_real[client_index][task_id_new]
                self.current_class_proportion = self.model.class_distribution_proportion[client_index][task_id_new]
                self.real_task_id = task_id_new
                # print(self.current_class)
                self.client_learned_global_task_id.append(global_task_id_real[self.model.args.num_clients * task_id_new + client_index])
                try:
                    if "sharedcodap" in self.model.args.method:
                        self.model.module.prompt.task_id = task_id_new
                    self.model.module.task_id = task_id_new
                except:
                    if "sharedcodap" in self.model.args.method:
                        self.model.prompt.task_id = task_id_new
                    self.model.task_id = task_id_new

                if self.last_class != None:
                    m = int(self.memory_size / len(self.learned_classes))
                    self._reduce_exemplar_sets(m)
                    for i in self.last_class: 
                        images = self.train_dataset.get_image_class(self.last_class_real[self.last_class.index(i)], self.model.client_index, self.last_class_proportion)
                        self._construct_exemplar_set(images, m)
            
            else:
                self.last_class = None    
                self.last_class_real = None
                self.last_class_proportion = None
                self.review = False
                try:
                    if "sharedcodap" in self.model.args.method:
                        self.model.module.prompt.task_id = self.real_task_id
                    self.model.module.task_id = self.real_task_id
                except:
                    if "sharedcodap" in self.model.args.method:
                        self.model.prompt.task_id = self.real_task_id
                    self.model.task_id = self.real_task_id          

        else:
            try:
                if "sharedcodap" in self.model.args.method:
                    self.model.module.prompt.task_id = self.real_task_id
                self.model.module.task_id = self.real_task_id
            except:
                if "sharedcodap" in self.model.args.method:
                    self.model.prompt.task_id = self.real_task_id
                self.model.task_id = self.real_task_id

        if "sharedcodap" in self.model.args.method:
            self.model.prompt.client_learned_global_task_id = self.client_learned_global_task_id
            self.model.prompt.global_task_id_real = global_task_id_real
        self.model.set_client_class_min_output(sorted(list(set(self.current_class + self.learned_classes))))
        self.model.set_learned_unlearned_class(sorted(list(set(self.current_class + self.learned_classes))))
        self.model.current_class = self.current_class

        
        
        if group == 1:
            self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, False)
        elif group == 2:
            self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, True)
        else:
            self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, False)
            #print(self.learned_classes + self.current_class)

    def compute_class_mean(self, images, transform):
        if isinstance(self.device, int):
            x = self.Image_transform(images, transform).cuda(self.device)
        else:
            x = self.Image_transform(images, transform).cuda()
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output
    
    def Image_transform(self, images, transform):
        #print(images[0])
        #data = transform(Image.fromarray(images[0])).unsqueeze(0)
        if self.dataset == 'MNIST':
            images = images.numpy()
            data = transform(Image.fromarray(images[0])).unsqueeze(0)
            data = data.expand(1, 3, data.size(2), data.size(3))
        elif self.dataset == 'DomainNet':
            data = transform(Image.fromarray(jpg_image_to_array(images[0]))).unsqueeze(0)
        elif self.dataset == 'ImageNet_R':
            data = transform(Image.fromarray(jpg_image_to_array(images[0]))).unsqueeze(0)
        else:
            data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            if self.dataset == 'MNIST':
                data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0).expand(1, 3, data.size(2), data.size(3))), dim=0)
            elif self.dataset == 'DomainNet':
                '''
                img = jpg_image_to_array(img_path)

                img = Image.fromarray(img)
                if self.transform is not None:
                    img = self.transform(img)
                '''
                data = torch.cat((data, self.transform(Image.fromarray(jpg_image_to_array(images[0]))).unsqueeze(0)), dim=0)
            elif self.dataset == 'ImageNet_R':
                data = torch.cat((data, self.transform(Image.fromarray(jpg_image_to_array(images[0]))).unsqueeze(0)), dim=0)
            else:
                data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data
    
    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
        

    def _construct_exemplar_set(self, images, m):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 768))
     
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_set.append(exemplar)


    def _get_train_and_test_dataloader(self, train_classes, train_classes_real, train_classes_proportion, review_sample):
        #self.train_dataset.getTrainData_sample(train_classes, 1)
        if review_sample:
            #self.train_dataset.getTrainData_sample(train_classes, 1)
            #self.train_dataset.getTrainData(train_classes, [], [])     
            self.review = True
            self.train_dataset.getTrainData([], self.exemplar_set, self.learned_classes, self.model.client_index, classes_real=[], classes_proportion=train_classes_proportion, exe_class=self.learned_classes)
        else:
            number_imbalance = self.train_dataset.getTrainImbalance(train_classes_real, [], [], self.model.client_index)
            self.number_imbalance = torch.tensor(number_imbalance, requires_grad=False, device=self.device)
            self.train_dataset.getTrainData(train_classes, [], [], self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=2,
                                  pin_memory=True)

        return train_loader
    
    def train(self, ep_g, model_old):
        loss = 0
        self.model.ep_g = ep_g
        self.old_model = model_old
        if self.review:
            #self.model = model_to_device(self.model, False, self.device)
            #self.model.eval()
            self.old_model.eval()
            dis_loss = DistillationLoss()
            old_targets = []
            with torch.no_grad():
                for batch_idx, (indexs, trains, labels) in enumerate(self.train_loader):
                    if isinstance(self.device, int):
                        trains, labels = trains.cuda(self.device), labels.cuda(self.device)
                    else:
                        trains, labels = trains.cuda(), labels.cuda()
                    output = self.old_model(trains.float())
                    output[:, sorted(list(set(list(range(self.model.numclass))) - set(self.learned_classes)))] = -float('inf')
                    old_targets.append(output)
            
            self.model.train()
            #optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
            #optimizer = optim.SGD(self.model.fc.parameters(), lr=self.learning_rate, weight_decay=0.00001)
            if "sharedfc" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(list(self.model.module.fc.parameters())+list(self.model.module.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            elif "sharedencoder" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters())+list(self.model.feature.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(list(self.model.module.fc.parameters())+list(self.model.module.client_fc.parameters())+list(self.model.feature.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            elif "sharedprompt" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()) + list([self.model.prompt[-1]]) + list([self.model.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                                    weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(self.model.module.fc.parameters()+list(self.model.module.client_fc.parameters()) + list([self.model.module.prompt[-1]]) + list([self.model.module.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            elif "sharedcodap" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()) + list([self.model.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                                    weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(self.model.module.fc.parameters()+list(self.model.module.client_fc.parameters()) + list([self.model.module.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            else:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(list(self.model.module.fc.parameters())+list(self.model.module.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            scheduler = CosineSchedule(optimizer, K=self.epochs)
            epoch_loss = []
            
            for epoch in range(self.epochs):
                batch_loss = []
                if epoch > 0:
                    scheduler.step()
                for batch_idx, (indexs, images, labels) in enumerate(self.train_loader):
                    if batch_idx % 2 == 0 and epoch % 2 == 0:
                        print('{}/{} {}/{} for {}'.format(batch_idx, len(self.train_loader), epoch, self.epochs, self.review), end="\r")
                    if isinstance(self.device, int):
                        images, labels = images.cuda(self.device), labels.cuda(self.device)
                    else:
                        images, labels = images.cuda(), labels.cuda()
                    # labels -= int(self.args.task_list[self.current_task][0])
                    #self.model.zero_grad()
                    output = self.model(images)
                    output[:, sorted(list(set(list(range(self.model.numclass))) - set(self.learned_classes)))] = -float('inf')
                    loss = dis_loss(output, old_targets[batch_idx], 2.0, 0.1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            return sum(epoch_loss) / len(epoch_loss), len(self.train_loader)

        else:
            self.model.train()
            #print(list(self.model.parameters()))
            #optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)
            #optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
            if "sharedfc" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(list(self.model.module.fc.parameters())+list(self.model.module.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            elif "sharedencoder" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters())+list(self.model.feature.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(list(self.model.module.fc.parameters())+list(self.model.module.client_fc.parameters())+list(self.model.feature.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            elif "sharedprompt" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()) + list([self.model.prompt[-1]]) + list([self.model.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                                    weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(self.model.module.fc.parameters()+list(self.model.module.client_fc.parameters()) + list([self.model.module.prompt[-1]]) + list([self.model.module.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            elif "sharedcodap" in self.model.args.method:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()) + list([self.model.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                                    weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(self.model.module.fc.parameters()+list(self.model.module.client_fc.parameters()) + list([self.model.module.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            else:
                if isinstance(self.device, int):
                    optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
                else:
                    optimizer = torch.optim.Adam(list(self.model.module.fc.parameters())+list(self.model.module.client_fc.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            scheduler = CosineSchedule(optimizer, K=self.epochs)
            epoch_loss = []
            #loss = torch.rand(1)
            for epoch in range(self.epochs):
                batch_loss = []
                if epoch > 0:
                    scheduler.step()
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    if step % 2 == 0 and epoch % 2 == 0:
                        print('{}/{} {}/{} loss:{} for {} target:{}'.format(step, len(self.train_loader), epoch, self.epochs, loss, self.review, target.shape), end="\r")
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    #target  = get_one_hot(target, self.numclass, self.device)
                    #self.model.zero_grad()
                    log_prob = self.model(images)
                    loss = self.criterion(log_prob, target)
                    #loss = F.binary_cross_entropy_with_logits(log_prob, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            return sum(epoch_loss) / len(epoch_loss), len(self.train_loader)
        


    
def jpg_image_to_array(image_path):
    """
    Loads JPEG image into 3D Numpy array of shape 
    (width, height, channels)
    """
    with Image.open(image_path) as image:      
        image = image.convert('RGB')
        im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
    return im_arr






    
