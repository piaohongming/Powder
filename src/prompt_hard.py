from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
from models_Cprompt.vision_transformer import VisionTransformer
import numpy as np
import scipy as sp
import scipy.linalg as linalg
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
from models_Cprompt.vit_coda_p import vit_pt_imnet
from utils.schedulers import CosineSchedule
torch.nn.BCEWithLogitsLoss
import dataloaders
from dataloaders.utils import *

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1) * 1.0e3   # scale up
        distance_negative = (anchor - negative).pow(2).sum(1) * 1.0e1  # scale up
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

def get_one_hot(target, num_class, device):
    if isinstance(device, int):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    else:
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
    
    one_hot=one_hot.scatter(dim=1,index=target.long(),value=1.)
    return one_hot

class DualPrompt_hard:

    def __init__(self, num_class, prompt_flag, prompt_param, task_size, batch_size, device, epochs, learning_rate, train_dataset, model_g, imbalance):
        super(DualPrompt_hard, self).__init__()
        self.log = print
        self.prompt_flag = prompt_flag
        self.prompt_param = prompt_param
        self.model = copy.deepcopy(model_g)
        self.task_id_old = -1
        self.task_size = task_size
        self.numclass = num_class
        self.current_class = None
        self.last_class = None
        self.train_loader = None
        self.train_dataset = train_dataset
        self.origin_train_dataset = None
        self.target_train_dataset = []
        self.origin_train_dataset_divide = None
        self.target_train_dataset_divide = []
        self.batch_size = batch_size
        self.old_model = None
        self.old_round_model = None
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.memory_size = 500
        self.dataset = "ImageNet_R"
        #self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        self.triplet_loss = TripletLoss(margin=1)
        #self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.KLDiv = nn.KLDivLoss(reduction="none")
        if isinstance(self.device, int):
            self.criterion_fn = nn.CrossEntropyLoss(reduction='none').cuda(self.device)
        else:
            self.criterion_fn = nn.CrossEntropyLoss(reduction='none').cuda()
        self.real_task_id = -1
        self.imbalance = imbalance
        self.client_learned_task_id = []
        self.client_learned_global_task_id = []
        self.learned_classes = []
        self.prototype = {}
        self.prototype_label = {}
        self.radius = {}
        self.exemplar_set = []
        self.exemplar_dict = {}
        self.class_mean_set = []
        self.client_class_frequency = {}
        self.client_class_num_dict_last = None
        self.client_class_num_dict = None
        self.transform = dataloaders.utils.get_transform(dataset=self.dataset, phase='train', aug=True, resize_imnet=True)
        
    def beforeTrain(self, task_id_new, group, classes=None, client_index=-1, global_task_id_real=None, class_real=None):
        try:
            self.model.module.prompt.client_index = client_index
            self.model.module.client_index = client_index
        except:
            self.model.prompt.client_index = client_index
            self.model.client_index = client_index
        
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            if self.prompt_flag == 'cprompt':
                try:
                    self.model.module.prompt.max_task_id = task_id_new
                except:
                    self.model.prompt.max_task_id = task_id_new
            else:
                pass
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
                    self.last_class_real = self.current_class_real
                    self.last_class_proportion = self.current_class_proportion
                    self.learned_classes += self.last_class
                    self.learned_classes = sorted(list(set(self.learned_classes)))
                
                self.current_class = self.model.class_distribution[client_index][task_id_new]
                classes_list = []
                for i in self.current_class:
                   classes_list.append(class_real[i])
                   if class_real[i] not in self.client_class_frequency.keys():
                       self.client_class_frequency[class_real[i]] = 1
                   else:
                       self.client_class_frequency[class_real[i]] = self.client_class_frequency[class_real[i]] + 1
                
                self.current_class = classes_list
                self.current_class_real = self.model.class_distribution_real[client_index][task_id_new]
                self.current_class_proportion = self.model.class_distribution_proportion[client_index][task_id_new]
                if self.model.class_distribution_client_di is not None:
                    self.class_distribution_client_di = self.model.class_distribution_client_di[client_index][task_id_new]
                else:
                    self.class_distribution_client_di = None
            
                self.real_task_id = task_id_new
                self.client_learned_task_id.append(task_id_new)
                
                self.client_learned_global_task_id.append(global_task_id_real[self.model.prompt.num_clients * task_id_new + client_index])
                self.client_learned_global_task_id = sorted(list(set(self.client_learned_global_task_id)))
                try:
                    self.model.module.prompt.task_id = task_id_new
                    self.model.module.task_id = task_id_new
                except:
                    self.model.prompt.task_id = task_id_new
                    self.model.task_id = task_id_new
            else:
                self.last_class = None
                try:
                    self.model.module.prompt.task_id = self.real_task_id
                    self.model.module.task_id = self.real_task_id
                except:
                    self.model.prompt.task_id = self.real_task_id
                    self.model.task_id = self.real_task_id
        else:
            try:
                self.model.module.prompt.task_id = self.real_task_id
                self.model.module.task_id = self.real_task_id
            except:
                self.model.prompt.task_id = self.real_task_id
                self.model.task_id = self.real_task_id

        for t in range(len(self.client_learned_global_task_id)):
            self.client_learned_global_task_id[t] = global_task_id_real[self.client_learned_global_task_id[t]]
        self.client_learned_global_task_id = sorted(list(set(self.client_learned_global_task_id)))
        self.model.prompt.client_learned_global_task_id = self.client_learned_global_task_id
        self.model.learned_classes = sorted(list(set(self.learned_classes + self.current_class)))
        self.model.current_class = self.current_class
        self.model.prompt.global_task_id_real = global_task_id_real
        self.model.set_client_class_min_output()
        self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion,False)

    
    def _get_train_and_test_dataloader(self, train_classes, train_classes_real, train_classes_proportion, mix):
        if mix:
            #number_imbalance = self.train_dataset.getTrainImbalance(train_classes_real, self.exemplar_set, self.learned_classes, self.model.client_index)
            #self.number_imbalance = torch.tensor(number_imbalance, requires_grad=False, device=self.device)
            self.exemplar_set = []
            for i in self.learned_classes:
                self.exemplar_set.append(self.exemplar_dict[i])
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.learned_classes, self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion, class_distribution_client_di=self.class_distribution_client_di)
        else:
            #number_imbalance = self.train_dataset.getTrainImbalance(train_classes_real, [], [], self.model.client_index)
            #self.number_imbalance = torch.tensor(number_imbalance, requires_grad=False, device=self.device)
            self.train_dataset.getTrainData(train_classes, [], [], self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion, class_distribution_client_di=self.class_distribution_client_di)

        #print(self.train_dataset.TrainData[0])
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  num_workers=8,
                                  pin_memory=True)

        return train_loader

    def proto_save(self):
        self.model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for batch_idx, (indexs, images, target) in enumerate(self.train_loader):
                if isinstance(self.device, int):
                    feature = self.model.feature_extractor_withprompt(images.to(self.device))
                    #feature = self.model.feature_extractor(images.to(self.device))
                else:
                    feature = self.model.feature_extractor_withprompt(images.cuda())
                    #feature = self.model.feature_extractor(images.cuda())
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
        labels = np.concatenate([label_vector for label_vector in labels])
        features = np.concatenate([feature_vector for feature_vector in features], axis=0)
        feature_dim = features.shape[1]

        prototype = {}
        radius = {}
        class_label = []
        for item in self.current_class:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]

            if np.size(feature_classwise) == 0:
                prototype[item] = np.zeros(768,)
            else:
                prototype[item] = np.mean(feature_classwise, axis=0)
            cov = np.cov(feature_classwise.T)
            if not math.isnan(np.trace(cov)):
                radius[item] = np.trace(cov) / feature_dim
            else:
                radius[item] = 0
        radius = np.sqrt(np.mean(list(radius.values())))  
        self.model.train()
        for c in prototype.keys():
            if c in self.prototype.keys():
                self.prototype[c] = prototype[c]
                #self.prototype_label[c] = self.current_class
                #self.prototype_label[c] = self.model.prompt.task_id * self.model.prompt.num_clients + self.model.prompt.client_index
                self.prototype_label[c] = int(self.model.class_distribution[self.model.prompt.client_index][self.model.prompt.task_id][0] // 4)
                self.radius[c] = radius
            else:
                self.prototype[c] = prototype[c]
                #self.prototype_label[c] = self.current_class
                #self.prototype_label[c] = self.model.prompt.task_id * self.model.prompt.num_clients + self.model.prompt.client_index
                self.prototype_label[c] = int(self.model.class_distribution[self.model.prompt.client_index][self.model.prompt.task_id][0] // 4)
                self.radius[c] = radius
        return
    
    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
    
    def _reduce_exemplar_dict(self, client_class_num_dict_last, client_class_num_dict):
        for k in client_class_num_dict_last.keys():
            if client_class_num_dict[k] >= client_class_num_dict_last[k]:
                self.exemplar_dict[k] = self.exemplar_dict[k]
            else:
                self.exemplar_dict[k] = self.exemplar_dict[k][:client_class_num_dict[k]]

    
    def jpg_image_to_array(self, image_path):
        """
        Loads JPEG image into 3D Numpy array of shape 
        (width, height, channels)
        """
        with Image.open(image_path) as image:      
            image = image.convert('RGB')
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
        return im_arr

    def Image_transform(self, images, transform):
        #print(images[0])
        #data = transform(Image.fromarray(images[0])).unsqueeze(0)
        if self.dataset == 'MNIST':
            images = images.numpy()
            data = transform(Image.fromarray(images[0])).unsqueeze(0)
            data = data.expand(1, 3, data.size(2), data.size(3))
        elif self.dataset == 'DomainNet':
            data = transform(Image.fromarray(self.jpg_image_to_array(images[0]))).unsqueeze(0)
        elif self.dataset == 'ImageNet_R':
            data = transform(Image.fromarray(self.jpg_image_to_array(images[0]))).unsqueeze(0)
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
                data = torch.cat((data, self.transform(Image.fromarray(self.jpg_image_to_array(images[0]))).unsqueeze(0)), dim=0)
            elif self.dataset == 'ImageNet_R':
                data = torch.cat((data, self.transform(Image.fromarray(self.jpg_image_to_array(images[0]))).unsqueeze(0)), dim=0)
            else:
                data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        if isinstance(self.device, int):
            x = self.Image_transform(images, transform).cuda(self.device)
        else:
            x = self.Image_transform(images, transform).cuda()
        feature_extractor_output = F.normalize(self.model.feature_extractor_withprompt(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

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

    def _construct_exemplar_dict(self, images, i, client_class_num):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 768))

        for j in range(client_class_num):
            x = class_mean - (now_class_mean + feature_extractor_output) / (j + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])

        self.exemplar_dict[i] = exemplar
 
    
    def update_new_set(self, task_id, client_index, ep_g):
        self.model = model_to_device(self.model, False, self.device)
        self.model.eval()
        if self.last_class != None and ep_g % self.model.args.tasks_global == 0 and self.real_task_id == self.task_id_old:
            self.learned_numclass = len(self.learned_classes)
            class_num_list = []
            class_num_dict = {}
            for c in self.learned_classes:
                class_num_list.append(self.client_class_frequency[c])
            class_frequency_sum = sum(class_num_list)
            for c in range(len(class_num_list)):
                class_num_dict[self.learned_classes[c]] = int(self.memory_size * (class_frequency_sum - class_num_list[c]) / (class_frequency_sum * (len(class_num_list) - 1)))
            
            if self.client_class_num_dict is None:
                self.client_class_num_dict = class_num_dict
            else:
                self.client_class_num_dict_last = self.client_class_num_dict
                self.client_class_num_dict = class_num_dict
                self._reduce_exemplar_dict(self.client_class_num_dict_last, self.client_class_num_dict)
            print(self.last_class)
            print(self.learned_classes)
            print(self.last_class_real)
            for i in range(len(self.last_class_real)): 
                #images = self.train_dataset.get_image_class(i, self.model.client_index)
                if self.last_class[i] in self.exemplar_dict.keys():
                    images = self.train_dataset.get_image_class_dict(self.last_class_real[i], self.model.client_index, self.last_class_proportion, self.exemplar_dict[self.last_class[i]])
                else:
                    images = self.train_dataset.get_image_class_dict(self.last_class_real[i], self.model.client_index, self.last_class_proportion)
                self._construct_exemplar_dict(images, self.last_class[i], self.client_class_num_dict[self.last_class[i]])

        self.model.train()
        #self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, True)
    
    def train(self, ep_g, model_old, model_g, group=1):
        
        #self.model = model_to_device(self.model, False, self.device)

        self.model.train()
        self.model.ep_g = ep_g
        self.model.prompt.ep_g = ep_g
        self.data_weighting(self.train_dataset)
        if isinstance(self.device, int):
            if "CLsamefix" in self.model.args.method and ep_g >= 3:
                params_to_opt = list(self.model.prompt.parameters())
            else:
                params_to_opt = list(self.model.prompt.parameters()) + list(self.model.fc.parameters())
            #print(params_to_opt)
        else:
            params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.fc.parameters())
        
        
        if group == 0:
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_opt), lr=self.learning_rate,
                                         weight_decay=0, betas=(0.9, 0.999))
            scheduler = CosineSchedule(opt, K=50)
            #for e in range((ep_g % 4) * self.epochs):
                #scheduler.step()
            #return len(self.train_loader), opt.state_dict(), scheduler.state_dict(), self.current_class
        else:
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_opt), lr=self.learning_rate,
                                         weight_decay=0, betas=(0.9, 0.999))
            
            scheduler = CosineSchedule(opt, K=50)
            #for e in range((ep_g % 4) * self.epochs):
                #scheduler.step()
        
        self.old_model = model_old #old task
        self.old_round_model = copy.deepcopy(self.model) #old round
        if self.old_model != None:
            print('load old model')
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()
        
        
        if "full" in self.model.args.method and self.model.client_index != 0 and self.task_id_old != 0:
            pass
        else:
            for epoch in range(self.epochs):
                if epoch > 0:
                    scheduler.step()
                for param_group in opt.param_groups:
                    self.log('LR:', param_group['lr'])
                if "classincremental" not in self.model.args.method:
                    loss_value_sum, normal_loss_sum, t_c2loss_sum, prompt_loss_sum, prelogits_disloss_sum, output_disloss_sum, ntd_loss_sum, align_loss_sum, len_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0
                else:
                    loss_value_sum, normal_loss_sum, t_c2loss_sum, prompt_loss_sum, prelogits_disloss_sum, output_disloss_sum, ntd_loss_sum, align_loss_sum, len_sum, logits_loss_for_divide_sum, repre_loss_for_divide_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    
                    if "classincremental" not in self.model.args.method:
                        loss_value, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss, ntd_loss, align_loss, mean_aqk_list = self._compute_loss(indexs, images, target)
                    else:
                        loss_value, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss, ntd_loss, align_loss, mean_aqk_list, logits_loss_for_divide, repre_loss_for_divide = self._compute_loss(indexs, images, target)
                   

                    loss_value_sum += loss_value
                    normal_loss_sum += normal_loss
                    t_c2loss_sum += t_c2loss
                    prompt_loss_sum += prompt_loss
                    prelogits_disloss_sum += prelogits_disloss
                    output_disloss_sum += output_disloss
                    ntd_loss_sum += ntd_loss
                    align_loss_sum += align_loss
                    if "classincremental" in self.model.args.method:
                        logits_loss_for_divide_sum += logits_loss_for_divide
                        repre_loss_for_divide_sum += repre_loss_for_divide
                        
                    len_sum += 1
                    loss_all = loss_value
                    opt.zero_grad()
                    loss_all.backward()
                    opt.step()
                if epoch % 1 == 0:
                    if "classincremental" not in self.model.args.method:
                        print('{}/{} {} {} {} {} {} {} {} {}'.format(epoch, self.epochs, loss_value_sum/len_sum, normal_loss_sum/len_sum, t_c2loss_sum/len_sum, prompt_loss_sum/len_sum, prelogits_disloss_sum/len_sum, output_disloss_sum/len_sum, ntd_loss_sum/len_sum, align_loss_sum/len_sum)) #output_disloss is also control loss
                    else:
                        print('{}/{} {} {} {} {} {} {} {} {} {} {}'.format(epoch, self.epochs, loss_value_sum/len_sum, normal_loss_sum/len_sum, t_c2loss_sum/len_sum, prompt_loss_sum/len_sum, prelogits_disloss_sum/len_sum, output_disloss_sum/len_sum, ntd_loss_sum/len_sum, align_loss_sum/len_sum, logits_loss_for_divide_sum/len_sum, repre_loss_for_divide_sum/len_sum))
        
        print("************ begin ova training **************")
        print(self.client_learned_task_id)
        if "classincremental" in self.model.args.method:
            if "full" in self.model.args.method and self.model.client_index != 0 and self.task_id_old != 0:
                pass
            else:
                with torch.no_grad():
                    task_embedding = None
                    for step, (indexs, images, target) in enumerate(self.train_loader):
                        if isinstance(self.device, int):
                            images, target = images.cuda(self.device), target.cuda(self.device)
                        else:
                            images, target = images.cuda(), target.cuda()
                        q, _, _, q_map = self.model.feat(images)
                        q = q[:,0,:]
                        _, _, _, _, _, _, out_divide = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
                        if task_embedding is None:
                            out_divide = out_divide[:,3 * self.model.prompt.e_p_length_2,:]
                            task_embedding = out_divide
                        else:
                            out_divide = out_divide[:,3 * self.model.prompt.e_p_length_2,:]
                            task_embedding = torch.cat((task_embedding, out_divide), dim=0)
                    self.model.prompt.task_embedding[self.model.prompt.global_task_id_real[self.model.prompt.task_id * self.model.prompt.num_clients + self.model.prompt.client_index], :] = torch.mean(task_embedding, dim=0)

        return len(self.train_loader), opt.state_dict(), scheduler.state_dict(), self.current_class
    
    def generate_consolidation_dataset(self, class_real):
        weight = self.model.prompt.weight.detach().clone()
        global_task_id = self.model.prompt.task_id * self.model.prompt.num_clients + self.model.prompt.client_index
        global_task_id = self.model.prompt.global_task_id_real[global_task_id]
        _, idx = weight[global_task_id][self.model.prompt.client_learned_global_task_id].topk(2)
        target_task_id = self.model.prompt.client_learned_global_task_id[idx[1]]
        target_task_id_data_list = []
        for i in self.model.prompt.global_task_id_real.keys():
            if self.model.prompt.global_task_id_real[i] == target_task_id:
                target_task_id_data_list.append(i)   

        origin_client_index = self.model.client_index
        origin_model_task_id = self.model.prompt.task_id
        origin_class_min_output = self.model.client_class_min_output
        origin_class_max_output = self.model.client_class_max_output

        self.origin_train_dataset = copy.deepcopy(self.train_dataset)
        origin_train_loader = DataLoader(dataset=self.origin_train_dataset,
                                        shuffle=False,
                                        batch_size=self.batch_size,
                                        num_workers=8,
                                        pin_memory=True)
        with torch.no_grad():
            if "classincremental" in self.model.args.method:
                origin_representation_label = None
                origin_logits_label = None
            else:
                origin_logits_label = None
            for step, (indexs, images, target) in enumerate(origin_train_loader):
                if isinstance(self.device, int):
                    images, target = images.cuda(self.device), target.cuda(self.device)
                else:
                    images, target = images.cuda(), target.cuda()
                q, _, _, q_map = self.model.feat(images)
                q = q[:,0,:]
                if "classincremental" in self.model.args.method:
                    out, _, _, _, _, _, out_divide = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
                    if "v2" in self.model.args.method:
                        out = out[:,3 * self.model.prompt.e_p_length,:]
                        out_divide = out_divide[:,3 * self.model.prompt.e_p_length_2,:]
                    else:
                        out = out[:,0,:]
                        out_divide = out_divide[:,0,:]
                else:
                    out, _, _, _, _, _ = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
                    if "v2" in self.model.args.method:
                        out = out[:,3 * self.model.prompt.e_p_length,:]
                    else:
                        out = out[:,0,:]
                out = out.view(out.size(0), -1)
                out = self.model.fc(out, self.model.client_class_min_output, self.model.client_class_max_output)
                out[:,self.model.client_class_min_output] = -float('inf')
                if "classincremental" in self.model.args.method:
                    if origin_representation_label is None:
                        origin_representation_label = out_divide
                        origin_logits_label = out
                    else:
                        origin_representation_label = torch.cat((origin_representation_label, out_divide), dim=0)
                        origin_logits_label = torch.cat((origin_logits_label, out), dim=0)
                else:
                    if origin_logits_label is None:
                        origin_logits_label = out
                    else:
                        origin_logits_label = torch.cat((origin_logits_label, out), dim=0)
        
        
        target_class_min_output_list = []
        target_class_max_output_list = []
        for i in target_task_id_data_list:
            target_client_index = int(i % self.model.args.num_clients)
            target_model_task_id = int(i // self.model.args.num_clients)
            target_class = self.model.class_distribution[target_client_index][target_model_task_id]
            target_class_list = []
            for j in target_class:
                target_class_list.append(class_real[j])
            target_class = target_class_list
            target_class_real = self.model.class_distribution_real[target_client_index][target_model_task_id]
            target_class_min_output = sorted(list(set(list(range(self.model.args.numclass)))-set(target_class)))
            target_class_min_output_list.append(target_class_min_output)
            target_class_max_output = target_class
            target_class_max_output_list.append(target_class_max_output)
            self.model.client_index = target_client_index
            self.model.prompt.client_index = target_client_index
            self.model.prompt.task_id = target_model_task_id
            self.model.client_class_min_output = target_class_min_output
            self.model.client_class_max_output = target_class_max_output
            
            self.target_train_dataset.append(copy.deepcopy(self.train_dataset))
            print(class_real)
            print(self.exemplar_dict.keys())
            print(target_class)
            print("consolidated model client id:")
            print(target_client_index)
            print("consolidated model task id:")
            print(target_model_task_id)
            self.target_train_dataset[-1].getDistillTrainData(self.exemplar_dict, target_class)
            target_train_loader = DataLoader(dataset=self.target_train_dataset[-1],
                                            shuffle=False,
                                            batch_size=self.batch_size,
                                            num_workers=8,
                                            pin_memory=True)
            with torch.no_grad():
                if "classincremental" in self.model.args.method:
                    target_representation_label = None
                    target_logits_label = None
                else:
                    target_logits_label = None
                for step, (indexs, images, target) in enumerate(target_train_loader):
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    q, _, _, q_map = self.model.feat(images)
                    q = q[:,0,:]
                    if "classincremental" in self.model.args.method:
                        out, _, _, _, _, _, out_divide = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
                        if "v2" in self.model.args.method:
                            out = out[:,3 * self.model.prompt.e_p_length,:]
                            out_divide = out_divide[:,3 * self.model.prompt.e_p_length_2,:]
                        else:
                            out = out[:,0,:]
                            out_divide = out_divide[:,0,:]
                    else:
                        out, _, _, _, _, _ = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
                        if "v2" in self.model.args.method:
                            out = out[:,3 * self.model.prompt.e_p_length,:]
                        else:
                            out = out[:,0,:]
                    out = out.view(out.size(0), -1)
                    out = self.model.fc(out, self.model.client_class_min_output, self.model.client_class_max_output)
                    out[:,self.model.client_class_min_output] = -float('inf')
                    if "classincremental" in self.model.args.method:
                        if target_representation_label is None:
                            target_representation_label = out_divide
                            target_logits_label = out
                        else:
                            target_representation_label = torch.cat((target_representation_label, out_divide), dim=0)
                            target_logits_label = torch.cat((target_logits_label, out), dim=0)
                    else:
                        if target_logits_label is None:
                            target_logits_label = out
                        else:
                            target_logits_label = torch.cat((target_logits_label, out), dim=0)

        
            if "classincremental" in self.model.args.method:
                self.origin_train_dataset_divide = copy.deepcopy(self.origin_train_dataset)
                self.target_train_dataset_divide.append(copy.deepcopy(self.target_train_dataset))
                self.origin_train_dataset_divide.getDistillTrainDataLabel(origin_representation_label.cpu().numpy())
                self.target_train_dataset_divide[-1].getDistillTrainDataLabel(target_representation_label.cpu().numpy())
            self.origin_train_dataset.getDistillTrainDataLabel(origin_logits_label.cpu().numpy())
            self.target_train_dataset[-1].getDistillTrainDataLabel(target_logits_label.cpu().numpy())

        
        self.model.client_index = origin_client_index
        self.model.prompt.task_id = origin_model_task_id
        self.model.client_class_min_output = origin_class_min_output
        self.model.client_class_max_output = origin_class_max_output
        for t in target_task_id_data_list:
            self.model.prompt.global_task_id_real[t] = global_task_id
        print("global_task_id_real the dict:")
        print(self.model.prompt.global_task_id_real)
        return target_task_id_data_list, global_task_id, target_class_min_output_list, target_class_max_output_list


    def consolidation_train(self, target_class_min_output_list, target_class_max_output_list):
        origin_train_loader = DataLoader(dataset=self.origin_train_dataset,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  num_workers=8,
                                  pin_memory=True)
        target_train_loader = []
        target_train_loader_iter = []
        for i in range(len(self.target_train_dataset)):
            target_train_loader_1 = DataLoader(dataset=self.target_train_dataset[i],
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    num_workers=8,
                                    pin_memory=True)
        target_train_loader.append(target_train_loader_1)
        target_train_loader_iter.append(iter(target_train_loader_1))
        if "classincremental" in self.model.args.method:
            origin_train_loader_divide = DataLoader(dataset=self.origin_train_dataset_divide,
                                    shuffle=True,
                                    batch_size=self.batch_size,
                                    num_workers=8,
                                    pin_memory=True)
            origin_train_loader_divide_iter = iter(origin_train_loader_divide)
            target_train_loader_divide = []
            target_train_loader_divide_iter = []
            for i in range(len(self.target_train_dataset_divide)):
                target_train_loader_divide_1 = DataLoader(dataset=self.target_train_dataset_divide[i],
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        num_workers=8,
                                        pin_memory=True)
            target_train_loader_divide.append(target_train_loader_divide_1)
            target_train_loader_divide_iter.append(iter(target_train_loader_divide_1))
        self.model.train()
        if isinstance(self.device, int):
            params_to_opt = list(self.model.prompt.parameters())
        else:
            params_to_opt = list(self.model.module.prompt.parameters())
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, params_to_opt), lr=self.learning_rate,
                                         weight_decay=0, betas=(0.9, 0.999))
        scheduler = CosineSchedule(opt, K=5)
        for epoch in range(5):
            if epoch > 0:
                scheduler.step()
            for param_group in opt.param_groups:
                self.log('LR:', param_group['lr'])
            if "classincremental" in self.model.args.method:
                origin_loss_divide_sum, target_loss_divide_sum, origin_loss_sum, target_loss_sum, len_sum = 0, 0, 0, 0, 0
            else:
                origin_loss_sum, target_loss_sum, len_sum = 0, 0, 0
            for step, (origin_indexs, origin_images, origin_target) in enumerate(origin_train_loader):
                if isinstance(self.device, int):
                    origin_images, origin_target = origin_images.cuda(self.device), origin_target.cuda(self.device)
                else:
                    origin_images, origin_target = origin_images.cuda(), origin_target.cuda()
                origin_loss = self.batch_consolidation_forward(origin_images, origin_target, self.model.client_class_min_output, self.model.client_class_max_output)
                if "classincremental" in self.model.args.method:
                    try:
                        origin_indexs_divide, origin_images_divide, origin_target_divide = next(origin_train_loader_divide_iter)
                    except:
                        origin_train_loader_divide_iter = iter(origin_train_loader_divide)
                        origin_indexs_divide, origin_images_divide, origin_target_divide = next(origin_train_loader_divide_iter)
                    if isinstance(self.device, int):
                        origin_images_divide, origin_target_divide = origin_images_divide.cuda(self.device), origin_target_divide.cuda(self.device)
                    else:
                        origin_images_divide, origin_target_divide = origin_images_divide.cuda(), origin_target_divide.cuda()
                    origin_loss_divide = self.batch_consolidation_forward_divide(origin_images_divide, origin_target_divide)
                target_loss, target_loss_divide = 0, 0
                for i in range(len(target_train_loader_iter)):
                    try:
                        target_indexs, target_images, target_target = next(target_train_loader_iter[i])
                    except:
                        target_train_loader_iter[i] = iter(target_train_loader[i])
                        target_indexs, target_images, target_target = next(target_train_loader_iter[i])
                    if isinstance(self.device, int):
                        target_images, target_target = target_images.cuda(self.device), target_target.cuda(self.device)
                    else:
                        target_images, target_target = target_images.cuda(), target_target.cuda()
                    target_loss += self.batch_consolidation_forward(target_images, target_target, target_class_min_output_list[i], target_class_max_output_list[i])
                    if "classincremental" in self.model.args.method:
                        try:
                            target_indexs_divide, target_images_divide, target_target_divide = next(target_train_loader_divide_iter[i])
                        except:
                            target_train_loader_divide_iter[i] = iter(target_train_loader_divide[i])
                            target_indexs_divide, target_images_divide, target_target_divide = next(target_train_loader_divide_iter[i])
                        if isinstance(self.device, int):
                            target_images_divide, target_target_divide = target_images_divide.cuda(self.device), target_target_divide.cuda(self.device)
                        else:
                            target_images_divide, target_target_divide = target_images_divide.cuda(), target_target_divide.cuda()
                        target_loss_divide += self.batch_consolidation_forward_divide(target_images_divide, target_target_divide)

                if "classincremental" in self.model.args.method:
                    loss_all = origin_loss_divide + target_loss_divide + origin_loss + target_loss
                    origin_loss_divide_sum += origin_loss_divide
                    target_loss_divide_sum += target_loss_divide
                    origin_loss_sum += origin_loss
                    target_loss_sum += target_loss
                else:
                    loss_all = origin_loss + target_loss
                    origin_loss_sum += origin_loss
                    target_loss_sum += target_loss
                opt.zero_grad()
                loss_all.backward()
                opt.step()
            if "classincremental" in self.model.args.method:
                if epoch % 1 == 0:
                    print('{}/{} {} {} {} {}'.format(epoch, self.epochs, origin_loss_divide_sum/len_sum, target_loss_divide_sum/len_sum, origin_loss_sum/len_sum, target_loss_sum/len_sum)) 
                else:
                    print('{}/{} {} {}'.format(epoch, self.epochs, origin_loss_sum/len_sum, target_loss_sum/len_sum))

        self.origin_train_dataset = None
        self.target_train_dataset = []
        self.origin_train_dataset_divide = None
        self.target_train_dataset_divide = []
        
    def batch_consolidation_forward(self, images, targets, class_min_output, class_max_output):
        q, _, _, q_map = self.model.feat(images)
        q = q[:,0,:]
        if "classincremental" in self.model.args.method:
            out, _, _, _, _, _, out_divide = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
            if "v2" in self.model.args.method:
                out = out[:,3 * self.model.prompt.e_p_length,:]
                out_divide = out_divide[:,3 * self.model.prompt.e_p_length_2,:]
            else:
                out = out[:,0,:]
                out_divide = out_divide[:,0,:]
        else:
            out, _, _, _, _, _ = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
            if "v2" in self.model.args.method:
                out = out[:,3 * self.model.prompt.e_p_length,:]
            else:
                out = out[:,0,:]
        out = out.view(out.size(0), -1)
        out = self.model.fc(out, class_min_output, class_max_output)
        
        targets = targets[:, class_max_output]
        targets_probs = torch.softmax(targets / 1, dim=1)
        out = out[:, class_max_output]
        out_probs = torch.softmax(out / 1, dim=1)
        loss = (1 ** 2) * self.KLDiv(out_probs, targets_probs)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss.view(-1, 1), dim=0)
        return loss

    def batch_consolidation_forward_divide(self, images, targets):
        q, _, _, q_map = self.model.feat(images)
        q = q[:,0,:]
        if "classincremental" in self.model.args.method:
            out, _, _, _, _, _, out_divide = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
            if "v2" in self.model.args.method:
                out = out[:,3 * self.model.prompt.e_p_length,:]
                out_divide = out_divide[:,3 * self.model.prompt.e_p_length_2,:]
            else:
                out = out[:,0,:]
                out_divide = out_divide[:,0,:]
        else:
            out, _, _, _, _, _ = self.model.feat(images, prompt=self.model.prompt, q=q, train=True, task_id=self.model.task_id, aq_k=None, ep_g=self.model.ep_g, client_index=self.model.prompt.client_index)
            if "v2" in self.model.args.method:
                out = out[:,3 * self.model.prompt.e_p_length,:]
            else:
                out = out[:,0,:]
        out = out.view(out.size(0), -1)
        loss = torch.sum(out * targets, dim=1).mean()
        return loss

    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = torch.tensor(np.ones(self.numclass + 1, dtype=np.float32))
        # cuda
        if isinstance(self.device, int):
            self.dw_k = self.dw_k.cuda(self.device)
        else:
            self.dw_k = self.dw_k.cuda()


    def _compute_loss(self, indexs, imgs, targets):
        if isinstance(self.device, int):
            t_c2loss = torch.zeros((1,), requires_grad=True).cuda(self.device)
            output_disloss = torch.zeros((1,), requires_grad=True).cuda(self.device)
            prelogits_disloss = torch.zeros((1,), requires_grad=True).cuda(self.device)
        else:
            t_c2loss = torch.zeros((1,), requires_grad=True).cuda()
            output_disloss = torch.zeros((1,), requires_grad=True).cuda()
            prelogits_disloss = torch.zeros((1,), requires_grad=True).cuda()
        if self.prompt_flag == 'codap_2d_v2' and "classincremental" in self.model.args.method:
            logits, prompt_loss, prelogits_current, prompt_client_current, control_loss, mean_aqk_list, q_map, out_map, logits_loss_for_divide, repre_loss_for_divide = self.model(imgs, train=True, device=self.device)
            
        elif self.prompt_flag == 'codap_2d_v2':
            logits, prompt_loss, prelogits_current, prompt_client_current, control_loss, mean_aqk_list, q_map, out_map = self.model(imgs, train=True, device=self.device)
        else:
            logits, prompt_loss, prelogits_current, prompt_client_current, control_loss, mean_aqk_list, q_map, out_map = self.model(imgs, train=True, device=self.device)
        
        if self.old_model == None:
            if self.prompt_flag == 'cprompt':
                with torch.no_grad():
                    logits_global, _, prelogits_global, prompt_client_global, _, _, _, _ = self.old_round_model(imgs, train=True)               
                t_c2loss = 0
                prelogits_disloss = positive_loss()(prelogits_current, prelogits_global)
        else:
            if self.prompt_flag == 'cprompt':
                with torch.no_grad():
                    logits_previous_task, _, prelogits_previous_task, prompt_client_previous, _, _, _, _ = self.old_model(imgs, train=True)
                    _, _, prelogits_global, prompt_client_global, _, _, _, _ = self.old_round_model(imgs, train=True)
                
                t_c2loss = 0
                prelogits_disloss = self.triplet_loss(prelogits_current, prelogits_global, prelogits_previous_task)
                
         
        
        if self.imbalance == 'importance':
            dw_cls = self.efficient_old_class_weight(logits, targets)
        elif self.imbalance == 'number':
            dw_cls = self.number_imbalance[targets]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        normal_loss = self.criterion(logits, targets.long(), dw_cls)
        ntd_loss = 0
        
        if "classincremental" in self.model.args.method:
            if self.model.ep_g > 0 and 'v2' in self.model.args.method and '_ntd' not in self.model.args.method:
                with torch.no_grad():
                    dg_logits, _, _, _, _, _, _, _, _, _ = self.old_round_model(imgs, train=True, device=self.device)
                ntd_loss = self._ntd_loss(logits, dg_logits, targets)
            elif self.model.ep_g > 0 and 'v2' in self.model.args.method and 'nor' in self.model.args.method:
                with torch.no_grad():
                    dg_logits, _, _, _, _, _, _, _, _, _ = self.old_round_model(imgs, train=True, device=self.device)
                ntd_loss = self._nor_loss(logits, dg_logits, targets)
        else:
            if self.model.ep_g > 0 and 'v2' in self.model.args.method and '_ntd' not in self.model.args.method:
                with torch.no_grad():
                    dg_logits, _, _, _, _, _, _, _ = self.old_round_model(imgs, train=True, device=self.device)
                ntd_loss = self._ntd_loss(logits, dg_logits, targets)
            elif self.model.ep_g > 0 and 'v2' in self.model.args.method and 'nor' in self.model.args.method:
                with torch.no_grad():
                    dg_logits, _, _, _, _, _, _, _ = self.old_round_model(imgs, train=True, device=self.device)
                ntd_loss = self._nor_loss(logits, dg_logits, targets)
        
    
        align_loss = 0
        if self.imbalance == 'importance' or self.imbalance == 'number':
            control_loss = (control_loss * dw_cls).sum()
        else:
            control_loss = (control_loss * dw_cls).mean()
        if self.prompt_flag == 'cprompt':
            total_loss = normal_loss + prelogits_disloss
        elif self.prompt_flag == 'codap':
            total_loss = normal_loss + prompt_loss.sum() + ntd_loss
        elif self.prompt_flag == 'dual':
            total_loss = normal_loss + prompt_loss
        elif self.prompt_flag == 'l2p':
            total_loss = normal_loss + prompt_loss
        elif self.prompt_flag == 'codap_weight':
            total_loss = normal_loss + prompt_loss.sum()
        elif self.prompt_flag == 'codap_2d':
            total_loss = normal_loss + prompt_loss.sum()
        elif self.prompt_flag == 'codap_2d_v2' and "classincremental" not in self.model.args.method:
            total_loss = normal_loss + prompt_loss.sum() + control_loss + ntd_loss + 0.05 * align_loss
        elif self.prompt_flag == 'codap_2d_v2' and "classincremental" in self.model.args.method:
            total_loss = normal_loss + control_loss + ntd_loss + 0.05 * align_loss + 0 * logits_loss_for_divide + 0.1 * repre_loss_for_divide
        if  "classincremental" not in self.model.args.method:
            if self.prompt_flag == 'codap_2d_v2' or self.prompt_flag == 'codap':
                return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, control_loss, ntd_loss, align_loss, mean_aqk_list
            else:
                return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss, 0, 0, None
        else:
            if self.prompt_flag == 'codap_2d_v2' or self.prompt_flag == 'codap':
                return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, control_loss, ntd_loss, align_loss, mean_aqk_list, logits_loss_for_divide, repre_loss_for_divide
            else:
                return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss, 0, 0, None, 0, 0
        
    
        
    def _compute_loss_withmodel(self, model, indexs, imgs, targets):
        
        if isinstance(self.device, int):
            t_c2loss = torch.zeros((1,), requires_grad=True).cuda(self.device)
            output_disloss = torch.zeros((1,), requires_grad=True).cuda(self.device)
            prelogits_disloss = torch.zeros((1,), requires_grad=True).cuda(self.device)
        else:
            t_c2loss = torch.zeros((1,), requires_grad=True).cuda()
            output_disloss = torch.zeros((1,), requires_grad=True).cuda()
            prelogits_disloss = torch.zeros((1,), requires_grad=True).cuda()
        if self.prompt_flag == 'codap_2d_v2':
            logits, prompt_loss, prelogits_current, prompt_client_current, control_loss, mean_aqk_list = model(imgs, train=True, device=self.device)
        else:
            logits, prompt_loss, prelogits_current, prompt_client_current, control_loss, mean_aqk_list = model(imgs, train=True, device=self.device)
        
        if self.old_model == None:
            if self.prompt_flag == 'cprompt':
                with torch.no_grad():
                    logits_global, _, prelogits_global, prompt_client_global = self.old_round_model(imgs, train=True)
                t_c2loss = positive_loss()(prompt_client_current[0].view(self.batch_size, -1), prompt_client_global[0].view(self.batch_size, -1))
                prelogits_disloss = positive_loss()(prelogits_current, prelogits_global)
            
            
        else:
            if self.prompt_flag == 'cprompt':
                with torch.no_grad():
                    logits_previous_task, _, prelogits_previous_task, prompt_client_previous = self.old_model(imgs, train=True)
                    _, _, prelogits_global, prompt_client_global = self.old_round_model(imgs, train=True)
                t_c2loss =self.triplet_loss(prompt_client_current[0].view(self.batch_size, -1), prompt_client_global[0].view(self.batch_size, -1), prompt_client_previous[0].view(self.batch_size, -1))
                prelogits_disloss = self.triplet_loss(prelogits_current, prelogits_global, prelogits_previous_task)
                
            
        if self.imbalance == 'importance':
            dw_cls = self.efficient_old_class_weight(logits, targets)
        elif self.imbalance == 'number':
            dw_cls = self.number_imbalance[targets-torch.max(targets)]
        else:
            dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        
        normal_loss = self.criterion(logits, targets.long(), dw_cls)
        if self.imbalance == 'importance' or self.imbalance == 'number':
            control_loss = torch.sum(control_loss.view(-1, 1) * dw_cls, dim=0)
        else:
            control_loss = (control_loss * dw_cls).mean()
        if self.prompt_flag == 'cprompt':
            total_loss = normal_loss + prelogits_disloss
        elif self.prompt_flag == 'codap':
            total_loss = normal_loss + prompt_loss.sum()
        elif self.prompt_flag == 'dual':
            total_loss = normal_loss + prompt_loss
        elif self.prompt_flag == 'l2p':
            total_loss = normal_loss + prompt_loss
        elif self.prompt_flag == 'codap_weight':
            total_loss = normal_loss + prompt_loss.sum()
        elif self.prompt_flag == 'codap_2d':
            total_loss = normal_loss + prompt_loss.sum()
        elif self.prompt_flag == 'codap_2d_v2':
            total_loss = normal_loss + prompt_loss.sum() + control_loss
        if self.prompt_flag == 'codap_2d_v2':
            return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, control_loss
        else:
            return total_loss, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss
    
    def transmitting_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp
    
    def top_eigenvalue(self, K, n_power_iterations=10, dim=1):
            v = torch.ones(K.shape[0], K.shape[1], 1).to(self.device)
            for _ in range(n_power_iterations):
                m = torch.bmm(K, v)
                n = torch.norm(m, dim=1).unsqueeze(1)
                v = m / n

            top_eigenvalue = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
            return top_eigenvalue

    def criterion(self, logits, targets, data_weights):
        
        if self.imbalance == 'importance' or self.imbalance == 'number':
            loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).sum()
        else:
            loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        
        return loss_supervised
    
    def refine_as_not_true(self, logits, targets, num_classes):
        nt_positions = torch.arange(0, num_classes).to(logits.device)
        nt_positions = nt_positions.repeat(logits.size(0), 1)
        nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
        nt_positions = nt_positions.view(-1, num_classes - 1)

        logits = torch.gather(logits, 1, nt_positions)

        return logits

    def refine_as_not_true2(self, logits, targets, current_classes):
        nt_positions = torch.tensor(current_classes).to(logits.device)
        nt_positions = nt_positions.repeat(logits.size(0), 1)
        nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
        nt_positions = nt_positions.view(-1, len(current_classes) - 1)

        logits = torch.gather(logits, 1, nt_positions)

        return logits

    
    def calculate_distillation_weight(self, targets):
        distillation_weight = self.model.prompt.fc_weight.detach().clone()[targets.view(-1, 1)]
        distillation_weight, _ = torch.max(distillation_weight, dim=2)
        distillation_weight = 1 - distillation_weight
        distillation_weight = distillation_weight.squeeze()
        distillation_weight = torch.softmax(distillation_weight, dim=0)
        return distillation_weight
        
        
    def _nor_loss(self, logits, dg_logits, targets, tau=3):
        
        logits = logits[:, self.current_class]
        #logits = self.refine_as_not_true2(logits, targets, self.current_class)
        pred_probs = F.log_softmax(logits / tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            #dg_logits = self.refine_as_not_true(dg_logits, targets, self.numclass)
            #dg_logits = dg_logits[:, self.current_class[0]:self.current_class[len(self.current_class) - 1]]
            dg_logits = dg_logits[:, self.current_class]
            #dg_logits = self.refine_as_not_true2(dg_logits, targets, self.current_class)
            dg_probs = torch.softmax(dg_logits / tau, dim=1)
        
        
        
        loss = (tau ** 2) * self.KLDiv(pred_probs, dg_probs)
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss.view(-1, 1), dim=0)
        #loss = torch.sum(loss * distillation_weight, dim=1)
        #loss = torch.mean(loss, dim=0)
        return loss
    
    def _ntd_loss(self, logits, dg_logits, targets, tau=3):
        distillation_weight = self.calculate_distillation_weight(targets)
        
        if "classincremental" in self.model.args.method or "extension" in self.model.args.method or "extencl" in self.model.args.method:
            logits_dis = None
            dg_logits_dis = None
            for i in range(logits.shape[0]):
                distill_list = list(set(self.current_class) - set([int(targets[i])]))
                if logits_dis is None:
                    logits_dis = logits[i][distill_list].unsqueeze(0)
                else:
                    logits_dis = torch.cat((logits_dis, logits[i][distill_list].unsqueeze(0)), dim=0)
            pred_probs = F.log_softmax(logits_dis / tau, dim=1)
            with torch.no_grad():
                for i in range(logits.shape[0]):
                    distill_list = list(set(self.current_class) - set([int(targets[i])]))
                    if dg_logits_dis is None:
                        dg_logits_dis = dg_logits[i][distill_list].unsqueeze(0)
                    else:
                        dg_logits_dis = torch.cat((dg_logits_dis, dg_logits[i][distill_list].unsqueeze(0)), dim=0)
                dg_probs = torch.softmax(dg_logits_dis / tau, dim=1)
        else:
            logits = self.refine_as_not_true(logits, targets, self.numclass)
            
            logits = logits[:, self.current_class[0]:self.current_class[len(self.current_class) - 1]]
            #logits = logits[:, self.current_class]
            #logits = self.refine_as_not_true2(logits, targets, self.current_class)
            pred_probs = F.log_softmax(logits / tau, dim=1)

            # Get smoothed global model prediction
            with torch.no_grad():

                dg_logits = self.refine_as_not_true(dg_logits, targets, self.numclass)
                dg_logits = dg_logits[:, self.current_class[0]:self.current_class[len(self.current_class) - 1]]
                #dg_logits = dg_logits[:, self.current_class]
                #dg_logits = self.refine_as_not_true2(dg_logits, targets, self.current_class)
                dg_probs = torch.softmax(dg_logits / tau, dim=1)
        
        
        
        loss = (tau ** 2) * self.KLDiv(pred_probs, dg_probs)
        loss = torch.sum(loss, dim=1)
        loss = torch.sum(loss.view(-1, 1) * distillation_weight.view(-1, 1), dim=0)
        #loss = torch.sum(loss * distillation_weight, dim=1)
        #loss = torch.mean(loss, dim=0)
        
        return loss
    
    def efficient_old_class_weight(self, output, label):
        pred = torch.softmax(output, dim=1)
        #pred = output
        
        N, C = pred.size(0), pred.size(1)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = get_one_hot(label.view(-1, 1), self.numclass, self.device)
        #g = torch.abs(pred.detach() - target)
        g = torch.abs(pred.detach() * target - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.current_class) != 0:
            for i in self.current_class:
                ids = torch.where(ids != i, ids, ids.clone().fill_(-1))

            index1 = torch.eq(ids, -1).float()
            index2 = torch.ne(ids, -1).float()
            if index1.sum() != 0:
                w1 = torch.div(g * index1, (g * index1).sum() / index1.sum())
            else:
                w1 = g.clone().fill_(0.)
            if index2.sum() != 0:
                w2 = torch.div(g * index2, (g * index2).sum() / index2.sum())
            else:
                w2 = g.clone().fill_(0.)

            w = w1 + w2
        
        else:
            w = g.clone().fill_(1.)

        return w
    
    def compute_prompt_importance(self):
        model_importance = copy.deepcopy(self.model)
        model_importance.train()
        self.data_weighting(self.train_dataset)
        prompt_weight = []
        prompt_name = []
        for name, w in model_importance.named_parameters():
            if 'p_share_' in name or 'p_specific_' in name:
            
                prompt_weight.append(w)
                prompt_name.append(name)
        prompt_importance = []
        for w in prompt_weight:
            prompt_importance.append(torch.zeros(w.shape[1]).cuda(self.device))
        for step, (indexs, images, target) in enumerate(self.train_loader):
            if isinstance(self.device, int):
                images, target = images.cuda(self.device), target.cuda(self.device)
            else:
                images, target = images.cuda(), target.cuda()
            loss_value, normal_loss, t_c2loss, prompt_loss, prelogits_disloss, output_disloss = self._compute_loss_withmodel(model_importance, indexs, images, target)
            
            loss_value.backward()
            for name, w, current_importance in zip(prompt_name, prompt_weight, prompt_importance):
                current_importance += ((w * w.grad)[model_importance.prompt.task_id * model_importance.prompt.num_clients + model_importance.prompt.client_index].reshape(w.shape[1], -1).sum(dim=1)).abs().detach()

       
        return prompt_importance
            




    def reorder_prompt(self, prompt_importance):
        idx = []
        
        for pi in prompt_importance:
            
            idx.append(torch.sort(pi, descending=True)[-1]) 
        
        self.model.prompt.reorder_prompt(idx)
        return idx


        
    




