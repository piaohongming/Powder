import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random
from Fed_utils import * 
from fractal_learning.training import datamodule
import math
from models_fedspace.fedspace_network import fedspace_network
from models_fedspace.fedspace_network_hard import fedspace_network_hard
import dataloaders
from dataloaders.utils import *
from utils.schedulers import CosineSchedule

LOSS_KEYS = ["Proto_aug_loss"]


class FedSpace_model_hard:
    def __init__(self, numclass, feature_extractor, batch_size, task_size, epochs, learning_rate, train_set, device, optimizer, centralized_pretrain, centralized_fractal_pretrain_steps, temp, repr_loss_temp, lambda_proto_aug, lambda_repr_loss, dataset):
        super(FedSpace_model_hard, self).__init__()
        self.numclass = numclass
        self.model = None
        self.lambda_proto_aug = lambda_proto_aug
        self.lambda_repr_loss = lambda_repr_loss
        self.device = device
        self.current_class = None
        self.train_loader = None
        self.optimizer = optimizer
        self.epochs = epochs
        self.centralized_pretrain = centralized_pretrain
        self.centralized_fractal_pretrain_steps = centralized_fractal_pretrain_steps
        self.temp = temp
        self.repr_loss_temp = repr_loss_temp
        self.prototype = {"global": {}, "local": {}}
        self.radius = {"global": 0, "local": 0}
        self.batchsize = batch_size
        self.task_size = task_size
        self.task_id_old = -1
        self.learning_rate = learning_rate
        self.train_dataset = train_set
        self.learned_classes = []
        self.dataset = dataset
        self.real_task_id = -1
        #self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             #transforms.ToTensor(),
                                            #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        if dataset == 'MNIST':
            self.transform = transforms.Compose([#transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1,), (0.2752,))])
        else:
            self.transform = dataloaders.utils.get_transform(dataset=dataset, phase='train', aug=True, resize_imnet=True)


        if self.centralized_pretrain:
            self.pretrain_loader = datamodule.FractalClassDataModule().train_dataloader()

        self.client_learned_global_task_id = []

    
    def beforeTrain(self, task_id_new, group, proto_global, radius_global, client_index, global_task_id_real, class_real=None):
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
            #self.numclass = self.task_size * (task_id_new + 1)
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
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
            else:
                self.last_class = None
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
        
        if proto_global is not None:
            self.prototype['global'] = copy.deepcopy(proto_global)
            self.radius['global'] = copy.deepcopy(radius_global)

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion,False)
        print("current class:")
        print(self.current_class)
    
    def _get_train_and_test_dataloader(self, train_classes, train_classes_real, train_classes_proportion, mix):
        if mix:
            number_imbalance = self.train_dataset.getTrainImbalance(train_classes_real, self.exemplar_set, self.learned_classes, self.model.client_index)
            self.number_imbalance = torch.tensor(number_imbalance, requires_grad=False, device=self.device)
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.learned_classes, self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion)
        else:
            number_imbalance = self.train_dataset.getTrainImbalance(train_classes_real, [], [], self.model.client_index)
            self.number_imbalance = torch.tensor(number_imbalance, requires_grad=False, device=self.device)
            self.train_dataset.getTrainData(train_classes, [], [], self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion)

        #print(self.train_dataset.TrainData[0])
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=2,
                                  pin_memory=True)

        return train_loader
    
    
    def fractal_pretrain(self):
        self.model = model_to_device(self.model, False, self.device)
        self.model.train()
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                                        momentum=0.9, weight_decay=0.00001)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                                         weight_decay=0.00001)
        for epoch in range(1):
            for batch_idx, (images, labels) in enumerate(self.pretrain_loader):
                print('{}/{}'.format(batch_idx, len(self.pretrain_loader)), end='\r')
                if batch_idx == self.centralized_fractal_pretrain_steps:
                    break
                
                labels = labels.long()
                if isinstance(self.device, int):
                    images, labels = images.to(self.device), labels.to(self.device)
                else:
                    images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()
                output = self.model(images)
                loss = nn.CrossEntropyLoss()(output / self.temp, labels)
                loss.backward()

    def train(self, ep_g, model_old):
        
        self.model.train()
        self.model.ep_g = ep_g
        self.old_model = model_old

        if "sharedfc" in self.model.args.method:
            optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()), lr=self.learning_rate,
                                            weight_decay=0, betas=(0.9, 0.999))
        elif "sharedencoder" in self.model.args.method:
            optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters())+list(self.model.feature.parameters()), lr=self.learning_rate,
                                            weight_decay=0, betas=(0.9, 0.999))
        elif "sharedprompt" in self.model.args.method:
            optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()) + list([self.model.prompt[-1]]) + list([self.model.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                            weight_decay=0, betas=(0.9, 0.999))
        elif "sharedcodap" in self.model.args.method:
            optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()) + list([self.model.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                            weight_decay=0, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adam(list(self.model.fc.parameters())+list(self.model.client_fc.parameters()), lr=self.learning_rate,
                                            weight_decay=0, betas=(0.9, 0.999))

        scheduler = CosineSchedule(optimizer, K=self.epochs)
            
        epoch_loss = []
        loss_terms = epoch_loss_terms = {name: [] for name in LOSS_KEYS}
        num_sample_class = {k: 0 for k in range(self.numclass)}
        if "full" in self.model.args.method and self.model.client_index != 0 and self.task_id_old != 0:
            pass
        else:
            for epoch in range(self.epochs):
                batch_loss = []
                batch_loss_terms = {name: [] for name in LOSS_KEYS}
                if epoch > 0:
                    scheduler.step()
                for batch_idx, (indexs, images, labels) in enumerate(self.train_loader):
                    #print('{}/{} {}/{}'.format(batch_idx, len(self.train_loader), epoch, self.epochs), end='\r')
                    for lab in labels.tolist():
                        num_sample_class[lab] += 1
                    if isinstance(self.device, int):
                        images, labels = images.cuda(self.device), labels.cuda(self.device)
                    else:
                        images, labels = images.cuda(), labels.cuda()
                    #images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                    images = images.view(-1, 3, 224, 224)
                    labels = labels.view(-1)
                    #labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)

                    optimizer.zero_grad() 
                    loss, soft_predict, soft_label, soft_correct = self._compute_loss(images, labels, self.old_model)
                    if batch_idx % 2 == 0 and epoch % 2 == 0:
                        print('{}/{} {}/{}'.format(batch_idx, len(self.train_loader), epoch, self.epochs))
                    optimizer.zero_grad()
                    loss["Total_loss"].backward()
                    optimizer.step()
                    batch_loss.append(loss["Total_loss"].item())

                    for key in LOSS_KEYS:
                            batch_loss_terms[key].append(loss[key].item())
            
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                for key in LOSS_KEYS:
                    epoch_loss_terms[key].append(np.mean(batch_loss_terms[key]))
            
            for key in LOSS_KEYS:
                loss_terms[key].append(np.mean(epoch_loss_terms[key]))


        return loss_terms, num_sample_class, len(self.train_loader)
    

    def _compute_loss(self, images, labels, old_model):
        soft_feat_aug_predicts = None
        proto_aug_label = None
        correct = None
        
        output = self.model(images)
        
        loss_cls = nn.CrossEntropyLoss()(output / self.temp, labels)
        if isinstance(self.device, int):
            loss_dict = {"CE_loss": loss_cls,
                        "Proto_aug_loss": torch.zeros(1).cuda(self.device),
                        "Repr_learn_loss": torch.zeros(1).cuda(self.device),
                        "Proto_dis_loss": torch.zeros(1).cuda(self.device),
                        "Total_loss": 0}
        else:
            loss_dict = {"CE_loss": loss_cls,
                        "Proto_aug_loss": torch.zeros(1).cuda(),
                        "Repr_learn_loss": torch.zeros(1).cuda(),
                        "Proto_dis_loss": torch.zeros(1).cuda(),
                        "Total_loss": 0}
        
        proto_aug = []
        proto_aug_label = []
        location = 'global'

        prototype = self.prototype[location]
        radius = self.radius[location]

        index = [k for k, v in prototype.items() if np.sum(v) != 0 and k in sorted(list(set(self.current_class + self.learned_classes)))]

        
        if index:
            for i in range(self.batchsize):
                np.random.shuffle(index)
                temp = prototype[index[0]] + np.random.normal(0, 1, prototype[index[0]].shape[0]) * radius
                
                proto_aug.append(temp)
                proto_aug_label.append(index[0])
                
            if isinstance(self.device, int):
                proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            else:
                proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().cuda()
            if len(proto_aug.shape) > 2:
                proto_aug = proto_aug.squeeze()
            if isinstance(self.device, int):
                proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).long().to(self.device)
            else:
                proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).long().cuda()
            #old_model.eval()
            #soft_proto_label = torch.sigmoid(old_model.predict(proto_aug))
            soft_feat_aug = self.model.predict(proto_aug)
            soft_feat_aug_predicts = torch.max(soft_feat_aug, dim=1)[1]
            correct = (soft_feat_aug_predicts.cpu() == proto_aug_label.cpu()).sum()
            #print(correct)
            #soft_feat_aug = (soft_feat_aug[:, 0::4] + soft_feat_aug[:, 1::4] + soft_feat_aug[:, 2::4] + soft_feat_aug[:, 3::4]) / 4
            loss_dict["Proto_aug_loss"] = nn.CrossEntropyLoss()(soft_feat_aug / self.temp, proto_aug_label)
            #loss_dict["Proto_dis_loss"] = F.binary_cross_entropy_with_logits(soft_feat_aug, soft_proto_label)
            soft_feat_aug_predicts = soft_feat_aug_predicts.cpu()
            proto_aug_label = proto_aug_label.cpu()
        
        '''
        index = [k for k, v in prototype.items() if np.sum(v) != 0 and k in sorted(list(set(self.current_class))) and k in self.learned_classes]
        #index = [k for k, v in prototype.items() if np.sum(v) != 0]
        
        proto_aug_feat = []
        proto_aug_lab = []
        if index:
            for _ in range(self.batchsize):
                np.random.shuffle(index)
                temp = prototype[index[0]] + np.random.normal(0, 1, prototype[index[0]].shape[0]) * radius
                proto_aug_feat.append(temp)
                proto_aug_lab.append(index[0])
            if isinstance(self.device, int):   
                proto_aug_feat = torch.from_numpy(np.float32(np.asarray(proto_aug_feat))).float().to(self.device)
            else:
                proto_aug_feat = torch.from_numpy(np.float32(np.asarray(proto_aug_feat))).float().cuda()
            if len(proto_aug_feat.shape) > 2:
                proto_aug_feat = proto_aug_feat.squeeze()
            if isinstance(self.device, int):
                proto_aug_lab = torch.from_numpy(np.asarray(proto_aug_lab)).long().to(self.device)
            else:
                proto_aug_lab = torch.from_numpy(np.asarray(proto_aug_lab)).long().cuda()
            slc = slice(None)
            curr_cls_feat = feature[slc]
            if isinstance(self.device, int):
                curr_cls_lab = torch.tensor([i for i in labels][slc]).int().to(self.device)
            else:
                curr_cls_lab = torch.tensor([i for i in labels][slc]).int().cuda()
            #curr_cls_lab = torch.tensor([i / 4 for i in labels][slc]).int().to(self.device)
            feat, lab = torch.cat([curr_cls_feat, proto_aug_feat], dim=0), torch.cat([curr_cls_lab, proto_aug_lab], dim=0)
            feat = F.normalize(feat, p=2., dim=1)
            loss_tot = 0.
            for c in self.current_class:
                Nc = (lab==c).sum()
                if Nc<=1: continue
                feat_c = feat[lab==c]                                                       # Nc x D
                feat_not_c = feat[lab!=c]                                                   # Nnc x D
                pos = feat_c @ feat_c.T / self.repr_loss_temp                                                  # Nc x Nc
                pos[torch.eye(Nc).bool()] *= 0.
                neg = feat_c @ feat_not_c.T / self.repr_loss_temp                                              # Nc x Nnc
                loss = pos - torch.logsumexp(torch.cat([pos,neg], dim=1), dim=1).unsqueeze(1)   # Nc x Nc
                loss_tot += -1 * loss[~torch.eye(Nc).bool()].sum() / (Nc-1)
            loss_dict["Repr_learn_loss"] = loss_tot / lab.size(0)
        '''
        


        
        if old_model is None:
            if isinstance(self.device, int):
                loss_dict["Proto_aug_loss"] = torch.zeros(1).to(self.device)
            else:
                loss_dict["Proto_aug_loss"] = torch.zeros(1).cuda()
        
        loss_dict["Total_loss"] = loss_dict["CE_loss"] + \
            loss_dict["Proto_aug_loss"] * self.lambda_proto_aug + \
            loss_dict["Repr_learn_loss"] * self.lambda_repr_loss
            #loss_dict["Repr_learn_loss"] * self.lambda_repr_loss + loss_dict["Proto_dis_loss"]
        
        return loss_dict, soft_feat_aug_predicts, proto_aug_label, correct

            


    def proto_save(self, ):
        features = []
        labels = []
        self.model = model_to_device(self.model, False, self.device)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (indexs, images, target) in enumerate(self.train_loader):
                if isinstance(self.device, int):
                    feature = self.model.feature_extractor(images.to(self.device))
                else:
                    feature = self.model.feature_extractor(images.cuda())
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
                prototype[item] = np.zeros(self.feature_size,)
            else:
                prototype[item] = np.mean(feature_classwise, axis=0)

            cov = np.cov(feature_classwise.T)
            if not math.isnan(np.trace(cov)):
                    radius[item] = np.trace(cov) / feature_dim
            else:
                radius[item] = 0

        radius = np.sqrt(np.mean(list(radius.values())))
        self.model.train()
        return radius, prototype, class_label

            





        



