import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from myNetwork_hard import *
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random
from Fed_utils import * 
import dataloaders
from dataloaders.utils import *
from utils.schedulers import CosineSchedule

def get_one_hot(target, num_class, device):
    if isinstance(device, int):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    else:
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class GLFC_model_hard:

    def __init__(self, numclass, feature_extractor, batch_size, task_size, memory_size, epochs, learning_rate, train_set, device, encode_model, dataset):

        super(GLFC_model_hard, self).__init__()
        self.numclass = numclass
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.encode_model = encode_model
        self.dataset = dataset


        self.exemplar_set = []
        self.class_mean_set = []
        self.learned_numclass = 0
        self.learned_classes = []
        if dataset == 'MNIST':
            self.transform = transforms.Compose([#transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1,), (0.2752,))])
        else:
            self.transform = dataloaders.utils.get_transform(dataset=dataset, phase='train', aug=True, resize_imnet=True)

        self.old_model = None
        self.train_dataset = train_set
        self.start = True
        self.signal = False

        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size

        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.last_class_real = None
        self.last_class_proportion = None
        self.task_id_old = -1
        self.device = device
        self.last_entropy = 0
        self.real_task_id = -1
        self.client_learned_global_task_id = []

    # get incremental train data
    def beforeTrain(self, task_id_new, group, client_index, global_task_id_real, class_real=None):
        try:
            if "sharedcodap" in self.model.args.method:
                self.model.module.prompt.client_index = client_index
            self.model.module.client_index = client_index
        except:
            if "sharedcodap" in self.model.args.method:
                self.model.prompt.client_index = client_index
            self.model.client_index = client_index
        

        self.signal = False 
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            if group != 0:
                self.signal = True
                if self.current_class != None:
                    self.last_class = self.current_class
                    self.last_class_proportion = self.current_class_proportion
                    self.last_class_real = self.current_class_real
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
                self.last_class_real = None
                self.last_class_proportion = None
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

        #self.model.set_client_class_min_output()
        if "sharedcodap" in self.model.args.method:
            self.model.prompt.client_learned_global_task_id = self.client_learned_global_task_id
            self.model.prompt.global_task_id_real = global_task_id_real
        self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, False)
        
    def update_new_set(self, task_id, client_index):
        self.model = model_to_device(self.model, False, self.device)
        self.model.eval()
        

        if self.signal and (self.last_class != None):
            self.learned_numclass += len(self.last_class)
            self.learned_classes += self.last_class
        
            m = int(self.memory_size / self.learned_numclass)
            self._reduce_exemplar_sets(m)
            for i in self.last_class: 
                images = self.train_dataset.get_image_class(self.last_class_real[self.last_class.index(i)], self.model.client_index, self.last_class_proportion)
                self._construct_exemplar_set(images, m)

        self.model.train()
        if "sharedencoder" in self.model.args.method and "weit" in self.model.args.method:
            self.model.set_client_class_min_output(sorted(list(set(self.current_class + self.learned_classes))))
        elif "sharedprompt" in self.model.args.method and "weit" in self.model.args.method:
            self.model.set_client_class_min_output(sorted(list(set(self.current_class + self.learned_classes))))
        elif "sharedcodap" in self.model.args.method and "weit" in self.model.args.method:
            self.model.set_client_class_min_output(sorted(list(set(self.current_class + self.learned_classes))))    
        else:
            self.model.set_client_class_min_output(sorted(list(set(self.current_class + self.learned_classes))))
        self.model.set_learned_unlearned_class(sorted(list(set(self.current_class + self.learned_classes))))
        self.model.current_class = self.current_class
        if "notran" in self.model.args.method:
            self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, False)
        else:
            self.train_loader = self._get_train_and_test_dataloader(self.current_class, self.current_class_real, self.current_class_proportion, True)

    def _get_train_and_test_dataloader(self, train_classes, train_classes_real, train_classes_proportion, mix):
        if mix:
            self.train_dataset.getTrainData(train_classes, self.exemplar_set, self.learned_classes, self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion, exe_class=self.learned_classes)
            #self.train_dataset.getTrainData(train_classes, [], [])
        else:
            self.train_dataset.getTrainData(train_classes, [], [], self.model.client_index, classes_real=train_classes_real, classes_proportion=train_classes_proportion)

        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batchsize,
                                  num_workers=2,
                                  pin_memory=True)

        return train_loader

    # train model
    def train(self, ep_g, model_old):
        self.model = model_to_device(self.model, False, self.device)
        self.model.ep_g = ep_g
        if "weit" in self.model.args.method and self.signal and ep_g % self.model.args.tasks_global == 0:
            self.model.normalize_attention()
        #opt = optim.SGD(self.model.fc.parameters(), lr=self.learning_rate, weight_decay=0.00001) #因为换成了VIT，所以只优化FC
        if "sharedfc" in self.model.args.method:       
            if isinstance(self.device, int):
                opt = torch.optim.Adam(list(self.model.fc.parameters())+list([self.model.aggregate_weight] + list(self.model.client_fc.parameters())), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            else:
                opt = torch.optim.Adam(self.model.module.fc.parameters()+list([self.model.module.aggregate_weight] + list(self.model.module.client_fc.parameters())), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
        elif "sharedencoder" in self.model.args.method:
            if isinstance(self.device, int):
                opt = torch.optim.Adam(list(self.model.fc.parameters())+list([self.model.aggregate_weight] + list(self.model.client_fc.parameters())) + list(self.model.feature[-1].parameters()) + list(self.model.feature[self.model.task_id * self.model.args.num_clients + self.model.client_index].parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999), eps=1e-3)
            else:
                opt = torch.optim.Adam(self.model.module.fc.parameters()+list([self.model.module.aggregate_weight] + list(self.model.module.client_fc.parameters())) + list(self.model.module.feature[-1].parameters()) + list(self.model.module.feature[self.model.task_id * self.model.args.num_clients + self.model.client_index].parameters()), lr=self.learning_rate,
                                               weight_decay=0, betas=(0.9, 0.999), eps=1e-3)
        elif "sharedprompt" in self.model.args.method:
            if isinstance(self.device, int):
                opt = torch.optim.Adam(list(self.model.fc.parameters())+list([self.model.aggregate_weight] + list(self.model.client_fc.parameters())) + list([self.model.prompt[-1]]) + list([self.model.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999), eps=1e-3)
            else:
                opt = torch.optim.Adam(self.model.module.fc.parameters()+list([self.model.module.aggregate_weight] + list(self.model.module.client_fc.parameters())) + list([self.model.module.prompt[-1]]) + list([self.model.module.prompt[self.model.task_id * self.model.args.num_clients + self.model.client_index]]), lr=self.learning_rate,
                                               weight_decay=0, betas=(0.9, 0.999), eps=1e-3)
        elif "sharedcodap" in self.model.args.method:
            if isinstance(self.device, int):
                opt = torch.optim.Adam(list(self.model.fc.parameters())+list([self.model.aggregate_weight] + list(self.model.client_fc.parameters())) + list([self.model.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999), eps=1e-3)
            else:
                opt = torch.optim.Adam(self.model.module.fc.parameters()+list([self.model.module.aggregate_weight] + list(self.model.module.client_fc.parameters())) + list([self.model.module.global_prompt]) + list(self.model.prompt.parameters()), lr=self.learning_rate,
                                               weight_decay=0, betas=(0.9, 0.999), eps=1e-3)
        else:
            if isinstance(self.device, int):
                opt = torch.optim.Adam(list(self.model.fc.parameters())+list([self.model.aggregate_weight] + list(self.model.client_fc.parameters())), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
            else:
                opt = torch.optim.Adam(self.model.module.fc.parameters()+list([self.model.module.aggregate_weight] + list(self.model.module.client_fc.parameters())), lr=self.learning_rate,
                                                weight_decay=0, betas=(0.9, 0.999))
        scheduler = CosineSchedule(opt, K=self.epochs)
        
        if model_old[1] != None:
            if self.signal:
                self.old_model = model_old[1]
            else:
                self.old_model = model_old[0]
        else:
            if self.signal:
                self.old_model = model_old[0]
        
        if self.old_model != None:
            print('load old model')
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()
        
        if "full" in self.model.args.method and self.model.client_index != 0 and self.task_id_old != 0:
            pass
        else:
            for epoch in range(self.epochs):
                loss_cur_sum, loss_mmd_sum = [], []
                if epoch > 0:
                    scheduler.step()
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    #print(images.shape)
                    loss_value, loss_cur, loss_old = self._compute_loss(indexs, images, target)

                    if step % 2 == 0 and epoch % 2 == 0:
                        print('{}/{} {}/{} {} {} {}'.format(step, len(self.train_loader), epoch, self.epochs, loss_value, loss_cur, loss_old))
                    opt.zero_grad()
                    loss_value.backward()
                    opt.step()
        return len(self.train_loader)
        

    def entropy_signal(self, loader):
        return True

    def _compute_loss(self, indexs, imgs, label):
        output_ori = self.model(imgs)
        output = torch.sigmoid(output_ori)
        target = get_one_hot(label, self.numclass, self.device)
        
        if self.old_model == None:
            if "weit" in self.model.args.method:
                loss_cur = torch.mean(nn.CrossEntropyLoss()(output_ori, label))
            else:
                loss_cur = torch.mean(F.binary_cross_entropy(output, target, reduction='none'))
            return loss_cur, 0, 0
        else:
            if "fcil_imagenet" in self.model.args.method or "fcil_domainnet" in self.model.args.method:
                #w = self.efficient_old_class_weight(output, label)
                loss_cur = torch.mean(F.binary_cross_entropy(output, target, reduction='none'))
                #loss_cur = torch.mean(F.binary_cross_entropy_with_logits(output, target, reduction='none'))
                distill_target = target.clone()
                old_target = self.old_model(imgs)
                old_target[:, sorted(list(set(list(range(self.model.numclass))) - set(self.learned_classes)))] = -float('inf')
                old_target = torch.sigmoid(old_target)
                old_target_clone = old_target.clone()
                old_target_clone[..., self.current_class] = distill_target[..., self.current_class]
                loss_old = F.binary_cross_entropy(output, old_target_clone.detach())
                loss_proxi = 0
            elif "notran" in self.model.args.method:
                if "weit" in self.model.args.method:
                    loss_cur = torch.mean(nn.CrossEntropyLoss()(output_ori, label))
                    #loss_cur = torch.mean(F.binary_cross_entropy(output, target, reduction='none'))
                else:
                    loss_cur = torch.mean(F.binary_cross_entropy(output, target, reduction='none'))
                loss_old = 0
                loss_proxi = 0
            elif "weit" in self.model.args.method:
                loss_cur = torch.mean(nn.CrossEntropyLoss()(output_ori, label))
                loss_old = 0
                if "sharedfc" in self.model.args.method:
                    loss_proxi = torch.square(self.model.fc.weight.data[self.learned_classes] - self.old_model.fc.weight.data[self.learned_classes]).sum() + torch.square(self.model.client_fc.weight.data - self.old_model.client_fc.weight.data).sum()
                elif "sharedencoder" in self.model.args.method:
                    loss_proxi = torch.square(self.model.feature[-1].weight.data - self.old_model.feature[-1].weight.data).sum()
                elif "sharedprompt" in self.model.args.method:
                    loss_proxi = torch.square(self.model.prompt[-1].data - self.old_model.prompt[-1].data).sum()
                elif "sharedcodap" in self.model.args.method:
                    loss_proxi = torch.square(self.model.global_prompt.data - self.old_model.global_prompt.data).sum()
                else:
                    loss_proxi = 0
                if (not isinstance(loss_proxi, int)) and loss_proxi.item() != 0:
                    loss_proxi = (loss_cur.item()/loss_proxi.item()) * loss_proxi
                else:
                    loss_proxi = 0

            return loss_cur + loss_old + loss_proxi, loss_cur, loss_old


    def efficient_old_class_weight(self, output, label):
        pred = output
        
        N, C = pred.size(0), pred.size(1)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        target = get_one_hot(label, self.numclass, self.device)
        g = torch.abs(pred.detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)

        if len(self.learned_classes) != 0:
            for i in self.learned_classes:
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

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]

    def Image_transform(self, images, transform):
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
                data = torch.cat((data, self.transform(Image.fromarray(jpg_image_to_array(images[0]))).unsqueeze(0)), dim=0)
            elif self.dataset == 'ImageNet_R':
                data = torch.cat((data, self.transform(Image.fromarray(jpg_image_to_array(images[0]))).unsqueeze(0)), dim=0)
            else:
                data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        if isinstance(self.device, int):
            x = self.Image_transform(images, transform).cuda(self.device)
        else:
            x = self.Image_transform(images, transform).cuda()
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            exemplar=self.exemplar_set[index]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def proto_grad_sharing(self):
        if self.signal:
            proto_grad = self.prototype_mask()
        else:
            proto_grad = None

        return proto_grad

    def prototype_mask(self):
        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        iters = 50
        if isinstance(self.device, int):
            criterion = nn.CrossEntropyLoss().cuda(self.device)
        else:
            criterion = nn.CrossEntropyLoss().cuda()
        proto = []
        proto_grad = []

        for i in self.current_class:
            images = self.train_dataset.get_image_class(i)
            class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
            dis = class_mean - feature_extractor_output
            dis = np.linalg.norm(dis, axis=1)
            pro_index = np.argmin(dis)
            proto.append(images[pro_index])

        for i in range(len(proto)):
            self.model.eval()
            data = proto[i]
            label = self.current_class[i]
            if self.dataset == 'mnist':
                data = Image.fromarray(data.numpy())
            else:
                data = Image.fromarray(data)
            label_np = label
            
            data, label = tt(data), torch.Tensor([label]).long()
            if isinstance(self.device, int):
                data, label = data.cuda(self.device), label.cuda(self.device)
            else:
                data, label = data.cuda(), label.cuda()
            data = data.unsqueeze(0).requires_grad_(True)
            target = get_one_hot(label, self.numclass, self.device)

            opt = optim.SGD([data, ], lr=self.learning_rate / 10, weight_decay=0.00001)
            proto_model = copy.deepcopy(self.model)
            proto_model = model_to_device(proto_model, False, self.device)

            for ep in range(iters):
                outputs = proto_model(data)
                loss_cls = F.binary_cross_entropy_with_logits(outputs, target)
                opt.zero_grad()
                loss_cls.backward()
                opt.step()

            self.encode_model = model_to_device(self.encode_model, False, self.device)
            if isinstance(self.device, int):
                data = data.detach().clone().to(self.device).requires_grad_(False)
            else:
                data = data.detach().clone().cuda().requires_grad_(False)
            outputs = self.encode_model(data)
            loss_cls = criterion(outputs, label)
            dy_dx = torch.autograd.grad(loss_cls, self.encode_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append(original_dy_dx)

        return proto_grad
    
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