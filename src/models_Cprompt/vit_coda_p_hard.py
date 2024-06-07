import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models_Cprompt.vision_transformer import VisionTransformer
import numpy as np
import math
import copy
from models_Cprompt.vit_coda_p import DualPrompt, L2P, CodaPrompt, CodaPrompt_weight, CodaPrompt_2d_v2
from torch.utils.data import DataLoader

DEBUG_METRICS=True

class Linear_mine(nn.Module):
    def __init__(self, in_dim, out_dim, args=None):
        super(Linear_mine, self).__init__() 
        self.args = args
        self.task_class_num = self.args.class_per_task
        self.global_class_min_output = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.not_trained_task_id = None
        fc_init = nn.Linear(in_dim, self.task_class_num)
        weight_init = fc_init.weight.data
        bias_init = fc_init.bias.data
        self.fc = nn.Linear(in_dim, out_dim)
        
        for i in range(int(out_dim//self.task_class_num)):
            self.fc.weight.data[i*self.task_class_num:(i+1)*self.task_class_num] = weight_init
            self.fc.bias.data[i*self.task_class_num:(i+1)*self.task_class_num] = bias_init
        

        '''
        self.fc = nn.Linear(in_dim, out_dim)

        '''
        #self.fc_ova = nn.Linear(in_dim, out_dim)
        #self.fc_ova = nn.Linear(in_dim, out_dim * 2)
        #self.fc_ova = nn.Linear(in_dim, 50)
    
    def process_frequency(self, task_id, class_distribution):
        slice_before = []
        for i in range(task_id + 1):
            for j in range(10):
               slice_before.extend(class_distribution[j][i]) 
        slice = [slice_before.index(i) for i in slice_before]
        weight = self.fc.weight.data.clone()
        #print(weight.size())
        #bias = self.fc.bias.data.clone()
        weight = weight[slice_before]
        #bias = bias[slice_before]
        weight = weight.view((task_id + 1) * 10, 4, -1) #TODO
        #bias = bias.view((task_id + 1) * 10, -1)
        #print(weight.size())
        weight = self.gram_schmidt(weight, task_id)
        #print(weight.size())
        #print(bias.size())
        #bias = self.gram_schmidt(bias, task_id)
        #print(weight[0][1])
        weight = weight.reshape((task_id + 1) * 40, -1)
        #print(weight[1])
        #bias = bias.transpose(0, 1).view(200)
        self.fc.weight.data[slice_before] = weight[slice]
        #self.bias.weight.data[slice_before] = bias[slice]

    def gram_schmidt(self, vv, task_id):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = (len(vv.shape) >= 3)
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        s = int(task_id * 10)
        f = int((task_id + 1) * 10)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
            
        return uu


    def forward(self, input, class_min_output, class_max_output):
        weight = self.fc.weight
        bias = self.fc.bias
        
        slice_before = class_min_output + class_max_output
        slice = [slice_before.index(i) for i in range(self.out_dim)]
        weight = torch.cat((weight[class_min_output, :].detach().clone(), weight[class_max_output, :]), dim=0)[slice, :]
        bias = torch.cat((bias[class_min_output].detach().clone(), bias[class_max_output]), dim=0)[slice]
        
        return F.linear(input, weight, bias)
    
    def forward_for_ova(self, prompt_proto, client_learned_task_id=None):
        output = self.fc_ova(prompt_proto)
        #print(output.size())
        #print(output[0])
        '''
        client_unlearned_task_id = []
        for i in range(50):
            if i not in client_learned_task_id:
                client_unlearned_task_id.append(i)
        '''
        #print(self.not_trained_task_id)
        min_list = []
        for i in self.global_class_min_output:
            min_list.append(int(i // 4))
        output[:,list(set(min_list))] = -float('inf')
        #output = output.view(-1, 2, self.out_dim)
        #output = F.log_softmax(output, dim=1)[:, 0, :]
        return output




def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p        

class ResNetZoo_hard(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None, task_size=10, device='cuda:0', local_clients=10, num_clients=10, class_distribution=None, tasks_global=3, class_distribution_real=None, class_distribution_proportion=None, class_distribution_client_di=None, params=None, args=None):
        super(ResNetZoo_hard, self).__init__()

        # get last layer
        self.params = params #搜出来的参数
        self.args = args
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.numclass = num_classes
        self.total_class_list = list(range(self.numclass))
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.task_size = task_size
        self.client_index = -1
        self.class_distribution = class_distribution
        self.class_distribution_real = class_distribution_real
        self.class_distribution_proportion = class_distribution_proportion
        self.class_distribution_client_di = class_distribution_client_di
        self.client_class_min_output = []
        self.client_class_max_output = []
        self.global_class_max_output_previous = []
        self.client_class_min_output_not_contain_previous = []
        self.global_class_min_output_contain_previous = []
        self.global_class_min_output = []
        self.global_class_max_output = []
        self.ep_g = 0
        self.tasks_global = tasks_global
        self.learned_classes = []
        self.unlearned_classes = []
        self.device = device
        self.num_clients = num_clients
        self.current_class = []
        #self.initial_promptchoosing = {}

        # get feature encoder
        if mode == 0:
            if pt:
                print("++++++++++++++++++ in feature+++++++++++++++++++++")
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0, device=device, args=self.args
                                          )
                from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
                '''
                if prompt_flag:
                    print(" freezing original model")
                    for n,p  in zoo_model.named_parameters():
                        if not "prompt" in n:
                            print(f"freezing {n}")
                            p.requires_grad = False
                '''

            # classifier
            #self.fc = nn.Linear(768, num_classes)
            self.fc = Linear_mine(768, num_classes, args=self.args)
            self.criterion_fn = nn.CrossEntropyLoss(reduction='none').cuda(self.device)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, task_size, prompt_param, device=device, args=self.args)

        elif self.prompt_flag == 'codap' or self.prompt_flag == 'cprompt':
            self.prompt = CodaPrompt(768, task_size, prompt_param, device=device, clients_local=local_clients, num_clients=num_clients, args=self.args)
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, task_size, prompt_param, device=device, args=self.args)
        elif self.prompt_flag == 'codap_weight' or self.prompt_flag == 'cprompt_weight':
            self.prompt = CodaPrompt_weight(768, task_size, prompt_param, device=device, clients_local=local_clients, num_clients=num_clients, args=self.args)
        elif self.prompt_flag == 'codap_2d_v2':
            self.prompt = CodaPrompt_2d_v2(768, task_size, prompt_param, device=device, clients_local=local_clients, num_clients=num_clients, args=self.args)
        
        elif self.prompt_flag == 'l2p_weight':
            pass
        elif self.prompt_flag == 'dual_weight':
            pass
        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model
    
    
    def calculate_prompt_choosing(self, train_dataset, c, t, trained_task_id, current_trained_task_id, finished_task):
        with torch.no_grad():
            indices =[]
            for i in current_trained_task_id:
                indices.append(trained_task_id.index(i))
            #print(indices)
            choosing_class = {}
            classes = self.class_distribution[c][t]
            classes_real = self.class_distribution_real[c][t]
            classes_proportion = self.class_distribution_proportion[c][t]
            if self.class_distribution_client_di is not None:
                class_distribution_client_di = self.class_distribution_client_di[c][t]
            else:
                class_distribution_client_di = None
            mean_aqk_task = None
            for i in range(len(classes)):
                train_dataset.getTrainData([classes[i]], [], [], c, classes_real=[classes_real[i]], classes_proportion=classes_proportion, class_distribution_client_di=class_distribution_client_di)
                train_loader = DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=self.args.batch_size,
                                    num_workers=8,
                                    pin_memory=True)
                mean_aqk_class = None
                for step, (indexs, images, target) in enumerate(train_loader):
                    if isinstance(self.device, int):
                        images, target = images.cuda(self.device), target.cuda(self.device)
                    else:
                        images, target = images.cuda(), target.cuda()
                    
                    with torch.no_grad():
                        q, _, _, q_map = self.feat(images)
                        q = q[:,0,:]
                    mean_aqk_list = self.feat.get_aqk(images, prompt=self.prompt, client_index=c, q=q, task_id=t, trained_task_id = trained_task_id, finished_task=finished_task).unsqueeze(0)
                    #print(mean_aqk_list.size())
                    mean_aqk_list = mean_aqk_list.reshape(mean_aqk_list.shape[0], mean_aqk_list.shape[1], len(trained_task_id), -1)
                    mean_aqk_list = mean_aqk_list[:, :, indices, :]
                    mean_aqk_list = mean_aqk_list.reshape(mean_aqk_list.shape[0], mean_aqk_list.shape[1], -1)
                    #print(mean_aqk_list.size())
                    if mean_aqk_class is None:
                        mean_aqk_class = mean_aqk_list
                    else:
                        mean_aqk_class = torch.cat((mean_aqk_class, mean_aqk_list), dim=0)
                mean_aqk_class = torch.mean(mean_aqk_class, dim=0)
                choosing_class[classes[i]] = mean_aqk_class
                if mean_aqk_task is None:
                    mean_aqk_task = mean_aqk_class.unsqueeze(0)
                else:
                    mean_aqk_task = torch.cat((mean_aqk_task, mean_aqk_class.unsqueeze(0)), dim=0)
        return torch.mean(mean_aqk_task, dim=0), choosing_class   
        

    
    def updateweight_with_promptchoosing(self, clients_index, clients_index_push, old_client_0, train_dataset, new_task, task_id, models, global_trained_task_id, choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, args, ep_g):
        #print(class_real)
        trained_task_id_previous = copy.deepcopy(global_trained_task_id)
        trained_task_id_current = copy.deepcopy(global_trained_task_id)
        
        if new_task:
            if task_id > 0:
                for c in clients_index:
                    if c in old_client_0:
                        if "full" not in self.args.method:
                            global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                        else:
                            if models[c].real_task_id == 0:
                                global_task_id = c
                            else:
                                global_task_id = models[c].real_task_id + 49
                        trained_task_id_previous = sorted(list(set(trained_task_id_previous) - set([global_task_id])))
                    else:
                        if "full" not in self.args.method:
                            global_task_id = task_id * self.prompt.num_clients + c
                        else:
                            if task_id == 0:
                                global_task_id = c
                            else:
                                global_task_id = task_id + 49
                        trained_task_id_current = sorted(list(trained_task_id_current + [global_task_id]))
            else:
                for c in clients_index:
                    if c in old_client_0:
                        if "full" not in self.args.method:
                            global_task_id = 0 * self.prompt.num_clients + c
                        else:
                            global_task_id = c
                            
                        trained_task_id_previous = sorted(list(set(trained_task_id_previous) - set([global_task_id])))
                    else:
                        if "full" not in self.args.method:
                            global_task_id = 0 * self.prompt.num_clients + c
                        else:
                            global_task_id = c
                        trained_task_id_current = sorted(list(trained_task_id_current + [global_task_id]))
        
        new_task_id = []
        

        for c in clients_index:
            if new_task:
                #

                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    if "full" not in self.args.method:
                        global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    else:
                        if models[c].real_task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = models[c].real_task_id + 49
                    finished_task[global_task_id] = trained_task_id_current
                    #finished_task_forchoosing[global_task_id] = trained_task_id_current
                    new_task_id.append(global_task_id)
               
                else:
                    if task_id > 0:
                        previous_client_id = c
                        previous_task_id = models[c].real_task_id
                        if "full" not in self.args.method:
                            previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                        else:
                            if models[c].real_task_id == 0:
                                previous_global_task_id = c
                            else:
                                previous_global_task_id = models[c].real_task_id + 49
                        finished_task[previous_global_task_id] = trained_task_id_previous
                        #finished_task_forchoosing[previous_global_task_id] = global_trained_task_id
                        

                    current_client_id = c
                    current_task_id = task_id
                    if "full" not in self.args.method:
                        global_task_id = task_id * self.prompt.num_clients + c
                    else:
                        if task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = task_id + 49
                    finished_task[global_task_id] = trained_task_id_current
                    #finished_task_forchoosing[global_task_id] = trained_task_id_current
                    new_task_id.append(global_task_id)
                    
            else:
                #    
                current_client_id = c
                current_task_id = models[c].real_task_id
                if "full" not in self.args.method:
                    global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                else:
                    if models[c].real_task_id == 0:
                        global_task_id = c
                    else:
                        global_task_id = models[c].real_task_id + 49
                finished_task[global_task_id] = global_trained_task_id
                #finished_task_forchoosing[global_task_id] = global_trained_task_id
                new_task_id.append(global_task_id)

        #print(trained_task_id_current)
        
        for c in clients_index:
            if new_task:
                #

                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    if "full" not in self.args.method:
                        global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    else:
                        if models[c].real_task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = models[c].real_task_id + 49
                    if c in clients_index_push:
                        choosing_, choosing_class_ = self.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task)
                    else:
                        choosing_, choosing_class_ = models[c].model.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task)
                    choosing[global_task_id] = choosing_.detach().cpu()
                    if "full" not in self.args.method and "extension" not in self.args.method:
                        for cl in self.class_distribution[c][models[c].real_task_id]:
                            choosing_class[cl] = choosing_class_[cl].detach().cpu()
                            finished_class[cl] = trained_task_id_current
                    else:
                        choosing_class = {}
                        finished_class = {}
                else:
                    if task_id > 0:
                        previous_client_id = c
                        previous_task_id = models[c].real_task_id
                        if "full" not in self.args.method:
                            previous_global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                        else:
                            if models[c].real_task_id == 0:
                                previous_global_task_id = c
                            else:
                                previous_global_task_id = models[c].real_task_id + 49
                        
                        if c in clients_index_push:
                            previous_choosing_, previous_choosing_class_ = self.calculate_prompt_choosing(train_dataset, previous_client_id, previous_task_id, global_trained_task_id, trained_task_id_previous, finished_task=finished_task)
                        else:
                            previous_choosing_, previous_choosing_class_ = models[c].model.calculate_prompt_choosing(train_dataset, previous_client_id, previous_task_id, global_trained_task_id, trained_task_id_previous, finished_task=finished_task)
                        choosing[previous_global_task_id] = previous_choosing_.detach().cpu()
                        if "full" not in self.args.method and "extension" not in self.args.method:
                            for cl in self.class_distribution[c][models[c].real_task_id]:
                                choosing_class[cl] = previous_choosing_class_[cl].detach().cpu()
                                finished_class[cl] = trained_task_id_previous
                        else:
                            choosing_class = {}
                            finished_class = {}
                        

                    current_client_id = c
                    current_task_id = task_id
                    if "full" not in self.args.method:
                        global_task_id = task_id * self.prompt.num_clients + c
                    else:
                        if task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = task_id + 49
                    if c in clients_index_push:
                        choosing_, choosing_class_ = self.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task)
                    else:
                        choosing_, choosing_class_ = models[c].model.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, trained_task_id_current, trained_task_id_current, finished_task=finished_task)
                    choosing[global_task_id] = choosing_.detach().cpu()
                    #print(c)
                    #print(task_id)
                    #print(choosing_class_.keys())
                    #print(self.class_distribution[c][task_id])
                    if "full" not in self.args.method and "extension" not in self.args.method:
                        for cl in self.class_distribution[c][task_id]:
                            choosing_class[cl] = choosing_class_[cl].detach().cpu()
                            finished_class[cl] = trained_task_id_current
                    else:
                        choosing_class = {}
                        finished_class = {}
            '''       
            else:
                #    
                current_client_id = c
                current_task_id = models[c].real_task_id
                if "full" not in self.args.method and "extension" not in self.args.method:
                    global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                else:
                    if models[c].real_task_id == 0:
                        global_task_id = c
                    else:
                        global_task_id = models[c].real_task_id + 19
                if c in clients_index_push:
                    choosing_, choosing_class_ = self.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, global_trained_task_id, global_trained_task_id, finished_task=finished_task)
                else:
                    choosing_, choosing_class_ = models[c].model.calculate_prompt_choosing(train_dataset, current_client_id, current_task_id, global_trained_task_id, global_trained_task_id, finished_task=finished_task)
                choosing[global_task_id] = choosing_.detach().cpu()
                if "full" not in self.args.method and "extension" not in self.args.method:
                    for cl in self.class_distribution[c][models[c].real_task_id]:
                        choosing_class[cl] = choosing_class_[cl].detach().cpu()
                        finished_class[cl] = global_trained_task_id
                else:
                    choosing_class = {}
                    finished_class = {}
            '''    
        #if True:
        if ep_g % args.tasks_global == 0:
            weight = None
            for t_1 in choosing.keys():
                weight_line = None
                for t_2 in choosing.keys():
                    if t_1 in range(10,15) and t_2 in range(10,15):
                        prompt_choosing_1 = choosing[t_1]
                        prompt_choosing_2 = choosing[t_2]
                        #print(prompt_choosing_1.size())
                        #print(prompt_choosing_2.size())
                        finished_task_1 = finished_task[t_1]
                        finished_task_2 = finished_task[t_2]
                        print(finished_task_1)
                        print(finished_task_2)
                            
                        prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                        prompt_choosing_1 = prompt_choosing_1[:, len(finished_task_1)-5:, :]
                        prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                        #print(prompt_choosing_1)
                        #print(prompt_choosing_2)
                    
                        prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                        prompt_choosing_2 = prompt_choosing_2[:, len(finished_task_2)-5:, :]
                        prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                    else:
                        prompt_choosing_1 = choosing[t_1]
                        prompt_choosing_2 = choosing[t_2]
                        #print(prompt_choosing_1.size())
                        #print(prompt_choosing_2.size())
                        finished_task_1 = finished_task[t_1]
                        finished_task_2 = finished_task[t_2]
                        if len(finished_task_1) > len(finished_task_2):
                            
                            indices = []
                            for i in finished_task_2:
                                indices.append(finished_task_1.index(i))
                            #print(indices)
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                            prompt_choosing_1 = prompt_choosing_1[:, indices, :]
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                            #print(prompt_choosing_1)
                            #print(prompt_choosing_2)
                        else:
                            indices = []
                            for i in finished_task_1:
                                indices.append(finished_task_2.index(i))
                            #print(indices)
                            #print(prompt_choosing_2.size())
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                            prompt_choosing_2 = prompt_choosing_2[:, indices, :]
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                    
                    prompt_choosing_1 = nn.functional.normalize(prompt_choosing_1, dim=1)
                    prompt_choosing_2 = nn.functional.normalize(prompt_choosing_2, dim=1)
                    #print(prompt_choosing_1.size())
                    #print(prompt_choosing_2.size())
                    #print(similarity)
                    similarity = torch.einsum('bd,bd->b', prompt_choosing_1, prompt_choosing_2)
                    weight_point = torch.mean(similarity, dim=0).unsqueeze(0)
                    
                    '''
                    if weight_point > self.params['same_task_threshold'] and t_2 > t_1:
                        global_task_id_real[t_2] = global_task_id_real[t_1]
                    '''

                    
                    weight_point = weight_point**self.params['task_index']
                    
                    '''
                    weight_point = -(weight_point-self.params['zero_task_threshold'])*(weight_point-2+self.params['zero_task_threshold'])
                    if weight_point < torch.zeros(1):
                        weight_point = torch.zeros(1)
                    '''
                    
                    if weight_line is None:
                        weight_line = weight_point
                    else:
                        weight_line = torch.cat((weight_line, weight_point), dim=0)
                
                #topk_for_task = self.params['topk_for_task']
                topk_for_task = len(trained_task_id_current)
                if topk_for_task > weight_line.shape[0]:
                    topk_for_task = weight_line.shape[0]
                _, idx = weight_line.topk(topk_for_task)
                #print("task id:")
                #print(t_1)
                #print(list(choosing.keys())[idx[0]])
                #print(list(choosing.keys())[idx[1]])
                #print(list(choosing.keys()))
                #print(weight_line)
                line_choose = torch.ones(weight_line.shape)
                line_choose[idx] = 0
                weight_line = weight_line.masked_fill(line_choose.bool(), 0)
                #print(weight_line)
                weight_line = weight_line / weight_line.sum()
                if weight is None:
                    weight = weight_line.unsqueeze(0)
                else:
                    weight = torch.cat((weight, weight_line.unsqueeze(0)), dim=0)
            if "full" not in self.args.method:
                fc_weight = None
                for c_1 in choosing_class.keys():
                    fc_weight_line = None
                    for c_2 in choosing_class.keys():
                        prompt_choosing_1 = choosing_class[c_1]
                        prompt_choosing_2 = choosing_class[c_2]
                        finished_task_1 = finished_class[c_1]
                        finished_task_2 = finished_class[c_2]
                        #print(finished_task_1)
                        #print(finished_task_2)
                        if len(finished_task_1) > len(finished_task_2):
                            indices = []
                            for i in finished_task_2:
                                indices.append(finished_task_1.index(i))
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], len(finished_task_1), -1)
                            prompt_choosing_1 = prompt_choosing_1[:, indices, :]
                            prompt_choosing_1 = prompt_choosing_1.reshape(prompt_choosing_1.shape[0], -1)
                        else:
                            indices = []
                            for i in finished_task_1:
                                indices.append(finished_task_2.index(i))
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], len(finished_task_2), -1)
                            prompt_choosing_2 = prompt_choosing_2[:, indices, :]
                            prompt_choosing_2 = prompt_choosing_2.reshape(prompt_choosing_2.shape[0], -1)
                        
                        prompt_choosing_1 = nn.functional.normalize(prompt_choosing_1, dim=1)
                        prompt_choosing_2 = nn.functional.normalize(prompt_choosing_2, dim=1)
                        similarity = torch.einsum('bd,bd->b', prompt_choosing_1, prompt_choosing_2)
                        
                        if int(c_1 // self.args.class_per_task) == int(c_2 // self.args.class_per_task) and c_1 != c_2: 
                        #if int(c_1 // 20) == int(c_2 // 20):
                            fc_weight_point = torch.zeros(1)
                        else:
                            fc_weight_point = torch.mean(similarity, dim=0).unsqueeze(0)

                            
                            
                            '''
                            if fc_weight_point < self.params['same_class_threshold'] and c_2 != c_1:
                                weight_point = torch.zeros(1)
                            '''
                            
                            
                            fc_weight_point = fc_weight_point**self.params['class_index']
                            
                            '''
                            fc_weight_point = -(fc_weight_point-self.params['zero_class_threshold'])*(fc_weight_point-2+self.params['zero_class_threshold'])
                            if fc_weight_point < torch.zeros(1):
                                fc_weight_point = torch.zeros(1)
                            '''
                            
                        if fc_weight_line is None:
                            fc_weight_line = fc_weight_point
                        else:
                            fc_weight_line = torch.cat((fc_weight_line, fc_weight_point), dim=0)
                    
                    _, idx = fc_weight_line.topk(self.params['topk_for_class'])
                    #print("class id:")
                    #print(c_1)
                    #print(list(choosing_class.keys())[idx[0]])
                    #print(list(choosing_class.keys())[idx[1]])
                    line_choose = torch.ones(fc_weight_line.shape)
                    line_choose[idx] = 0
                    fc_weight_line = fc_weight_line.masked_fill(line_choose.bool(), 0)
                    
                    #print(fc_weight_line)
                    fc_weight_line = fc_weight_line / fc_weight_line.sum()
                    if fc_weight is None:
                        fc_weight = fc_weight_line.unsqueeze(0)
                    else:
                        fc_weight = torch.cat((fc_weight, fc_weight_line.unsqueeze(0)), dim=0)
            #print(weight.size())
            #print(self.prompt.weight[list(choosing.keys())][:, list(choosing.keys())].size())
            #weight_in_prompt = torch.zeros((50, 50), device=self.device)
            #fc_weight_in_prompt = torch.zeros((10, 10), device=self.device)
            #print(fc_weight)
            for i in range(len(list(choosing.keys()))):
                self.prompt.weight[list(choosing.keys()), list(choosing.keys())[i]] = torch.tensor(weight[:, i], device=self.device)
            #self.prompt.weight[list(choosing.keys()), :][:, list(choosing.keys())] = weight
            self.prompt.weight = torch.tensor(self.prompt.weight, device=self.device)
            self.prompt.weight_c[new_task_id] = self.prompt.weight.clone()[new_task_id]
            if "full" not in self.args.method:
                for i in range(len(list(choosing_class.keys()))):
                    self.prompt.fc_weight[list(choosing_class.keys()), list(choosing_class.keys())[i]] = torch.tensor(fc_weight[:, i], device=self.device)
                self.prompt.fc_weight = torch.tensor(self.prompt.fc_weight, device=self.device)
                #print(self.prompt.fc_weight[list(choosing_class.keys()), list(choosing_class.keys())])
        #print(self.prompt.fc_weight)
            
            for c in clients_index:
                if c in old_client_0:
                    current_client_id = c
                    current_task_id = models[c].real_task_id
                    if "full" not in self.args.method:
                        global_task_id = models[c].real_task_id * self.prompt.num_clients + c
                    else:
                        if models[c].real_task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = models[c].real_task_id + 49
                    _, idx = self.prompt.weight[global_task_id].topk(self.params['topk_for_task_selection'])
                    finished_task_forchoosing[global_task_id] = idx
                else:
                    current_client_id = c
                    current_task_id = task_id
                    if "full" not in self.args.method:
                        global_task_id = task_id * self.prompt.num_clients + c
                    else:
                        if task_id == 0:
                            global_task_id = c
                        else:
                            global_task_id = task_id + 49
                    _, idx = self.prompt.weight[global_task_id].topk(self.params['topk_for_task_selection'])
                    finished_task_forchoosing[global_task_id] = idx

        print(class_real)
        return choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real


    def Incremental_learning(self, task_id):
        
        self.task_id = task_id
        self.prompt.task_id = self.task_id
        if "noortho" in self.args.method:
            pass
        else:
            self.prompt.process_frequency()

    def set_global_class_min_output(self, global_class_output, global_class_output_now):
        self.global_class_min_output = []
        self.global_class_min_output_contain_previous = []
        self.global_class_max_output_previous = self.global_class_max_output
        self.global_class_max_output = global_class_output
         
        for i in range(self.numclass):
            if i in global_class_output:
                continue
            else:
                self.global_class_min_output.append(i) 
        for i in range(self.numclass):
            if i in global_class_output_now:
                continue
            else:
                self.global_class_min_output_contain_previous.append(i)
        self.fc.global_class_min_output = self.global_class_min_output

    def set_client_class_min_output(self):
        client_class_output = self.current_class
        self.client_class_min_output = []
        self.client_class_min_output_not_contain_previous = []
        self.unlearned_classes = []
        self.client_class_max_output = client_class_output
        for i in range(self.numclass):
            if i in client_class_output:
               continue
            else:
                self.client_class_min_output.append(i)
        for i in range(self.numclass):
            if (i in client_class_output) or (i in self.global_class_max_output_previous):
               continue
            else:
                self.client_class_min_output_not_contain_previous.append(i)
        for i in range(self.numclass):
            if i in self.learned_classes:
               continue
            else:
                self.unlearned_classes.append(i)

           

    def forward(self, x, pen=False, train=False, aq_k=None, device=0, ova='none', client_learned_task_id=None, labels=None):
        
        #torch.autograd.set_detect_anomaly(True)
        if self.prompt is not None:
            
            with torch.no_grad():
                q, _, _, q_map = self.feat(x)
                q = q[:,0,:]
            
            if "classincremental" in self.args.method:
                if train:  
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map, out_divide = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k, ep_g=self.ep_g, client_index=self.prompt.client_index)
                else:
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map, out_divide = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k)
            else:
                if train:  
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k, ep_g=self.ep_g, client_index=self.prompt.client_index)
                else:
                    out, prompt_loss, prompt_client, indices_taskchoosing, mean_aqk_list, out_map = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k)
            #print(indices_taskchoosing)
            #out = out[:,0,:]
            if "classincremental" not in self.args.method:
                if "v2" in self.args.method:
                    out = out[:,3 * self.prompt.e_p_length,:]
                else:
                    out = out[:,0,:]
            else:
                if "v2" in self.args.method:
                    out = out[:,3 * self.prompt.e_p_length,:]
                    out_divide = out_divide[:,3 * self.prompt.e_p_length_2,:]
                else:
                    out = out[:,0,:]
                    out_divide = out_divide[:,0,:]
            
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
        
        #out, _, _ = self.feat(x)
        #out = out[:,0,:]
            
        out = out.view(out.size(0), -1)
        pre_logits = out # for fedmoon
        
        
        weight = self.prompt.fc_weight.detach().clone()
        
        #print(weight)
        #print(weight.size())
        '''
        
        weight = torch.zeros((10, 10), device=self.device)
        
        weight[0][0] = 0.5
        weight[0][1] = 0.5
        weight[1][0] = 0.5
        weight[1][1] = 0.5
        '''
        
        #weight = weight.float()


        if not pen:
            #if ova == 'ova':
                #return self.fc.forward_for_ova(pre_logits, client_learned_task_id)
            if self.client_index == -1:
                out = self.fc(out, self.global_class_min_output, self.global_class_max_output)
            else:
                out = self.fc(out, self.client_class_min_output, self.client_class_max_output)
            
            if "classincremental" in self.args.method:
                out_divide = nn.functional.normalize(out_divide, dim=1)
                task_embedding = nn.functional.normalize(self.prompt.task_embedding, dim=1)
                out_divide = torch.mm(out_divide, task_embedding.transpose(0,1))

            #print(out[:, self.client_class_max_output])
            
            
            #out = torch.einsum('bk,mk->bm', out, weight)
            
            '''
            
            out = out.reshape(out.shape[0], -1, self.fc.task_class_num)
            out = torch.einsum('bkd,km->bmd', out, weight)
            out = out.reshape(out.shape[0], -1)
            '''

   

            #print(out[:, self.client_class_max_output])
            
            #print(self.client_index)
            #print(out[:, self.client_class_max_output].size())
            #print(torch.mean(out[:, self.client_class_max_output], dim=1).size())
            #if self.client_index != -1:
                #print(self.client_class_min_output)
                #print(self.learned_classes)
                #control_loss = F.cross_entropy(torch.cat((torch.mean(out[:, self.client_class_max_output], dim=1).unsqueeze(1), torch.mean(out[:, self.client_class_min_output], dim=1).unsqueeze(1)), dim=1), torch.zeros((out.shape[0])).cuda(device).long(), reduction='none')
            #print(torch.tensor(self.client_class_max_output).cuda(device).unsqueeze(0).repeat(out.shape[0], 1).size())
            #target = torch.tensor(self.client_class_max_output).cuda(device).unsqueeze(0).repeat(out.shape[0], 1)
            #control_loss = F.binary_cross_entropy_with_logits(out.clone(), get_one_hot(target, out.shape[1], device))
            if "classincremental" in self.args.method and train:
                client_task_min_output = sorted(list(set(list(range(self.prompt.e_task_number))) - set(self.prompt.client_learned_global_task_id)))
                global_task_id = self.prompt.task_id * self.prompt.num_clients + self.prompt.client_index
                global_task_id = self.prompt.global_task_id_real[global_task_id]
                trained_task_id_removed = sorted(set(self.prompt.client_learned_global_task_id)-set([global_task_id]))
                out_divide[:,client_task_min_output] = -float('inf')
                #out_divide_2 = F.softmax(out_divide, dim=1)
                out_divide_2 = out_divide
                if len(trained_task_id_removed) > 0:
                    #logits_loss_for_divide = self.criterion_fn(out_divide, torch.tensor(global_task_id, device=self.device).repeat(out_divide.shape[0])).mean()
                    logits_loss_for_divide = 0
                    repre_loss_for_divide = (1.0 - out_divide_2[:,global_task_id].mean() \
                         + 1.0 * ((out_divide_2[:,trained_task_id_removed] * out_divide_2[:,trained_task_id_removed]).mean()) + \
                            prompt_loss + \
                                0) * 1.0
                    #repre_loss_for_divide = 1.0 - out_divide_2[:,global_task_id].mean() + out_divide_2[:,trained_task_id_removed].mean()
                    #repre_loss_for_divide = 0
                    #out_divide_2[:,global_task_id].var()
                else:

                    logits_loss_for_divide = 0
                    repre_loss_for_divide = (1.0 - out_divide_2[:,global_task_id].mean() \
                         + \
                            prompt_loss + \
                                0) * 1.0
                    #repre_loss_for_divide = 1.0 - out_divide_2[:,global_task_id].mean()
                #print("repre_loss")
                #print(out_divide_2[:,global_task_id].mean())
                #print(repre_loss_for_divide)
                #print("repre_loss_end")
            elif "classincremental" in self.args.method and not train:
                if self.client_index == -1:
                    global_task_min_output = sorted(list(set(list(range(self.prompt.e_task_number))) - set(self.prompt.trained_task_id)))
                    out_divide[:,global_task_min_output] = -float('inf')
                else:
                    client_task_min_output = sorted(list(set(list(range(self.prompt.e_task_number))) - set(self.prompt.client_learned_global_task_id)))
                    out_divide[:,client_task_min_output] = -float('inf')
                    print(out_divide[:,self.prompt.client_learned_global_task_id])
                
            control_loss = 0
            if self.client_index == -1 and not train:
                if "classincremental" in self.args.method:
                    detect_task_id = torch.max(out_divide, dim=1)[1].squeeze()
                    if len(detect_task_id.shape) == 0:
                        detect_task_id = detect_task_id.unsqueeze(0)
                    #print(out_divide)
                    if self.prompt.task_id == 1:
                        #print(detect_task_id)
                        pass
                    for i in range(out.shape[0]):
                        detect_class_list = self.class_distribution[int(detect_task_id[i] % self.args.num_clients)][int(detect_task_id[i] // self.args.num_clients)]
                        sample_class_min_output = sorted(set(self.total_class_list)-set(detect_class_list))
                        out[i,sample_class_min_output] = -float('inf')

                else:
                    out[:,self.global_class_min_output] = -float('inf')
                #print(out_ova[0])
                #out[:,self.global_class_min_output] = out[:,self.global_class_min_output]
                #print(out[0])
            elif not train:
                if "classincremental" in self.args.method:
                    detect_task_id = torch.max(out_divide, dim=1)[1].squeeze()
                    if len(detect_task_id.shape) == 0:
                        detect_task_id = detect_task_id.unsqueeze(0)
                    if self.prompt.task_id == 1:
                        #print(detect_task_id)
                        pass
                    for i in range(out.shape[0]):
                        detect_class_list = self.class_distribution[int(detect_task_id[i] % self.args.num_clients)][int(detect_task_id[i] // self.args.num_clients)]
                        sample_class_min_output = sorted(set(self.total_class_list)-set(detect_class_list))
                        out[i,sample_class_min_output] = -float('inf')
                else:
                    out[:,self.client_class_min_output] = -float('inf')
            else:
                out[:,self.client_class_min_output] = -float('inf')
                #out[:,self.client_class_min_output] = out[:,self.client_class_min_output]
                #print(self.client_class_min_output) 
        if "classincremental" in self.args.method:
            if self.prompt is not None and train:
                return out, prompt_loss, pre_logits, prompt_client, control_loss, mean_aqk_list, q_map, out_map, logits_loss_for_divide, repre_loss_for_divide
            else:
                return out
        else:
            if self.prompt is not None and train:
                return out, prompt_loss, pre_logits, prompt_client, control_loss, mean_aqk_list, q_map, out_map
            else:
                return out
        
        
    
    
    def feature_extractor(self, inputs):
        feature, _ = self.feat(inputs)
        return feature[:,0,:]
    
    def feature_extractor_withprompt(self, inputs):
        with torch.no_grad():
            q, _, _, _ = self.feat(inputs)
            q = q[:,0,:]  
        if "classincremental" in self.args.method:
            #feature, _, _, _ = self.feat(inputs, prompt=self.prompt, q=q, train=True, task_id=self.task_id, aq_k=None)
            feature, _, _, _, _, _, _ = self.feat(inputs, prompt=self.prompt, q=q, train=True, task_id=self.task_id, aq_k=None, ep_g=None, client_index=self.prompt.client_index)
        else:
            feature, _, _, _, _, _ = self.feat(inputs, prompt=self.prompt, q=q, train=True, task_id=self.task_id, aq_k=None, ep_g=None, client_index=self.prompt.client_index)
        if "v2" in self.args.method:
            feature = feature[:,3 * self.prompt.e_p_length,:]
        else:
            feature = feature[:,0,:]
        return feature
    
    def get_K_penalty(self, task):
        K_penalty = self.prompt.get_K_penalty(task)
        return K_penalty
    
    def get_A_penalty(self, task):
        A_penalty = self.prompt.get_A_penalty(task)
        return A_penalty
    
    def get_P_penalty(self, task):
        P_penalty = self.prompt.get_P_penalty(task)
        return P_penalty
    
    def getAttention(self, x, task):
        with torch.no_grad():
            q, _, _ = self.feat(x)
            q = q[:,0,:]
         
        attention = self.feat.getAttention(x, prompt=self.prompt, q=q, task=task)
        return attention
    
    def getPrompt(self, i=0):
        prompts, classes = self.prompt.getPrompt(i)
        return prompts, classes
    
    def getK(self):
        Ks, classes = self.prompt.getK()
        return Ks, classes
    
    def getA(self):
        As, classes = self.prompt.getA()
        return As, classes
            
def get_one_hot(target, num_class, device):
    if isinstance(device, int):
        one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    else:
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
    
    one_hot=one_hot.scatter(dim=1,index=target.long(),value=1.)
    return one_hot

def vit_pt_imnet_hard(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, task_size=10, device='cuda:0', local_clients = 10, num_clients=10, class_distribution=None, tasks_global=3, class_distribution_real=None, class_distribution_proportion=None, class_distribution_client_di=None, params=None, args=None):
    return ResNetZoo_hard(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param, task_size=task_size, device=device, local_clients=local_clients, num_clients=num_clients, class_distribution=class_distribution, tasks_global=tasks_global, class_distribution_real=class_distribution_real, class_distribution_proportion=class_distribution_proportion, class_distribution_client_di=class_distribution_client_di, params=params, args=args)