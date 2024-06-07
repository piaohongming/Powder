import torch
import torch.nn as nn
import copy
from models_Cprompt.vit_coda_p import DualPrompt, L2P, CodaPrompt, CodaPrompt_weight, CodaPrompt_2d_v2
import torch.nn.functional as F

class network_hard(nn.Module):

    def __init__(self, numclass, feature_extractor, class_distribution, class_distribution_real=None, class_distribution_proportion=None, args=None):
        super(network_hard, self).__init__()
        self.args = args
        if "sharedencoder" in self.args.method:
            self.feature = nn.ModuleList()
            for i in range(int(self.args.numclass/self.args.class_per_task) + 1):
                self.feature.append(copy.deepcopy(feature_extractor))
        elif "sharedprompt" in self.args.method:
            self.feature = feature_extractor
            self.prompt = nn.ParameterList()
            for i in range(int(self.args.numclass/self.args.class_per_task) + 1):
                self.prompt.append(nn.Parameter(torch.FloatTensor(8, 768).uniform_(0, 1), requires_grad=True))
        elif "sharedcodap" in self.args.method:
            self.feature = feature_extractor
            self.global_prompt = nn.Parameter(torch.FloatTensor(8, 768).uniform_(0, 1), requires_grad=True)
            self.prompt = CodaPrompt(768, self.args.task_size, self.args.prompt_param, device=self.args.device, clients_local=self.args.local_clients, num_clients=self.args.num_clients, args=self.args)
        else:
            self.feature = feature_extractor
        self.numclass = numclass
        #self.fc = nn.Linear(feature_extractor.fc.in_features, numclass, bias=True)
        self.fc = nn.Linear(768, numclass, bias=True)
        self.client_fc = nn.Linear(768, self.args.class_per_task, bias=True)
        self.aggregate_weight = torch.nn.Parameter(torch.FloatTensor(int(self.args.numclass/self.args.class_per_task), int(self.args.numclass/self.args.class_per_task)).uniform_(1, 1), requires_grad=True)
        self.class_distribution = class_distribution
        self.class_distribution_real = class_distribution_real
        self.class_distribution_proportion = class_distribution_proportion
        self.task_id = 0
        self.client_index = -1
        #self.client_class_output = []
        self.client_class_min_output = []
        self.global_class_min_output = [] #This is for evaluation of global model
        self.current_class = []
        self.trained_task_id = None
        self.learned_class = None
        self.unlearned_class = None
        self.ep_g = 0

    def Incremental_learning(self, task_id):
        self.task_id = task_id

    def set_global_class_min_output(self, global_class_output, global_class_output_now):
        self.global_class_max_output = global_class_output
        self.global_class_min_output = []
        for i in range(self.numclass):
            if i in global_class_output:
                continue
            else:
                self.global_class_min_output.append(i)

    def set_client_class_min_output(self, client_class_output):
        #self.client_class_output = client_class_output
        self.client_class_max_output = client_class_output
        self.client_class_min_output = []
        for i in range(self.numclass):
            if i in client_class_output:
               continue
            else:
                self.client_class_min_output.append(i) 
    
    def normalize_attention(self):
        with torch.no_grad():
            print("normalize the attention")
            self.aggregate_weight[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index] = self.aggregate_weight[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index] / len(self.trained_task_id)
            print("end normalize")

    def forward(self, input):
        #x = self.feature(input)
        client_global_task_id = self.task_id * self.args.num_clients + self.client_index
        if self.client_index == -1:
            x, _, _, _ = self.feature(input)
            x = x[:,0,:]
            x = self.fc(x)
        else:
            if "sharedfc" in self.args.method:
                x, _, _, _ = self.feature(input)
                feature = x[:,0,:]
                #print(x.shape)
                x = self.fc(feature)
                
                x = self.weit_weight(x, feature)

                x[:,self.client_class_min_output] = -float('inf')
            elif "sharedencoder"  in self.args.method:
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x, _, _, _ = self.feature[i](input)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x, _, _, _ = self.feature[i](input)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x, _, _, _ = self.feature[self.task_id * self.args.num_clients + self.client_index](input)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                    x = self.fc(x)
                    x[:,self.client_class_min_output] = -float('inf')
                else:
                    x, _, _, _ = self.feature[-1](input)
                    x = x[:,0,:]
                    x = self.fc(x)
                    x[:,self.client_class_min_output] = -float('inf')
            elif "sharedprompt" in self.args.method:
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x = self.feature.forward_sharedprompt(input, self.prompt, i, client_global_task_id=client_global_task_id)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x = self.feature.forward_sharedprompt(input, self.prompt, i, client_global_task_id=client_global_task_id)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x = self.feature.forward_sharedprompt(input, self.prompt, self.task_id * self.args.num_clients + self.client_index, client_global_task_id=client_global_task_id)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                    #print(x[0][0:10])
                    x = self.fc(x)
                    #print(x[0][0:10])
                    x[:,self.client_class_min_output] = -float('inf')
                else:
                    x = self.feature.forward_sharedprompt(input, self.prompt, -1, client_global_task_id=client_global_task_id)
                    x = x[:,0,:]
                    x = self.fc(x)
                    x[:,self.client_class_min_output] = -float('inf')  
            elif "sharedcodap" in self.args.method:
                with torch.no_grad():
                    q, _, _, _ = self.feature(input)
                    q = q[:,0,:]
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, i, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, i, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, self.task_id * self.args.num_clients + self.client_index, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                    x = self.fc(x)
                    x[:,self.client_class_min_output] = -float('inf')
                else:
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


    def feature_extractor(self, input):
        client_global_task_id = self.task_id * self.args.num_clients + self.client_index
        if self.client_index == -1:
            x, _, _, _ = self.feature(input)
            x = x[:,0,:]
        else:
            if "sharedfc" in self.args.method:
                x, _, _, _ = self.feature(input)
                x = x[:,0,:]
            elif "sharedencoder"  in self.args.method:
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x, _, _, _ = self.feature[i](input)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x, _, _, _ = self.feature[i](input)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x, _, _, _ = self.feature[self.task_id * self.args.num_clients + self.client_index](input)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                else:
                    x, _, _, _ = self.feature[-1](input)
                    x = x[:,0,:]
            elif "sharedprompt" in self.args.method:
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x = self.feature.forward_sharedprompt(input, self.prompt, i, client_global_task_id=client_global_task_id)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x = self.feature.forward_sharedprompt(input, self.prompt, i, client_global_task_id=client_global_task_id)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x = self.feature.forward_sharedprompt(input, self.prompt, self.task_id * self.args.num_clients + self.client_index, client_global_task_id=client_global_task_id)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                else:
                    x = self.feature.forward_sharedprompt(input, self.prompt, -1, client_global_task_id=client_global_task_id)
                    x = x[:,0,:]
            elif "sharedcodap" in self.args.method:
                with torch.no_grad():
                    q, _, _, _ = self.feature(input)
                    q = q[:,0,:]
                if "weit" in self.args.method:
                    temp = None
                    other_list = list(set(self.trained_task_id) - set([self.task_id * self.args.num_clients + self.client_index]))
                    for i in other_list:
                        if temp is None:
                            x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, i, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                            x = x[:,0,:]
                            temp = x.unsqueeze(0)
                        else:
                            x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, i, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                            x = x[:,0,:]
                            temp = torch.cat((temp, x.unsqueeze(0)), dim=0)
                    current_x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, self.task_id * self.args.num_clients + self.client_index, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                    current_x = current_x[:, 0, :]
                    aggregate_weight = self.aggregate_weight[other_list, :][:, self.task_id * self.args.num_clients + self.client_index]
                    #aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.task_id * self.args.num_clients + self.client_index]
                    aggregate_weight = F.softmax(aggregate_weight, dim=0)
                    x = torch.einsum('cbd,c->bd', temp, aggregate_weight) + current_x
                else:
                    x = self.feature.forward_sharedcodap(input, self.prompt, self.global_prompt, -1, q, train=True, task_id=self.task_id, ep_g=self.ep_g, client_index=self.prompt.client_index)
                    x = x[:,0,:]
            else:
                x, _, _, _ = self.feature(input)
                x = x[:,0,:]
        return x


    def predict(self, fea_input):
        if self.client_index == -1:
            x = self.fc(fea_input)
            x[:,self.global_class_min_output] = -float('inf')
        else:
            if "sharedfc" in self.args.method:
                x = self.fc(fea_input)
                x = self.weit_weight(x, fea_input)
                x[:,self.client_class_min_output] = -float('inf')
            elif "sharedencoder"  in self.args.method:
                x = self.fc(fea_input)
                x[:,self.client_class_min_output] = -float('inf')
            elif "sharedprompt" in self.args.method:
                x = self.fc(fea_input)
                x[:,self.client_class_min_output] = -float('inf')
            elif "sharedcodap" in self.args.method:
                x = self.fc(fea_input)
                x[:,self.client_class_min_output] = -float('inf')
            else:
                x = self.fc(fea_input)
                x[:,self.client_class_min_output] = -float('inf')
        return x
    
    def set_learned_unlearned_class(self, learned_class):
        self.learned_class = learned_class
        self.unlearned_class = sorted(list(set(list(range(self.numclass))) - set(learned_class)))

    def weit_weight(self, output, features):
        if "weit" in self.args.method:
            slice_before = self.client_class_max_output + self.client_class_min_output
            slice = [slice_before.index(i) for i in range(self.numclass)]
            output = torch.cat((output[:, self.client_class_max_output], output[:, self.client_class_min_output].detach().clone()), dim=1)
            output = output[:, slice]
            

            #output = torch.cat((output[:, self.global_class_max_output].reshape(output.shape[0], -1, 20).reshape(output.shape[0], -1), output[:, self.global_class_min_output]), dim=1)
            #aggregate_weight = self.aggregate_weight

            aggregate_weight = self.aggregate_weight.clone().fill_diagonal_(1)[self.trained_task_id, :][:, self.trained_task_id]
            #print(aggregate_weight)
            temp = output[:, self.global_class_max_output].reshape(output.shape[0], -1, self.args.class_per_task).transpose(1, 2).clone()
            output = torch.cat((torch.einsum('bkd,df->bkf', temp, aggregate_weight).reshape(output.shape[0], -1), output[:, self.global_class_min_output]), dim=1)
            slice_before = self.global_class_max_output + self.global_class_min_output
            slice = [slice_before.index(i) for i in range(self.numclass)]
            output = output[:, slice]
        
        output = output + self.client_fc(features).repeat(1, int(self.args.numclass/self.args.class_per_task)) #client_fc是全局的

        return output


class LeNet_hard(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet_hard, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())
