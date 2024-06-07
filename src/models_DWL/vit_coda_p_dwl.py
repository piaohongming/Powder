import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from models_DWL.vision_transformer_dwl import VisionTransformer
import numpy as np
import math

DEBUG_METRICS=True

class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__()
        self.task_count_f = 0
        print(" in DualPrompt prompt")

        
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # init frequency table
        for e in self.e_layers:
            setattr(self, f'freq_curr_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))
            setattr(self, f'freq_past_{e}',torch.nn.Parameter(torch.zeros(self.e_pool_size,), requires_grad=False))

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = prompt_param[2]
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

    def process_frequency(self):
        self.task_count_f += 1
        if not self.task_id_bootstrap:
            for e in self.e_layers:
                f_ = getattr(self, f'freq_curr_{e}')
                f_ = f_ / torch.sum(f_)
                setattr(self, f'freq_past_{e}',torch.nn.Parameter(f_, requires_grad=False))


    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:

                # prompting
                if self.task_id_bootstrap:
                    loss = 1.0 - cos_sim[:,task_id].sum()  # the cosine similarity is always le 1
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    if self.task_count_f > 0:
                        f_ = getattr(self, f'freq_past_{l}')
                        f_tensor = f_.expand(B,-1)
                        # cos_sim_scaled = 1.0 - (f_tensor * (1.0 - cos_sim))
                        cos_sim_scaled = cos_sim
                    else:
                        cos_sim_scaled = cos_sim
                    top_k = torch.topk(cos_sim_scaled, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = 1.0 - cos_sim[:,k_idx].sum()  # the cosine similarity is always le 1
                    P_ = p[k_idx][:,0]

                    # update frequency
                    f_ = getattr(self, f'freq_curr_{l}')
                    f_to_add = torch.bincount(k_idx.flatten().detach(),minlength=self.e_pool_size)
                    f_ += f_to_add
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices

                P_ = p[k_idx][:,0]
                
            # select prompts
            i = int(self.e_p_length/2) # prefix tuning 
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block


class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)
        print(" in L2P prompt")


    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]


class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        print(" in CODA prompt")
        super().__init__()
        self.task_id = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        self.next_task_locs= None

        # e prompt init
        if DEBUG_METRICS: self.metrics = {'attention':{},'keys':{}}
        for e in self.e_layers:
            e_l = self.e_p_length
            if self.ortho_mu == -1:
                p = tensor_prompt(self.e_pool_size, e_l, emb_d)
                k = tensor_prompt(self.e_pool_size, self.key_d)
                a = tensor_prompt(self.e_pool_size, self.key_d)
            else:
                # 4 = ablate p, 5 = ablate a, 6 = ablate k
                if self.ortho_mu == 4:
                    p = tensor_prompt(self.e_pool_size, e_l, emb_d, ortho=False)
                else:
                    p = tensor_prompt(self.e_pool_size, e_l, emb_d, ortho=True)
                if self.ortho_mu == 5:
                    a = tensor_prompt(self.e_pool_size, self.key_d, ortho=False)
                else:
                    a = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
                if self.ortho_mu == 6:
                    k = tensor_prompt(self.e_pool_size, self.key_d, ortho=False)
                else:
                    k = tensor_prompt(self.e_pool_size, self.key_d, ortho=True)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

            if DEBUG_METRICS:
                self.metrics['keys'][e] = torch.zeros((self.e_pool_size,))

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = prompt_param[0]
        self.e_p_length = prompt_param[1]

        # prompt locations
        if prompt_param[2] == 0:
            self.e_layers = [0,1,2,3,4]
        # single
        elif prompt_param[2] == 1:
            self.e_layers = [0]
        elif prompt_param[2] == 2:
            self.e_layers = [1]
        elif prompt_param[2] == 3:
            self.e_layers = [2]
        elif prompt_param[2] == 4:
            self.e_layers = [3]
        elif prompt_param[2] == 5:
            self.e_layers = [4]
        # double
        elif prompt_param[2] == 6:
            self.e_layers = [0,1]
        elif prompt_param[2] == 7:
            self.e_layers = [1,2]
        elif prompt_param[2] == 8:
            self.e_layers = [2,3]
        elif prompt_param[2] == 9:
            self.e_layers = [3,4]
        # triple
        elif prompt_param[2] == 10:
            self.e_layers = [0,1,2]
        elif prompt_param[2] == 11:
            self.e_layers = [1,2,3]
        elif prompt_param[2] == 12:
            self.e_layers = [2,3,4]
        else:
            print("error")

        # location of ortho penalty
        self.ortho_mu = prompt_param[3]
        print("ortho_mu ", self.ortho_mu)

        # ablations
        self.attention = True 
        self.attention_softmax = True 
        self.expand_and_freeze = True
        if prompt_param[4] > 0:
            if prompt_param[4] == 1:
                self.attention = False
                self.attention_softmax = False
            elif prompt_param[4] == 2:
                self.attention_softmax = False
            elif prompt_param[4] == 3:
                self.expand_and_freeze = False
                self.attention_softmax = False
        
    def process_frequency(self, next_task_locs = None):
        
        self.task_count_f += 1
        
        # print("prceoss freq changed task_count to ", self.task_count_f)
    def next_task_locs_f(self, next_task_locs = None):
        if next_task_locs:
            self.next_task_locs = next_task_locs


    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape


            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_id * pt)
            f = int((self.task_id + 1) * pt)
            
            # freeze/control past tasks
            if self.expand_and_freeze:
                # print("in expand and freeze")
                # print(self.task_count_f, pt , s, f )
                # print( " train ", train)
                
                if train:
                    # print(" ***************** \n \n ")

                    # print(" in forward self.task_count_f: ", self.task_count_f)
                    # print(" ***************** \n \n ")
                    '''
                    if self.next_task_locs:
                        K_next = K[self.next_task_locs[0]: self.next_task_locs[1]].detach().clone()
                        A_next = A[self.next_task_locs[0]: self.next_task_locs[1]].detach().clone()
                        p_next = p[self.next_task_locs[0]: self.next_task_locs[1]].detach().clone()
                    '''

                    if self.task_id > 0: 
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                        A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                        '''
                        if self.next_task_locs:
                            K = torch.cat((K,K_next), dim=0)
                            A = torch.cat((A,A_next), dim=0)
                            p = torch.cat((p,p_next), dim=0)
                        '''
                        
                    else:
                        K = K[s:f]
                        A = A[s:f]
                        p = p[s:f]
                else:
                    K = K[0:f]
                    A = A[0:f]
                    p = p[0:f]

            if self.attention:
                ##########
                # with attention and cosine sim
                ##########
                # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
                if self.attention_softmax:
                    a_querry = torch.einsum('bd,kd->bkd', x_querry, nn.functional.softmax(A,dim=1))
                else:
                    a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                # aq_k = nn.functional.softmax(aq_k_p,dim=1)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,kld->bld', aq_k, p)
            else:
                ##########
                # cosine sim
                ##########
                # # (b x 1 x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1)
                aq_k = torch.einsum('bd,kd->bk', q, n_K)
                # aq_k = nn.functional.softmax(aq_k_p,dim=1)
                # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
                P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
            # 4 = ablate p, 5 = ablate a, 6 = ablate k
            if train and self.ortho_mu > 0:
                '''
                K = getattr(self,f'e_k_{l}')
                A = getattr(self,f'e_a_{l}')
                p = getattr(self,f'e_p_{l}')
                # print( f"in train : {self.task_count_f}, {train},  {K.shape}")
                if self.task_count_f > 0:
                    
                    K = torch.cat((K[:s].detach().clone(),K[s:]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:]), dim=0)
                else:
                    K = K[s:]
                    A = A[s:]
                    p = p[s:]
                '''
               
                if self.ortho_mu == 1:
                    loss = ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = ortho_penalty(K)
                    loss += ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = ortho_penalty(K)
                    loss += ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += ortho_penalty(A)
                    loss += ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    # print("using all ortho penalty")

                    loss = ortho_penalty(K)
                    loss += ortho_penalty(A)
                    loss += ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None
            P_ = None

        # return
        if train:
            return p_return, loss, x_block, P_
        else:
            return p_return, 0, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda('cuda:2'))**2).mean() * 1e-6

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

class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None, task_size=10):
        super(ResNetZoo, self).__init__()

        # get last layer
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.task_size = task_size

        # get feature encoder
        if mode == 0:
            if pt:
                print("++++++++++++++++++ in feature+++++++++++++++++++++")
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0
                                          )
                from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
                load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)
                if prompt_flag:
                    print(" freezing original model")
                    for n,p  in zoo_model.named_parameters():
                        if not "prompt" in n:
                            print(f"freezing {n}")
                            p.requires_grad = False

            # classifier
            self.fc = nn.Linear(768, num_classes)

        # create prompting module
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(768, task_size, prompt_param)

        elif self.prompt_flag == 'codap':
            self.prompt = CodaPrompt(768, task_size, prompt_param)
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, task_size, prompt_param)

        else:
            self.prompt = None
        
        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model

    def Incremental_learning(self, numclasses):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features

        self.fc = nn.Linear(in_feature, numclasses, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias
        self.task_id = (numclasses / self.task_size) - 1
        self.prompt.task_id = self.task_id

        

    def forward(self, x, pen=False, train=False):
        #print("hhh")
        if self.prompt is not None:
            with torch.no_grad():
                q, _, _ = self.feat(x)
                q = q[:,0,:]
            #print("hhh")  
            out, prompt_loss, prompt_client = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
            out = out[:,0,:]
            #print("hhh")
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]
            
        out = out.view(out.size(0), -1)
        pre_logits = out # for fedmoon
        if not pen:
            out = self.fc(out)
        if self.prompt is not None and train:
            return out, prompt_loss, pre_logits, prompt_client
        else:
            return out

def vit_pt_imnet_dwl(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, task_size=10):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param, task_size=task_size)