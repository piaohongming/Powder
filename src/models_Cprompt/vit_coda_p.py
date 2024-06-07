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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DEBUG_METRICS=True

class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device=None, args=None):
        print(" in DualPrompt prompt")
        super().__init__()
        self.args = args
        self.task_count_f = 0
        self.task_id = 0
        #self.max_classes = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.client_index = -1
        self.num_clients = args.num_clients
        #self.weight = 0
        self.device = device
        self.weight = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.weight_c = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.fc_weight = torch.eye(self.args.numclass, device=self.device)
        
        
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
        self.g_p_length = prompt_param[5]
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]

    def process_frequency(self):
        self.task_count_f += 1
        if not self.task_id_bootstrap:
            for e in self.e_layers:
                f_ = getattr(self, f'freq_curr_{e}')
                f_ = f_ / torch.sum(f_)
                setattr(self, f'freq_past_{e}',torch.nn.Parameter(f_, requires_grad=False))

    def forward(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None):

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
            #print(cos_sim.size())
            
            if train:

                # prompting
                if self.task_id_bootstrap:
                    #if self.task_id > 0:
                        #print(self.task_id)
                        #print(cos_sim.dtype)
                        #print(cos_sim.size())
                    loss = 1.0 - (cos_sim[:,self.task_id].sum()/(q.size()[0]*self.top_k))  # the cosine similarity is always le 1
                    #loss = 1.0 - cos_sim[:,self.task_id].sum()
                    P_ = p[self.task_id].expand(len(x_querry),-1,-1)
                else:
                    if self.task_id > 0:
                        f_ = getattr(self, f'freq_past_{l}')        
                        f_tensor = f_.expand(B,-1)
                        cos_sim_scaled = 1.0 - (f_tensor * (1.0 - cos_sim))
                        #cos_sim_scaled = cos_sim
                    else:
                        cos_sim_scaled = cos_sim
                    
                    top_k = torch.topk(cos_sim_scaled, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = 1.0 - (cos_sim[:,k_idx][:,0].sum()/(q.size()[0]*self.top_k)) # the cosine similarity is always le 1
                    #print(cos_sim[:,k_idx][:,0].size())
                    #print(cos_sim)
                    
                    #loss = 1.0 - cos_sim[:,self.task_id].sum()
                    #print(loss)
                    P_ = p[k_idx]
                    

                    # update frequency
                    f_ = getattr(self, f'freq_curr_{l}')
                    f_to_add = torch.bincount(k_idx.flatten().detach(),minlength=self.e_pool_size)
                    f_ += f_to_add
                
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices

                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
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
            P_ = None

        # return
        if train:
            return p_return, loss, x_block, P_, None, None
        else:
            return p_return, 0, x_block, None


class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, device=None, key_dim=768, args=None):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim, device, args)
        print(" in L2P prompt")


    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        self.e_layers = [4, 5, 6]
        

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = prompt_param[1]
        self.e_pool_size = prompt_param[0]


class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device='cuda:0', clients_local=10, num_clients = 10, args=None):
        print(" in CODA prompt")
        super().__init__()
        self.args = args
        self.task_id = 0
        self.max_classes = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        self.next_task_locs= None
        self.device = device
        self.task_count_f = 0
        self.clients_local = clients_local
        self.num_clients = num_clients
        self.client_index = -1
        self.trained_task_id = None
        self.not_trained_task_id = None
        self.client_learned_global_task_id = None
        self.ep_g = None

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            #p0 = tensor_prompt(self.e_task_number, self.e_pool_size_0, e_l, emb_d) #TODO: 50 * 5 * 8 * 768
            #k0 = tensor_prompt(self.e_task_number, self.e_pool_size_0, self.key_d)
            #a0 = tensor_prompt(self.e_task_number, self.e_pool_size_0, self.key_d)
            #p0 = self.gram_schmidt(p0)
            #k0 = self.gram_schmidt(k0)
            #a0 = self.gram_schmidt(a0)
            
            
            if "extension" in self.args.method or "extencl" in self.args.method:
                #extension版本
                p1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, e_l, emb_d, same=False, ortho=True) #TODO: 50 * 5 * 8 * 768
                k1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=False, ortho=True)
                a1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=False, ortho=True)
            else:
                #baseline版本
                p1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, e_l, emb_d, same=True) #TODO: 50 * 5 * 8 * 768
                k1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=True)
                a1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=True)
            #setattr(self, f'e_p_specific_{e}',p0)
            #setattr(self, f'e_k_specific_{e}',k0)
            #setattr(self, f'e_a_specific_{e}',a0)
            setattr(self, f'e_p_share_{e}',p1)
            setattr(self, f'e_k_share_{e}',k1)
            setattr(self, f'e_a_share_{e}',a1)

            #k_representation = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d)
            #setattr(self, f'e_k_representation_{e}',k_representation)
        #self.weight = torch.nn.Parameter(torch.FloatTensor(20, 30).uniform_(0, 1), requires_grad=True)
        #self.fc_weight = torch.nn.Parameter(torch.FloatTensor(200, 30).uniform_(0, 1), requires_grad=True)
        #self.weight = torch.nn.Parameter(torch.FloatTensor(1, 50).uniform_(0, 1).repeat(self.e_task_number, 1), requires_grad=True)
        self.weight = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.weight_c = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.fc_weight = torch.eye(self.args.numclass, device=self.device)
        #self.weight = torch.ones((10, 10), device=self.device)
        #self.fc_weight = torch.ones((200, 200), device=self.device)
        #self.weight = torch.nn.Parameter(torch.ones((self.e_task_number, self.e_task_number)), requires_grad=True)    
        #nn.init.uniform_(self.weight, a=0.0, b=1.0)
        
    
    #def get_initial_promptchoosing(self, ):

    
    
    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        #self.e_pool_size = prompt_param[0]
        #self.e_p_length = prompt_param[1]

        self.e_task_number = prompt_param[0] #TODO: 50
        self.e_pool_size_0 = prompt_param[1] #TODO: 5
        self.e_pool_size_1 = prompt_param[2] #TODO: 5
        self.e_p_length = prompt_param[3] #TODO: 8


        # prompt locations
        #self.e_layers = [10, 11, 12]
        self.e_layers = [0, 1, 2, 3, 4]
        #self.e_layers = [4, 5, 6]

        # location of ortho penalty
        self.ortho_mu = prompt_param[4] 
        print("ortho_mu ", self.ortho_mu)

        # ablations
        '''
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
        '''
        
    def process_frequency(self, next_task_locs = None):
        
        self.task_count_f += 1
        '''        
        for e in self.e_layers:
            K0 = getattr(self,f'e_k_specific_{e}')
            A0 = getattr(self,f'e_a_specific_{e}')
            P0 = getattr(self,f'e_p_specific_{e}')
            k0 = self.gram_schmidt(K0)
            a0 = self.gram_schmidt(A0)
            p0 = self.gram_schmidt(P0)
            setattr(self, f'e_p_specific_{e}',p0)
            setattr(self, f'e_k_specific_{e}',k0)
            setattr(self, f'e_a_specific_{e}',a0)
        '''
        if "extension" in self.args.method or "extencl" in self.args.method:
            pass
        else:
            #baseline得把这个加上
            for e in self.e_layers:
                K1 = getattr(self,f'e_k_share_{e}')
                A1 = getattr(self,f'e_a_share_{e}')
                P1 = getattr(self,f'e_p_share_{e}')
                k1 = self.gram_schmidt(K1)
                a1 = self.gram_schmidt(A1)
                p1 = self.gram_schmidt(P1)
                setattr(self, f'e_p_share_{e}',p1)
                setattr(self, f'e_k_share_{e}',k1)
                setattr(self, f'e_a_share_{e}',a1)
        

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = (len(vv.shape) >= 3)
        #print(vv[0][1])
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0] * self.e_pool_size_1,-1)
        #print(vv[1])
        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        if "full" not in self.args.method:
            s = int(self.task_id * self.num_clients * self.e_pool_size_1)
            f = int((self.task_id + 1) * self.num_clients * self.e_pool_size_1)
        else:
            if self.task_id == 0:
                s = int(self.task_id * self.num_clients * self.e_pool_size_1)
                f = int((self.task_id + 1) * self.num_clients * self.e_pool_size_1)
            else:
                s = int((self.task_id + 49) * self.e_pool_size_1)
                f = int((self.task_id + 50) * self.e_pool_size_1)
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
        
        return torch.nn.Parameter(uu) 
        
        # print("prceoss freq changed task_count to ", self.task_count_f)
    def next_task_locs_f(self, next_task_locs = None):
        if next_task_locs:
            self.next_task_locs = next_task_locs

    

    def get_aqk(self, x_querry, l, x_block, task_id=None, trained_task_id=None, aq_k=None, client_index=None, finished_task=None):
        e_valid = False
        indices_taskchoosing = None
        if l in self.e_layers:
            #print(trained_task_id)
            e_valid = True
            B, C = x_querry.shape
            weight = self.weight.detach().clone()
            #weight = torch.sum(weight, dim=0).unsqueeze(0).repeat(weight.shape[1], 1)
            '''
            
            if self.client_index == -1:
                weight = self.weight.detach().clone()
            else:
                weight = self.weight.detach().clone()
            weight = nn.functional.normalize(weight, dim=1)
            weight = torch.mm(weight, weight.T)
            '''
            '''
            
            
            weight = torch.zeros((10, 10), device=self.device)
            
            for i in range(10):
                weight[i][i] = 0.5

            weight[0][1] = 0.5
            weight[1][0] = 0.5
            '''

            relative_trained_task_id = {0: [[0], [1, 3]], 1: [[1], [2, 4]], 2: [[2], [0, 5]]}
            
            if True:
                
                K_all = None
                A_all = None
                p_all = None
                
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                for i in trained_task_id:
                #for i in [task_id * self.num_clients + client_index for i in range(len(trained_task_id))]:
                    trained_task_id_removed = copy.deepcopy(trained_task_id)
                    trained_task_id_removed.remove(i)
                    K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                    if K_all is None:
                        K_all = K_share
                        A_all = A_share
                        p_all = p_share 
                    else:
                        K_all = torch.cat((K_all, K_share), dim=0)
                        A_all = torch.cat((A_all, A_share), dim=0)
                        p_all = torch.cat((p_all, p_share), dim=0)
                p_all = p_all / len(trained_task_id)
                
                '''
                K_all = K_share
                A_all = A_share
                p_all = p_share
                '''
            #TODO: task choosing
            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 
            #a_querry_client = torch.einsum('bd,kd->bkd', x_querry, A_client)
            #a_querry_share = torch.einsum('bd,kd->bkd', x_querry, A_share)
            #a_querry_share = x_querry.unsqueeze(1).repeat(1, A_share.shape[0], 1)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K_all, dim=1)
            
            #n_K_share_taskchoosing = nn.functional.normalize(K1, dim=1)
            #n_K_share_taskchoosing = nn.functional.normalize(K1_representation, dim=2)
            
            q = nn.functional.normalize(a_querry, dim=2)
            #q_client = nn.functional.normalize(a_querry_client, dim=2)
            #q_share = nn.functional.normalize(a_querry_share, dim=2)
            if aq_k is None:
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                '''
                if train and self.ep_g is not None and (self.ep_g + 1) % 3 == 0:
                    weight_matrix = aq_k.clone().detach().cpu().numpy()
                    fig, ax = plt.subplots()
                    im = ax.imshow(weight_matrix, cmap='Blues')
                    fig.tight_layout()
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar(im)
                    plt.savefig('/home/piaohongming/FCL/Baselines/src/Picture/Prompt_map_wp/{}_{}_{}.png'.format(self.client_index, l, self.ep_g), dpi=300)
                    plt.close()
                '''
            else:
                aq_k = aq_k[l]
                #print(aq_k.size())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            #P_ = torch.einsum('bk,kld->bld', aq_k_all, p_all)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
        else:
            loss = 0

        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
            mean_aq_k = torch.mean(aq_k, dim=0)
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None
            mean_aq_k = None
            
        return p_return, loss, x_block, P_, indices_taskchoosing, mean_aq_k

    def forward_sharedcodap(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None, global_task_id=None):
        e_valid = False
        indices_taskchoosing = None
        same = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            if "full" not in self.args.method:
                client_global_task_id = self.task_id * self.num_clients + self.client_index
            else:
                if self.task_id == 0:
                    client_global_task_id = self.client_index
                else:
                    client_global_task_id = self.task_id + 49

            if client_global_task_id != global_task_id:
                K1 = getattr(self,f'e_k_share_{l}')[global_task_id].clone().detach()
                A1 = getattr(self,f'e_a_share_{l}')[global_task_id].clone().detach()
                p1 = getattr(self,f'e_p_share_{l}')[global_task_id].clone().detach()
            else:
                same = True
                K1 = getattr(self,f'e_k_share_{l}')[global_task_id]
                A1 = getattr(self,f'e_a_share_{l}')[global_task_id]
                p1 = getattr(self,f'e_p_share_{l}')[global_task_id]
            K_all = K1
            A_all = A1
            p_all = p1
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all)
            n_K = nn.functional.normalize(K_all, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None
        return p_return, x_block, same
    
    def forward(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None):
        #print(getattr(self,f'e_k_{0}'))
        # e prompts
        e_valid = False
        indices_taskchoosing = None
        #print(l)
        if l in self.e_layers:
            #print(l)
            e_valid = True
            B, C = x_querry.shape

            #pt = int(self.e_pool_size / (self.n_tasks))
            #s = int(self.task_id * pt)
            #f = int((self.task_id + 1) * pt)\
            
            weight = self.weight.detach().clone()
            #print(weight)
            '''
            
            if self.client_index == -1:
                weight = self.weight.detach().clone()
            else:
                weight = self.weight.detach().clone()
                #global_task_id = self.task_id * self.num_clients + self.client_index
                #weight = torch.cat((self.weight[0:global_task_id].detach().clone(),self.weight[global_task_id].unsqueeze(0),self.weight[global_task_id:].detach().clone()), dim=0)
            #print(weight)
            weight = nn.functional.normalize(weight, dim=1)
            weight = torch.mm(weight, weight.T)
            '''
            
            
            
            weight = torch.zeros((int(self.args.numclass/self.args.class_per_task), int(self.args.numclass/self.args.class_per_task)), device=self.device)
            
            for i in range(int(self.args.numclass/self.args.class_per_task)):
                weight[i][i] = 1.0

            #weight[0][0] = 0.5
            #weight[1][1] = 0.5
            #weight[0][1] = 0.5
            #weight[1][0] = 0.5

            
            #print(self.num_clients)
            relative_trained_task_id = {0: [[0], [1, 3]], 1: [[1], [2, 4]], 2: [[2], [0, 5]]}
            
            '''
            for i in [0, 1, 2, 3, 4]:
                for j in [0, 1, 2, 3, 4]:
                    weight[i][j] = 1.0
            '''
            '''
            for i in [0, 1, 5, 12, 23]:
                for j in [0, 1, 5, 12, 23]:
                    weight[i][j] = 1.0
            '''
            '''
            for i in range(50):
                for j in range(50):
                   weight[i][j] = 1.0 
            '''

            #weight = torch.einsum('bj,tj->bt', weight, weight)
            #weight = torch.div(weight, torch.max(torch.abs(weight)))
            #weight = torch.sigmoid(weight)
            #print(weight)
            #weight = torch.sigmoid(torch.div(weight + weight.T, 2))
            #weight = weight + torch.eye(self.e_task_number, device=self.device)
            #weight = torch.sigmoid(torch.cov(weight))
            #cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            #print(weight)
            #weight[self.not_trained_task_id, :] = 0
            #weight[:, self.not_trained_task_id] = 0
            if self.client_index == -1:
                '''
                
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                K1_representation = getattr(self,f'e_k_representation_{l}')[self.trained_task_id]
                K2_representation = getattr(self,f'e_k_representation_max_{l}')[self.trained_task_id]
                #K1_representation = torch.einsum('nkd,nb->bkd', K1_representation, weight[self.trained_task_id, :][:, self.trained_task_id]) 
                #print(self.trained_task_id)
                #print(weight[self.trained_task_id, :][:, self.trained_task_id].size())
                #print(torch.sum(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).size())
                K_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), K0[self.trained_task_id].reshape(K0[self.trained_task_id].shape[0], -1))
                K_client = K_client.reshape(-1, self.emb_d)
                A_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), A0[self.trained_task_id].reshape(A0[self.trained_task_id].shape[0], -1))
                A_client = A_client.reshape(-1, self.emb_d)
                p_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), p0[self.trained_task_id].reshape(p0[self.trained_task_id].shape[0], -1))
                p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                K_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), K1[self.trained_task_id].reshape(K1[self.trained_task_id].shape[0], -1))
                K_share = K_share.reshape(-1, self.emb_d)
                A_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), A1[self.trained_task_id].reshape(A1[self.trained_task_id].shape[0], -1))
                A_share = A_share.reshape(-1, self.emb_d)
                p_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), p1[self.trained_task_id].reshape(p1[self.trained_task_id].shape[0], -1))
                p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                
                K_all = torch.cat((K_client, K_share), dim=0)
                A_all = torch.cat((A_client, A_share), dim=0)
                p_all = torch.cat((p_client, p_share), dim=0)
                '''
                
                
                K_all = None
                A_all = None
                p_all = None
                '''
                
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K_all = K0.reshape(-1, self.emb_d)
                A_all = A0.reshape(-1, self.emb_d)
                p_all = p0.reshape(-1, self.e_p_length, self.emb_d)
                '''

                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                #K1_representation = getattr(self,f'e_k_representation_{l}')[self.trained_task_id]
                #_, idx = weight[global_task_id].topk(2)
                #for i in idx:
                for i in self.trained_task_id:
                #for i in [global_task_id]:
                    trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                    trained_task_id_removed.remove(i)
                    K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                    if K_all is None:
                        K_all = K_share
                        A_all = A_share
                        p_all = p_share 
                    else:
                        K_all = torch.cat((K_all, K_share), dim=0)
                        A_all = torch.cat((A_all, A_share), dim=0)
                        p_all = torch.cat((p_all, p_share), dim=0)
                p_all = p_all / len(self.trained_task_id)
                
                '''
                K_all = K_share
                A_all = A_share
                p_all = p_share
                '''
            else:
                #K0 = getattr(self,f'e_k_specific_{l}')
                #A0 = getattr(self,f'e_a_specific_{l}')
                #p0 = getattr(self,f'e_p_specific_{l}')
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                #K1_representation = getattr(self,f'e_k_representation_{l}')[[self.task_id * self.num_clients + self.client_index]]
                #print(self.client_learned_global_task_id)
                #K_client = K0[self.task_id * self.num_clients + self.client_index]
                #A_client = A0[self.task_id * self.num_clients + self.client_index]
                #p_client = p0[self.task_id * self.num_clients + self.client_index]
                if "full" not in self.args.method:
                    global_task_id = self.task_id * self.num_clients + self.client_index
                else:
                    if self.task_id == 0:
                        global_task_id = self.client_index
                    else:
                        global_task_id = self.task_id + 49
                trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                #trained_task_id_removed_forclient = copy.deepcopy(self.client_learned_global_task_id)
                trained_task_id_removed.remove(global_task_id)
                #trained_task_id_removed_forclient.remove(global_task_id)
                
                K_all = None
                A_all = None
                p_all = None
                '''
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K0 = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed].detach().clone()), dim=0)
                A0 = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed].detach().clone()), dim=0)
                p0 = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed].detach().clone()), dim=0)
                K_all = K0.reshape(-1, self.emb_d)
                A_all = A0.reshape(-1, self.emb_d)
                p_all = p0.reshape(-1, self.e_p_length, self.emb_d)
                '''
                
                if train:
                    #K_share = K1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    #A_share = A1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    #p_share = p1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    '''
                    if global_task_id == 0:
                        K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[global_task_id + 1:]), dim=0)
                        A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[global_task_id + 1:]), dim=0)
                        p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[global_task_id + 1:]), dim=0)
                    elif global_task_id == (self.e_task_number - 1):
                        K_share = torch.cat((K1[0:global_task_id],K1[global_task_id].unsqueeze(0)), dim=0)
                        A_share = torch.cat((A1[0:global_task_id],A1[global_task_id].unsqueeze(0)), dim=0)
                        p_share = torch.cat((p1[0:global_task_id],p1[global_task_id].unsqueeze(0)), dim=0)
                    else:
                        K_share = torch.cat((K1[0:global_task_id],K1[global_task_id].unsqueeze(0),K1[global_task_id + 1:]), dim=0)
                        A_share = torch.cat((A1[0:global_task_id],A1[global_task_id].unsqueeze(0),A1[global_task_id + 1:]), dim=0)
                        p_share = torch.cat((p1[0:global_task_id],p1[global_task_id].unsqueeze(0),p1[global_task_id + 1:]), dim=0)
                    '''
                    '''
                    K_client = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    A_client = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    p_client = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    weight_client = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed_forclient]), dim=0).unsqueeze(0)
                    
                    K_client = torch.mm(weight_client, K_client.reshape(K_client.shape[0], -1))
                    K_client = K_client.reshape(-1, self.emb_d)
                    A_client = torch.mm(weight_client, A_client.reshape(A_client.shape[0], -1))
                    A_client = A_client.reshape(-1, self.emb_d)
                    p_client = torch.mm(weight_client, p_client.reshape(p_client.shape[0], -1))
                    p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed].detach().clone()), dim=0)
                    A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed].detach().clone()), dim=0)
                    p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed].detach().clone()), dim=0)
                    weight_share = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed]), dim=0).unsqueeze(0)
                    
                    
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_all = torch.cat((K_client, K_share), dim=0) 
                    A_all = torch.cat((A_client, A_share), dim=0)
                    p_all = torch.cat((p_client, p_share), dim=0)
                    '''
                    
                    
                    #for i in relative_trained_task_id[self.client_index][self.task_id]:
                    #_, idx = weight[global_task_id].topk(2)
                    #for i in idx:
                    for i in self.trained_task_id:
                        #print(self.trained_task_id)
                    #for i in [global_task_id]:
                        if i != global_task_id:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            trained_task_id_removed_again.remove(i)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[i].detach().clone().unsqueeze(0),K1[trained_task_id_removed_again].detach().clone()), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[i].detach().clone().unsqueeze(0),A1[trained_task_id_removed_again].detach().clone()), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[i].detach().clone().unsqueeze(0),p1[trained_task_id_removed_again].detach().clone()), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                        else:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed_again].detach().clone()), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed_again].detach().clone()), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed_again].detach().clone()), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                    p_all = p_all / len(self.trained_task_id)
                    
                    '''
                    K_all = K_share
                    A_all = A_share
                    p_all = p_share
                    '''
                    
                    
                else:
                    '''
                    K_client = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed_forclient]), dim=0)
                    A_client = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed_forclient]), dim=0)
                    p_client = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed_forclient]), dim=0)
                    weight_client = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed_forclient]), dim=0).unsqueeze(0)
                    
                    K_client = torch.mm(weight_client, K_client.reshape(K_client.shape[0], -1))
                    K_client = K_client.reshape(-1, self.emb_d)
                    A_client = torch.mm(weight_client, A_client.reshape(A_client.shape[0], -1))
                    A_client = A_client.reshape(-1, self.emb_d)
                    p_client = torch.mm(weight_client, p_client.reshape(p_client.shape[0], -1))
                    p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_all = torch.cat((K_client, K_share), dim=0)
                    A_all = torch.cat((A_client, A_share), dim=0)
                    p_all = torch.cat((p_client, p_share), dim=0)
                    '''
                    
                    
                    
                    #for i in relative_trained_task_id[self.client_index][self.task_id]:
                    #_, idx = weight[global_task_id].topk(2)
                    #for i in idx:
                    for i in self.trained_task_id:
                    #for i in [global_task_id]:
                        if i != global_task_id:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            trained_task_id_removed_again.remove(i)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[i].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[i].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[i].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                        else:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                            A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                            p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                    p_all = p_all / len(self.trained_task_id)
                    

                    '''
                    K_all = K_share
                    A_all = A_share
                    p_all = p_share
                    '''

            #TODO: task choosing


            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 
            #a_querry_client = torch.einsum('bd,kd->bkd', x_querry, A_client)
            #a_querry_share = torch.einsum('bd,kd->bkd', x_querry, A_share)
            #a_querry_share = x_querry.unsqueeze(1).repeat(1, A_share.shape[0], 1)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K_all, dim=1)
            
            #n_K_share_taskchoosing = nn.functional.normalize(K1, dim=1)
            #n_K_share_taskchoosing = nn.functional.normalize(K1_representation, dim=2)
            
            q = nn.functional.normalize(a_querry, dim=2)
            #q_client = nn.functional.normalize(a_querry_client, dim=2)
            #q_share = nn.functional.normalize(a_querry_share, dim=2)
            if aq_k is None:
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                #aq_k_client = torch.einsum('bkd,kd->bk', q_client, n_K_client)
                #aq_k_share = torch.einsum('bkd,kd->bk', q_share, n_K_share)
                #aq_k_share_taskchoosing = torch.einsum('bd,nd->bn', nn.functional.normalize(x_querry, dim=1), n_K_share_taskchoosing)
                #print(K1_representation.size())
                #x_querry_mean = torch.mean(x_querry, dim=0).unsqueeze(0).repeat(x_querry.shape[0], 1)
                #aq_k_share_taskchoosing = torch.einsum('bkd,nkd->bnk', nn.functional.normalize(x_querry_mean.unsqueeze(1).repeat(1, n_K_share_taskchoosing.shape[1], 1), dim=2), n_K_share_taskchoosing)
                #aq_k_share_taskchoosing = torch.mean(aq_k_share_taskchoosing, dim=2)
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, global_task_id])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, :])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, 31])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, 13])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, 44])
                #print(aq_k_share_taskchoosing.size())

                #aq_k_share_taskchoosing = aq_k_share_taskchoosing - 0.1 * K2_representation.unsqueeze(0).repeat(aq_k_share_taskchoosing.shape[0], 1, 1) 
                #aq_k_share_taskchoosing, _ = torch.max(aq_k_share_taskchoosing, dim=2)
                #print(aq_k_share_taskchoosing.size())
                #aq_k_share_taskchoosing = aq_k_share_taskchoosing
                #topk_for_taskchoosing = torch.topk(aq_k_share_taskchoosing, 1, dim=1)
                #K2_representation[topk_for_taskchoosing.indices]
                #topk_for_taskchoosing = torch.topk(aq_k_share_taskchoosing, 5, dim=1)
                #indices_taskchoosing = topk_for_taskchoosing.indices
                #for i in range(indices_taskchoosing.shape[0]):
                    #for j in range(indices_taskchoosing.shape[1]):
                        #indices_taskchoosing[i][j] = self.trained_task_id[int(indices_taskchoosing[i][j].cpu().numpy())]
                #print(indices_taskchoosing[0])
                #aq_k_all = torch.cat((aq_k_client, aq_k_share), dim=1)
                
                #aq_k = torch.softmax(aq_k, dim=1)
                '''
                if train and self.ep_g is not None and (self.ep_g + 1) % 3 == 0:
                    weight_matrix = aq_k.clone().detach().cpu().numpy()
                    fig, ax = plt.subplots()
                    im = ax.imshow(weight_matrix, cmap='Blues')
                    fig.tight_layout()
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar(im)
                    plt.savefig('/home/piaohongming/FCL/Baselines/src/Picture/Prompt_map_wp/{}_{}_{}.png'.format(self.client_index, l, self.ep_g), dpi=300)
                    plt.close()
                '''
            else:
                aq_k = aq_k[l]
                #print(aq_k.size())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            #P_ = torch.einsum('bk,kld->bld', aq_k_all, p_all)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
            
            #K_client_nor = nn.functional.normalize(K_client, dim=1)
            #K_share_nor = nn.functional.normalize(K_share, dim=1)
            #loss = (1.0 - torch.abs(aq_k_share_taskchoosing.mean())) * 1
            
            '''
            q_log_prob = torch.log(torch.mean(nn.functional.normalize(x_querry, dim=1), dim=0))
            p_log_prob = torch.log(torch.mean(n_K_share_specific, dim=0))
            print(q_log_prob)
            print(p_log_prob)
            loss = F.kl_div(p_log_prob, q_log_prob, reduction='sum')
            '''
            #loss += self.ortho_penalty(K_client_nor)
            #loss += self.ortho_penalty(K_share_nor)
            
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
                    loss = self.ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = self.ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    # print("using all ortho penalty")

                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None

        # return
        if train:
            if aq_k is not None:
                return p_return, loss, x_block, P_, indices_taskchoosing, torch.mean(aq_k, dim=0)
            else:
                return p_return, loss, x_block, P_, indices_taskchoosing, None
        else:
            return p_return, 0, x_block, indices_taskchoosing





class CodaPrompt_2d_v2(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device='cuda:0', clients_local=10, num_clients = 10, args=None):
        print(" in CODA prompt")
        super().__init__()
        self.args = args
        self.task_id = 0
        self.max_classes = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        self.next_task_locs= None
        self.device = device
        self.task_count_f = 0
        self.clients_local = clients_local
        self.num_clients = num_clients
        self.client_index = -1
        self.trained_task_id = None
        self.trained_task_id_forchoosing = None
        self.not_trained_task_id = None
        self.client_learned_global_task_id = None
        self.ep_g = None
        self.global_task_id_real = None
        self.topk_com = None

        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            e_l_2 = self.e_p_length_2
            #p0 = tensor_prompt(self.e_task_number, self.e_pool_size_0, e_l, emb_d) #TODO: 50 * 5 * 8 * 768
            #k0 = tensor_prompt(self.e_task_number, self.e_pool_size_0, self.key_d)
            #a0 = tensor_prompt(self.e_task_number, self.e_pool_size_0, self.key_d)
            #p0 = self.gram_schmidt(p0)
            #k0 = self.gram_schmidt(k0)
            #a0 = self.gram_schmidt(a0)
            if "extension" in self.args.method or "extencl" in self.args.method:
                if "classincremental" in self.args.method:
                    p0 = tensor_prompt(self.e_task_number, self.e_pool_size_2, e_l_2, emb_d, same=False, ortho=True)
                    k0 = tensor_prompt(self.e_task_number, self.e_pool_size_2, self.key_d, same=False, ortho=True)
                    a0 = tensor_prompt(self.e_task_number, self.e_pool_size_2, self.key_d, same=False, ortho=True)
                p1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, e_l, emb_d, same=False, ortho=True) #TODO: 50 * 5 * 8 * 768
                k1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=False, ortho=True)
                a1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=False, ortho=True)
            else:
                if "classincremental" in self.args.method:
                    p0 = tensor_prompt(self.e_task_number, self.e_pool_size_2, e_l_2, emb_d, same=True)
                    k0 = tensor_prompt(self.e_task_number, self.e_pool_size_2, self.key_d, same=True)
                    a0 = tensor_prompt(self.e_task_number, self.e_pool_size_2, self.key_d, same=True)
                p1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, e_l, emb_d, same=True) #TODO: 50 * 5 * 8 * 768
                k1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=True)
                a1 = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d, same=True)

            #setattr(self, f'e_p_specific_{e}',p0)
            #setattr(self, f'e_k_specific_{e}',k0)
            #setattr(self, f'e_a_specific_{e}',a0)
            if "classincremental" in self.args.method:
                setattr(self, f'e_p_divide_{e}',p0)
                setattr(self, f'e_k_divide_{e}',k0)
                setattr(self, f'e_a_divide_{e}',a0)
            setattr(self, f'e_p_share_{e}',p1)
            setattr(self, f'e_k_share_{e}',k1)
            setattr(self, f'e_a_share_{e}',a1)

            #k_representation = tensor_prompt(self.e_task_number, self.e_pool_size_1, self.key_d)
            #setattr(self, f'e_k_representation_{e}',k_representation)
        #self.weight = torch.nn.Parameter(torch.FloatTensor(20, 30).uniform_(0, 1), requires_grad=True)
        #self.fc_weight = torch.nn.Parameter(torch.FloatTensor(200, 30).uniform_(0, 1), requires_grad=True)
        #self.weight = torch.nn.Parameter(torch.FloatTensor(1, 50).uniform_(0, 1).repeat(self.e_task_number, 1), requires_grad=True)
        self.weight = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.weight_c = torch.eye(int(self.args.numclass/self.args.class_per_task), device=self.device)
        self.fc_weight = torch.eye(self.args.numclass, device=self.device)
        self.task_embedding = torch.FloatTensor(int(self.args.numclass/self.args.class_per_task), emb_d).to(self.device)
        self.K_extra_prompt = None
        self.A_extra_prompt = None
        self.p_extra_prompt = None
        self.K_extra_prompt_divide = None
        self.A_extra_prompt_divide = None
        self.p_extra_prompt_divide = None
        #self.weight = torch.ones((10, 10), device=self.device)
        #self.fc_weight = torch.ones((200, 200), device=self.device)
        #self.weight = torch.nn.Parameter(torch.ones((self.e_task_number, self.e_task_number)), requires_grad=True)    
        #nn.init.uniform_(self.weight, a=0.0, b=1.0)
        
    
    #def get_initial_promptchoosing(self, ):
    def delete_extra_prompt(self):
        self.K_extra_prompt = {}
        self.A_extra_prompt = {}
        self.p_extra_prompt = {}
        self.K_extra_prompt_divide = {}
        self.A_extra_prompt_divide = {}
        self.p_extra_prompt_divide = {}

    def save_extra_prompt(self):
        self.K_extra_prompt = {}
        self.A_extra_prompt = {}
        self.p_extra_prompt = {}
        self.K_extra_prompt_divide = {}
        self.A_extra_prompt_divide = {}
        self.p_extra_prompt_divide = {}
        #global_task_id = self.task_id * self.num_clients + self.client_index
        #global_task_id = self.global_task_id_real[global_task_id]
        for e in self.e_layers:
            if "classincremental" in self.args.method:
                K0 = getattr(self,f'e_k_divide_{e}')
                A0 = getattr(self,f'e_a_divide_{e}')
                p0 = getattr(self,f'e_p_divide_{e}')
                self.K_extra_prompt_divide[e] = copy.deepcopy(K0.data)
                self.A_extra_prompt_divide[e] = copy.deepcopy(A0.data)
                self.p_extra_prompt_divide[e] = copy.deepcopy(p0.data)
            K1 = getattr(self,f'e_k_share_{e}')
            A1 = getattr(self,f'e_a_share_{e}')
            p1 = getattr(self,f'e_p_share_{e}')
            self.K_extra_prompt[e] = copy.deepcopy(K1.data)
            self.A_extra_prompt[e] = copy.deepcopy(A1.data)
            self.p_extra_prompt[e] = copy.deepcopy(p1.data)
            

    def load_extra_prompt(self):
        for e in self.e_layers:
            if "classincremental" in self.args.method:
                K0 = getattr(self,f'e_k_divide_{e}')
                A0 = getattr(self,f'e_a_divide_{e}')
                p0 = getattr(self,f'e_p_divide_{e}')
                p0.data = self.p_extra_prompt_divide[e]
                A0.data = self.A_extra_prompt_divide[e]
                K0.data = self.K_extra_prompt_divide[e]
                setattr(self, f'e_p_divide_{e}',p0)
                setattr(self, f'e_k_divide_{e}',K0)
                setattr(self, f'e_a_divide_{e}',A0)
            K1 = getattr(self,f'e_k_share_{e}')
            A1 = getattr(self,f'e_a_share_{e}')
            p1 = getattr(self,f'e_p_share_{e}')
            p1.data = self.p_extra_prompt[e]
            A1.data = self.A_extra_prompt[e]
            K1.data = self.K_extra_prompt[e]
            setattr(self, f'e_p_share_{e}',p1)
            setattr(self, f'e_k_share_{e}',K1)
            setattr(self, f'e_a_share_{e}',A1)

    
    
    def init_with_representation(self, model_g, train_dataset, class_distribution_client_real):
        #K1 = getattr(self,f'e_k_representation_{0}')
        #A1 = getattr(self,f'e_a_share_{0}')
        #P1 = getattr(self,f'e_p_share_{0}')
        #k_representation = tensor_prompt(self.e_task_number, self.e_pool_size_1).cuda(self.device)
        #setattr(self, f'e_k_representation_max_{0}',k_representation)
        
        model_g.eval()
        with torch.no_grad():
            K1_list = []
            K2_list = []
            for i in range(50):
                print(f'global task {i}')
                client_index = i % 10
                train_classes = class_distribution_client_real[client_index][int(i // 10)]
                train_classes_real = train_classes
                train_dataset.getTrainData(train_classes, [], [], client_index, classes_real=train_classes_real)
                train_loader = DataLoader(dataset=train_dataset,
                                        shuffle=True,
                                        batch_size=128,
                                        num_workers=8,
                                        pin_memory=True)
                labels = []
                features = []
                for batch_idx, (indexs, images, target) in enumerate(train_loader):
                    if isinstance(self.device, int):
                        feature = model_g.feature_extractor(images.to(self.device))
                    else:
                        feature = model_g.feature_extractor(images.cuda())
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
                labels = np.concatenate([label_vector for label_vector in labels])
                features = np.concatenate([feature_vector for feature_vector in features], axis=0)
                max_K2, _ = torch.min(torch.einsum('bd,d->b', nn.functional.normalize(torch.FloatTensor(features).cuda(self.device), dim=1), nn.functional.normalize(torch.FloatTensor(np.mean(features, axis=0)).cuda(self.device), dim=0)), dim=0)
                K1_list.append(torch.FloatTensor(np.mean(features, axis=0)).cuda(self.device).unsqueeze(0).repeat(5, 1).unsqueeze(0))
                K2_list.append(max_K2.repeat(5).unsqueeze(0))
                #torch.nn.Parameter(torch.FloatTensor(1,b).uniform_(0, 1).repeat(a, 1), requires_grad=True)
                #print(K1[i].size())
        K1 = torch.nn.Parameter(torch.cat(K1_list, dim=0), requires_grad=False)
        K2 = torch.nn.Parameter(torch.cat(K2_list, dim=0), requires_grad=False)
        #setattr(self, f'e_p_share_{0}',P1)
        #setattr(self, f'e_k_share_{0}',K1)
        setattr(self, f'e_k_representation_{0}',K1)
        setattr(self, f'e_k_representation_max_{0}',K2)
        #setattr(self, f'e_a_share_{0}',A1)
        

    def init_with_class_representation(self, model_g, train_dataset, class_distribution_client_real):
        #K1 = getattr(self,f'e_k_representation_{0}')
        #A1 = getattr(self,f'e_a_share_{0}')
        #P1 = getattr(self,f'e_p_share_{0}')
        
        model_g.eval()
        with torch.no_grad():
            K1_list = []
            for i in range(50):
                print(f'global task {i}')
                client_index = i % 10
                train_classes = class_distribution_client_real[client_index][int(i // 10)]
                train_classes_real = train_classes
                features_task = None
                for i in train_classes:
                    train_dataset.getTrainData([i], [], [], client_index, classes_real=[i])
                    train_loader = DataLoader(dataset=train_dataset,
                                            shuffle=True,
                                            batch_size=128,
                                            num_workers=8,
                                            pin_memory=True)
                    labels = []
                    features = []
                    for batch_idx, (indexs, images, target) in enumerate(train_loader):
                        if isinstance(self.device, int):
                            feature = model_g.feature_extractor(images.to(self.device))
                        else:
                            feature = model_g.feature_extractor(images.cuda())
                        labels.append(target.numpy())
                        features.append(feature.cpu().numpy())
                    #labels = np.concatenate([label_vector for label_vector in labels])
                    if features_task is None:
                        features_task = torch.FloatTensor(np.mean(np.concatenate([feature_vector for feature_vector in features], axis=0), axis=0)).cuda(self.device).unsqueeze(0)
                    else:
                        features_task = torch.cat((features_task, torch.FloatTensor(np.mean(np.concatenate([feature_vector for feature_vector in features], axis=0), axis=0)).cuda(self.device).unsqueeze(0)), dim=0)
                K1_list.append(features_task.unsqueeze(0))
                #torch.nn.Parameter(torch.FloatTensor(1,b).uniform_(0, 1).repeat(a, 1), requires_grad=True)
                #print(K1[i].size())
        K1 = torch.nn.Parameter(torch.cat(K1_list, dim=0), requires_grad=False)

        #setattr(self, f'e_p_share_{0}',P1)
        #setattr(self, f'e_k_share_{0}',K1)
        setattr(self, f'e_k_representation_{0}',K1)
        
        #setattr(self, f'e_a_share_{0}',A1)
    
    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        #self.e_pool_size = prompt_param[0]
        #self.e_p_length = prompt_param[1]

        self.e_task_number = prompt_param[0] #TODO: 50
        self.e_pool_size_0 = prompt_param[1] #TODO: 5
        self.e_pool_size_1 = prompt_param[2] #TODO: 5
        self.e_p_length = prompt_param[3] #TODO: 8
        self.e_pool_size_2 = prompt_param[7] #TODO: 5
        self.e_p_length_2 = prompt_param[8] #TODO: 8


        # prompt locations
        #self.e_layers = [0, 1, 2, 3, 4]
        self.e_layers = [4, 5, 6]

        # location of ortho penalty
        self.ortho_mu = prompt_param[4] 
        print("ortho_mu ", self.ortho_mu)

        # ablations
        '''
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
        '''
        
    def process_frequency(self, next_task_locs = None):
        
        self.task_count_f += 1
        '''        
        for e in self.e_layers:
            K0 = getattr(self,f'e_k_specific_{e}')
            A0 = getattr(self,f'e_a_specific_{e}')
            P0 = getattr(self,f'e_p_specific_{e}')
            k0 = self.gram_schmidt(K0)
            a0 = self.gram_schmidt(A0)
            p0 = self.gram_schmidt(P0)
            setattr(self, f'e_p_specific_{e}',p0)
            setattr(self, f'e_k_specific_{e}',k0)
            setattr(self, f'e_a_specific_{e}',a0)
        '''
        if "extension" in self.args.method or "extencl" in self.args.method:
            pass
        else:
            for e in self.e_layers:
                if "classincremental" in self.args.method:
                    print("incremental for classincremental")
                    K0 = getattr(self,f'e_k_divide_{e}')
                    A0 = getattr(self,f'e_a_divide_{e}')
                    P0 = getattr(self,f'e_p_divide_{e}')
                    k0 = self.gram_schmidt(K0)
                    a0 = self.gram_schmidt(A0)
                    p0 = self.gram_schmidt(P0)
                    setattr(self, f'e_p_divide_{e}',p0)
                    setattr(self, f'e_k_divide_{e}',k0)
                    setattr(self, f'e_a_divide_{e}',a0)
                print("incremental for taskincremental")
                K1 = getattr(self,f'e_k_share_{e}')
                A1 = getattr(self,f'e_a_share_{e}')
                P1 = getattr(self,f'e_p_share_{e}')
                k1 = self.gram_schmidt(K1)
                a1 = self.gram_schmidt(A1)
                p1 = self.gram_schmidt(P1)
                setattr(self, f'e_p_share_{e}',p1)
                setattr(self, f'e_k_share_{e}',k1)
                setattr(self, f'e_a_share_{e}',a1)
        

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = (len(vv.shape) >= 3)
        #print(vv[0][1])
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0] * self.e_pool_size_1,-1)
        #print(vv[1])
        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        if "full" not in self.args.method:
            s = int(self.task_id * self.num_clients * self.e_pool_size_1)
            f = int((self.task_id + 1) * self.num_clients * self.e_pool_size_1)
        else:
            if self.task_id == 0:
                s = int(self.task_id * self.num_clients * self.e_pool_size_1)
                f = int((self.task_id + 1) * self.num_clients * self.e_pool_size_1)
            else:
                s = int((self.task_id + 49) * self.e_pool_size_1)
                f = int((self.task_id + 50) * self.e_pool_size_1)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if "classincremental" in self.args.method or "extension" in self.args.method or "extencl" in self.args.method:
                        if j < s:
                            '''
                            if (j % self.e_pool_size_1) in self.trained_task_id:
                                if not redo:
                                    uj = uu[:, j].clone()
                                    proj = projection(uj, vk)
                                    if proj is None:
                                        redo = True
                                        print('restarting!!!')
                                    else:
                                        uk = uk + proj
                            '''
                            if not redo:
                                uj = uu[:, j].clone()
                                proj = projection(uj, vk)
                                if proj is None:
                                    redo = True
                                    print('restarting!!!')
                                else:
                                    uk = uk + proj
                        else:
                            if not redo:
                                uj = uu[:, j].clone()
                                proj = projection(uj, vk)
                                if proj is None:
                                    redo = True
                                    print('restarting!!!')
                                else:
                                    uk = uk + proj
                    else:
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
        
        return torch.nn.Parameter(uu) 
        
        # print("prceoss freq changed task_count to ", self.task_count_f)
    def next_task_locs_f(self, next_task_locs = None):
        if next_task_locs:
            self.next_task_locs = next_task_locs

    def forward_with_attention(self, x_querry, l, x_block, train=False, task_id=None):
        #print(l)
        #print(self.e_layers)
        aq_k = None
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            #pt = int(self.e_pool_size / (self.n_tasks))
            #s = int(self.task_id * pt)
            #f = int((self.task_id + 1) * pt)
            weight = self.weight.detach().clone()
            weight = nn.functional.normalize(weight, dim=1)
            weight = torch.mm(weight, weight.T)
            if self.client_index == -1:
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                #print(self.trained_task_id)
                #print(weight[self.trained_task_id, :][:, self.trained_task_id].size())
                #print(torch.sum(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).size())
                K_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), K0[self.trained_task_id].reshape(K0[self.trained_task_id].shape[0], -1))
                K_client = K_client.reshape(-1, self.emb_d)
                A_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), A0[self.trained_task_id].reshape(A0[self.trained_task_id].shape[0], -1))
                A_client = A_client.reshape(-1, self.emb_d)
                p_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), p0[self.trained_task_id].reshape(p0[self.trained_task_id].shape[0], -1))
                p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                K_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), K1[self.trained_task_id].reshape(K1[self.trained_task_id].shape[0], -1))
                K_share = K_share.reshape(-1, self.emb_d)
                A_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), A1[self.trained_task_id].reshape(A1[self.trained_task_id].shape[0], -1))
                A_share = A_share.reshape(-1, self.emb_d)
                p_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), p1[self.trained_task_id].reshape(p1[self.trained_task_id].shape[0], -1))
                p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                K_all = torch.cat((K_client, K_share), dim=0)
                A_all = torch.cat((A_client, A_share), dim=0)
                p_all = torch.cat((p_client, p_share), dim=0)
            else:
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                #K_client = K0[self.task_id * self.num_clients + self.client_index]
                #A_client = A0[self.task_id * self.num_clients + self.client_index]
                #p_client = p0[self.task_id * self.num_clients + self.client_index]
                global_task_id = self.task_id * self.num_clients + self.client_index
                trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                trained_task_id_removed_forclient = copy.deepcopy(self.client_learned_global_task_id)
                trained_task_id_removed.remove(global_task_id)
                trained_task_id_removed_forclient.remove(global_task_id)
                if train:
                    #K_share = K1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    #A_share = A1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    #p_share = p1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    
                    '''
                    if global_task_id == 0:
                        K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[global_task_id + 1:]), dim=0)
                        A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[global_task_id + 1:]), dim=0)
                        p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[global_task_id + 1:]), dim=0)
                    elif global_task_id == (self.e_task_number - 1):
                        K_share = torch.cat((K1[0:global_task_id],K1[global_task_id].unsqueeze(0)), dim=0)
                        A_share = torch.cat((A1[0:global_task_id],A1[global_task_id].unsqueeze(0)), dim=0)
                        p_share = torch.cat((p1[0:global_task_id],p1[global_task_id].unsqueeze(0)), dim=0)
                    else:
                        K_share = torch.cat((K1[0:global_task_id],K1[global_task_id].unsqueeze(0),K1[global_task_id + 1:]), dim=0)
                        A_share = torch.cat((A1[0:global_task_id],A1[global_task_id].unsqueeze(0),A1[global_task_id + 1:]), dim=0)
                        p_share = torch.cat((p1[0:global_task_id],p1[global_task_id].unsqueeze(0),p1[global_task_id + 1:]), dim=0)
                    '''

                    K_client = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    A_client = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    p_client = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    weight_client = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed_forclient]), dim=0).unsqueeze(0)
                    
                    K_client = torch.mm(weight_client, K_client.reshape(K_client.shape[0], -1))
                    K_client = K_client.reshape(-1, self.emb_d)
                    A_client = torch.mm(weight_client, A_client.reshape(A_client.shape[0], -1))
                    A_client = A_client.reshape(-1, self.emb_d)
                    p_client = torch.mm(weight_client, p_client.reshape(p_client.shape[0], -1))
                    p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                    
                    K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed].detach().clone()), dim=0)
                    A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed].detach().clone()), dim=0)
                    p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed].detach().clone()), dim=0)
                    weight_share = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed]), dim=0).unsqueeze(0)
                    
                    '''
                    K_share = torch.cat((K1[0:4],K1[4:]), dim=0)
                    A_share = torch.cat((A1[0:4],A1[4:]), dim=0)
                    p_share = torch.cat((p1[0:4],p1[4:]), dim=0)
                    '''
                    '''
                    K_share = K1
                    A_share = A1
                    p_share = p1
                    weight_share = weight
                    '''
                    '''
                    K_share = K_share[self.task_id * self.num_clients + self.client_index]
                    A_share = A_share[self.task_id * self.num_clients + self.client_index]
                    p_share = p_share[self.task_id * self.num_clients + self.client_index]
                    '''
                    #weight_share = torch.softmax(weight_share, dim=1)

                    #weight = torch.ones(weight.shape).cuda(self.device)
                    
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                    
                    
                    K_all = torch.cat((K_client, K_share), dim=0) 
                    A_all = torch.cat((A_client, A_share), dim=0)
                    p_all = torch.cat((p_client, p_share), dim=0)

                else:
                    K_client = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed_forclient]), dim=0)
                    A_client = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed_forclient]), dim=0)
                    p_client = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed_forclient]), dim=0)
                    weight_client = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed_forclient]), dim=0).unsqueeze(0)
                    
                    K_client = torch.mm(weight_client, K_client.reshape(K_client.shape[0], -1))
                    K_client = K_client.reshape(-1, self.emb_d)
                    A_client = torch.mm(weight_client, A_client.reshape(A_client.shape[0], -1))
                    A_client = A_client.reshape(-1, self.emb_d)
                    p_client = torch.mm(weight_client, p_client.reshape(p_client.shape[0], -1))
                    p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)


                    K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed]), dim=0).unsqueeze(0)
                    #weight_share = torch.softmax(weight_share, dim=1)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                    K_all = torch.cat((K_client, K_share), dim=0)
                    A_all = torch.cat((A_client, A_share), dim=0)
                    p_all = torch.cat((p_client, p_share), dim=0)
                


            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 
            #a_querry = x_querry.unsqueeze(1).repeat(1, A.shape[0], 1)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K_all, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            #aq_k = torch.softmax(aq_k, dim=1)
            #print(aq_k.size())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)

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
                    loss = self.ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = self.ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    # print("using all ortho penalty")

                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
        else:
            p_return = None
            P_ = None

        # return
        return p_return, aq_k, x_block

    #def forward_with_another_attention(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None):

    def get_aqk(self, x_querry, l, x_block, task_id=None, trained_task_id=None, aq_k=None, client_index=None, finished_task=None):
        e_valid = False
        indices_taskchoosing = None
        if l in self.e_layers:
            #print(trained_task_id)
            e_valid = True
            B, C = x_querry.shape
            
            weight = self.weight_c.detach().clone()
            #weight = torch.sum(weight, dim=0).unsqueeze(0).repeat(weight.shape[1], 1)
            '''
            
            if self.client_index == -1:
                weight = self.weight.detach().clone()
            else:
                weight = self.weight.detach().clone()
            weight = nn.functional.normalize(weight, dim=1)
            weight = torch.mm(weight, weight.T)
            '''
            '''
            
            
            weight = torch.zeros((10, 10), device=self.device)
            
            for i in range(10):
                weight[i][i] = 0.5

            weight[0][1] = 0.5
            weight[1][0] = 0.5
            '''

            relative_trained_task_id = {0: [[0], [1, 3]], 1: [[1], [2, 4]], 2: [[2], [0, 5]]}
            
            if True:
                K_all = None
                A_all = None
                p_all = None
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                if "efficient1" not in self.args.method:
                    for i in trained_task_id:
                    #for i in [task_id * self.num_clients + client_index for i in range(len(trained_task_id))]:    
                        trained_task_id_removed = copy.deepcopy(trained_task_id)
                        #trained_task_id_removed = copy.deepcopy(finished_task[i])
                        trained_task_id_removed.remove(i)
                        K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                        A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                        p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                        weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                        K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                        K_share = K_share.reshape(-1, self.emb_d)
                        A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                        A_share = A_share.reshape(-1, self.emb_d)
                        p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                        p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                        if K_all is None:
                            K_all = K_share
                            A_all = A_share
                            p_all = p_share 
                        else:
                            K_all = torch.cat((K_all, K_share), dim=0)
                            A_all = torch.cat((A_all, A_share), dim=0)
                            p_all = torch.cat((p_all, p_share), dim=0)
                else:
                    pass
                
                p_all = p_all / len(trained_task_id)
                
                '''
                K_all = K_share
                A_all = A_share
                p_all = p_share
                '''
            #TODO: task choosing
            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 
            #a_querry_client = torch.einsum('bd,kd->bkd', x_querry, A_client)
            #a_querry_share = torch.einsum('bd,kd->bkd', x_querry, A_share)
            #a_querry_share = x_querry.unsqueeze(1).repeat(1, A_share.shape[0], 1)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K_all, dim=1)
            
            #n_K_share_taskchoosing = nn.functional.normalize(K1, dim=1)
            #n_K_share_taskchoosing = nn.functional.normalize(K1_representation, dim=2)
            
            q = nn.functional.normalize(a_querry, dim=2)
            #q_client = nn.functional.normalize(a_querry_client, dim=2)
            #q_share = nn.functional.normalize(a_querry_share, dim=2)
            if aq_k is None:
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                '''
                if train and self.ep_g is not None and (self.ep_g + 1) % 3 == 0:
                    weight_matrix = aq_k.clone().detach().cpu().numpy()
                    fig, ax = plt.subplots()
                    im = ax.imshow(weight_matrix, cmap='Blues')
                    fig.tight_layout()
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar(im)
                    plt.savefig('/home/piaohongming/FCL/Baselines/src/Picture/Prompt_map_wp/{}_{}_{}.png'.format(self.client_index, l, self.ep_g), dpi=300)
                    plt.close()
                '''
            else:
                aq_k = aq_k[l]
                #print(aq_k.size())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            #P_ = torch.einsum('bk,kld->bld', aq_k_all, p_all)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
        else:
            loss = 0

        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
            mean_aq_k = torch.mean(aq_k, dim=0)
            #print(mean_aq_k.size())
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None
            mean_aq_k = None
            
        return p_return, loss, x_block, P_, indices_taskchoosing, mean_aq_k
    
    
    def forward_detect(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None, x_divide=None, detect_task_id=None):
        topk_com = self.topk_com
        if topk_com > len(self.trained_task_id):
            topk_com = len(self.trained_task_id)
        e_valid = False
        indices_taskchoosing = None
        if l in self.e_layers:
            e_valid = True
            #B, C = x_querry.shape
            weight = self.weight.detach().clone()

            K_all = None
            A_all = None
            p_all = None
            K1 = getattr(self,f'e_k_share_{l}')
            A1 = getattr(self,f'e_a_share_{l}')
            p1 = getattr(self,f'e_p_share_{l}')
            global_task_id = detect_task_id
            _, idx = weight[global_task_id][self.trained_task_id].topk(topk_com)
            idx = [self.trained_task_id[id] for id in idx]
            for i in idx:
                trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                trained_task_id_removed.remove(i)
                K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                K_share = K_share.reshape(-1, self.emb_d)
                A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                A_share = A_share.reshape(-1, self.emb_d)
                p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                if K_all is None:
                    K_all = K_share
                    A_all = A_share
                    p_all = p_share 
                else:
                    K_all = torch.cat((K_all, K_share), dim=0)
                    A_all = torch.cat((A_all, A_share), dim=0)
                    p_all = torch.cat((p_all, p_share), dim=0)
            p_all = p_all / topk_com
            
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all)
            n_K = nn.functional.normalize(K_all, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
            loss = 0
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
                    loss = self.ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = self.ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    # print("using all ortho penalty")

                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
            loss = 0
        
        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None

        # return
        if train:
            if aq_k is not None:
                return p_return, loss, x_block, P_, indices_taskchoosing, torch.mean(aq_k, dim=0)
            else:
                return p_return, loss, x_block, P_, indices_taskchoosing, None
        else:
            return p_return, 0, x_block, indices_taskchoosing


    
    def forward_divide(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None, x_divide=None):
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            if self.client_index == -1:
                K_all = None
                A_all = None
                p_all = None
                K0 = getattr(self,f'e_k_divide_{l}')
                A0 = getattr(self,f'e_a_divide_{l}')
                p0 = getattr(self,f'e_p_divide_{l}')
                
                for i in self.trained_task_id:
                    if K_all is None:
                        K_all = K0[i].detach().clone().unsqueeze(0)
                        A_all = A0[i].detach().clone().unsqueeze(0)
                        p_all = p0[i].detach().clone().unsqueeze(0)
                    else:
                        K_all = torch.cat((K_all, K0[i].detach().clone().unsqueeze(0)), dim=0)
                        A_all = torch.cat((A_all, A0[i].detach().clone().unsqueeze(0)), dim=0)
                        p_all = torch.cat((p_all, p0[i].detach().clone().unsqueeze(0)), dim=0) 
                p_all = p_all / len(self.trained_task_id)            
            else:
                K_all = None
                A_all = None
                p_all = None
                K0 = getattr(self,f'e_k_divide_{l}')
                A0 = getattr(self,f'e_a_divide_{l}')
                p0 = getattr(self,f'e_p_divide_{l}')
                if "full" not in self.args.method:
                    global_task_id = self.task_id * self.num_clients + self.client_index
                    global_task_id = self.global_task_id_real[global_task_id]
                else:
                    if self.task_id == 0:
                        global_task_id = self.client_index
                    else:
                        global_task_id = self.task_id + 49
                
                #print(self.client_learned_global_task_id)
                for i in self.client_learned_global_task_id:
                    if i != global_task_id:
                        if K_all is None:
                            K_all = K0[i].detach().clone().unsqueeze(0)
                            A_all = A0[i].detach().clone().unsqueeze(0)
                            p_all = p0[i].detach().clone().unsqueeze(0)
                        else:
                            K_all = torch.cat((K_all, K0[i].detach().clone().unsqueeze(0)), dim=0)
                            A_all = torch.cat((A_all, A0[i].detach().clone().unsqueeze(0)), dim=0)
                            p_all = torch.cat((p_all, p0[i].detach().clone().unsqueeze(0)), dim=0)
                    else:
                        if K_all is None:
                            K_all = K0[global_task_id].unsqueeze(0)
                            A_all = A0[global_task_id].unsqueeze(0)
                            p_all = p0[global_task_id].unsqueeze(0)
                        else:
                            K_all = torch.cat((K_all, K0[global_task_id].unsqueeze(0)), dim=0)
                            A_all = torch.cat((A_all, A0[global_task_id].unsqueeze(0)), dim=0)
                            p_all = torch.cat((p_all, p0[global_task_id].unsqueeze(0)), dim=0)  
                #p_all = p_all / len(self.client_learned_global_task_id)
                p_all = p_all
            
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all.reshape(-1, C))
            n_K = nn.functional.normalize(K_all.reshape(-1, C), dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all.reshape(-1, self.e_p_length_2, C))
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None
            P_ = None
        if train:
            return p_return, 0, x_block, P_, None, None
        else:
            return p_return, 0, x_block, None

    
    def forward(self, x_querry, l, x_block, train=False, task_id=None, aq_k=None, x_divide=None):
        topk_com = self.topk_com
        if topk_com > len(self.trained_task_id):
            topk_com = len(self.trained_task_id)
        #print(getattr(self,f'e_k_{0}'))
        # e prompts
        e_valid = False
        indices_taskchoosing = None
        #print(l)
        if l in self.e_layers:
            #print(l)
            e_valid = True
            B, C = x_querry.shape

            #pt = int(self.e_pool_size / (self.n_tasks))
            #s = int(self.task_id * pt)
            #f = int((self.task_id + 1) * pt)\
            
            weight = self.weight.detach().clone()
            '''
            if (self.ep_g == 0 or self.ep_g == 1) and self.client_index == 3:
                print(weight[0])
                print(weight[1])
                print(weight[2])
                
                value, idx = self.fc_weight[10].topk(3)
                print(idx)
                print(value)
                value, idx = self.fc_weight[11].topk(3)
                print(idx)
                print(value)
                value, idx = self.fc_weight[12].topk(3)
                print(idx)
                print(value)
            '''   
            '''
            
            if self.client_index == -1:
                weight = self.weight.detach().clone()
            else:
                weight = self.weight.detach().clone()
                #global_task_id = self.task_id * self.num_clients + self.client_index
                #weight = torch.cat((self.weight[0:global_task_id].detach().clone(),self.weight[global_task_id].unsqueeze(0),self.weight[global_task_id:].detach().clone()), dim=0)
            #print(weight)
            weight = nn.functional.normalize(weight, dim=1)
            weight = torch.mm(weight, weight.T)
            '''
            
            
            '''
            #print(weight)
            weight_fake = torch.zeros((25, 25), device=self.device)
            
            for i in range(25):
                weight_fake[i][i] = 1.0

            #weight[0][0] = 0.5
            #weight[1][1] = 0.5
            #weight[0][1] = 0.5
            #weight[1][0] = 0.5

            
            #print(self.num_clients)
            relative_trained_task_id = {0: [[0], [1, 3]], 1: [[1], [2, 4]], 2: [[2], [0, 5]]}
            '''
            '''
            for i in [0, 1, 2, 3, 4]:
                for j in [0, 1, 2, 3, 4]:
                    weight[i][j] = 1.0
            '''
            '''
            for i in [0, 1, 5, 12, 23]:
                for j in [0, 1, 5, 12, 23]:
                    weight[i][j] = 1.0
            '''
            '''
            for i in range(50):
                for j in range(50):
                   weight[i][j] = 1.0 
            '''

            #weight = torch.einsum('bj,tj->bt', weight, weight)
            #weight = torch.div(weight, torch.max(torch.abs(weight)))
            #weight = torch.sigmoid(weight)
            #print(weight)
            #weight = torch.sigmoid(torch.div(weight + weight.T, 2))
            #weight = weight + torch.eye(self.e_task_number, device=self.device)
            #weight = torch.sigmoid(torch.cov(weight))
            #cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            #print(weight)
            #weight[self.not_trained_task_id, :] = 0
            #weight[:, self.not_trained_task_id] = 0
            if self.client_index == -1:
                '''
                
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                K1_representation = getattr(self,f'e_k_representation_{l}')[self.trained_task_id]
                K2_representation = getattr(self,f'e_k_representation_max_{l}')[self.trained_task_id]
                #K1_representation = torch.einsum('nkd,nb->bkd', K1_representation, weight[self.trained_task_id, :][:, self.trained_task_id]) 
                #print(self.trained_task_id)
                #print(weight[self.trained_task_id, :][:, self.trained_task_id].size())
                #print(torch.sum(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).size())
                K_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), K0[self.trained_task_id].reshape(K0[self.trained_task_id].shape[0], -1))
                K_client = K_client.reshape(-1, self.emb_d)
                A_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), A0[self.trained_task_id].reshape(A0[self.trained_task_id].shape[0], -1))
                A_client = A_client.reshape(-1, self.emb_d)
                p_client = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), p0[self.trained_task_id].reshape(p0[self.trained_task_id].shape[0], -1))
                p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                K_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), K1[self.trained_task_id].reshape(K1[self.trained_task_id].shape[0], -1))
                K_share = K_share.reshape(-1, self.emb_d)
                A_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), A1[self.trained_task_id].reshape(A1[self.trained_task_id].shape[0], -1))
                A_share = A_share.reshape(-1, self.emb_d)
                p_share = torch.mm(torch.mean(weight[self.trained_task_id, :][:, self.trained_task_id], dim=0).unsqueeze(0), p1[self.trained_task_id].reshape(p1[self.trained_task_id].shape[0], -1))
                p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                
                K_all = torch.cat((K_client, K_share), dim=0)
                A_all = torch.cat((A_client, A_share), dim=0)
                p_all = torch.cat((p_client, p_share), dim=0)
                '''
                
                
                K_all = None
                A_all = None
                p_all = None
                '''
                
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K_all = K0.reshape(-1, self.emb_d)
                A_all = A0.reshape(-1, self.emb_d)
                p_all = p0.reshape(-1, self.e_p_length, self.emb_d)
                '''

                
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                #K1_representation = getattr(self,f'e_k_representation_{l}')[self.trained_task_id]
                #_, idx = weight[global_task_id].topk(topk_com)
                #for i in idx:
                for i in self.trained_task_id:
                #for i in [global_task_id, global_task_id, global_task_id]:
                    trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                    trained_task_id_removed.remove(i)
                    K_share = torch.cat((K1[i].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[i].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[i].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                    if K_all is None:
                        K_all = K_share
                        A_all = A_share
                        p_all = p_share 
                    else:
                        K_all = torch.cat((K_all, K_share), dim=0)
                        A_all = torch.cat((A_all, A_share), dim=0)
                        p_all = torch.cat((p_all, p_share), dim=0)
                p_all = p_all / len(self.trained_task_id)
                
                '''
                K_all = K_share
                A_all = A_share
                p_all = p_share
                '''
            else:
                #K0 = getattr(self,f'e_k_specific_{l}')
                #A0 = getattr(self,f'e_a_specific_{l}')
                #p0 = getattr(self,f'e_p_specific_{l}')
                K1 = getattr(self,f'e_k_share_{l}')
                A1 = getattr(self,f'e_a_share_{l}')
                p1 = getattr(self,f'e_p_share_{l}')
                #K1_representation = getattr(self,f'e_k_representation_{l}')[[self.task_id * self.num_clients + self.client_index]]
                #print(self.client_learned_global_task_id)
                #K_client = K0[self.task_id * self.num_clients + self.client_index]
                #A_client = A0[self.task_id * self.num_clients + self.client_index]
                #p_client = p0[self.task_id * self.num_clients + self.client_index]

                if "full" not in self.args.method:
                    global_task_id = self.task_id * self.num_clients + self.client_index
                    global_task_id = self.global_task_id_real[global_task_id]
                else:
                    if self.task_id == 0:
                        global_task_id = self.client_index
                    else:
                        global_task_id = self.task_id + 49

                trained_task_id_removed = copy.deepcopy(self.trained_task_id)
                #trained_task_id_removed_forclient = copy.deepcopy(self.client_learned_global_task_id)
                trained_task_id_removed.remove(global_task_id)
                #trained_task_id_removed_forclient.remove(global_task_id)
                K_all = None
                A_all = None
                p_all = None
                '''
                K0 = getattr(self,f'e_k_specific_{l}')
                A0 = getattr(self,f'e_a_specific_{l}')
                p0 = getattr(self,f'e_p_specific_{l}')
                K0 = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed].detach().clone()), dim=0)
                A0 = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed].detach().clone()), dim=0)
                p0 = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed].detach().clone()), dim=0)
                K_all = K0.reshape(-1, self.emb_d)
                A_all = A0.reshape(-1, self.emb_d)
                p_all = p0.reshape(-1, self.e_p_length, self.emb_d)
                '''
                
                if train:
                    #K_share = K1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    #A_share = A1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    #p_share = p1 * self.weight[self.task_id * self.num_clients + self.client_index]
                    
                    '''
                    if global_task_id == 0:
                        K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[global_task_id + 1:]), dim=0)
                        A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[global_task_id + 1:]), dim=0)
                        p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[global_task_id + 1:]), dim=0)
                    elif global_task_id == (self.e_task_number - 1):
                        K_share = torch.cat((K1[0:global_task_id],K1[global_task_id].unsqueeze(0)), dim=0)
                        A_share = torch.cat((A1[0:global_task_id],A1[global_task_id].unsqueeze(0)), dim=0)
                        p_share = torch.cat((p1[0:global_task_id],p1[global_task_id].unsqueeze(0)), dim=0)
                    else:
                        K_share = torch.cat((K1[0:global_task_id],K1[global_task_id].unsqueeze(0),K1[global_task_id + 1:]), dim=0)
                        A_share = torch.cat((A1[0:global_task_id],A1[global_task_id].unsqueeze(0),A1[global_task_id + 1:]), dim=0)
                        p_share = torch.cat((p1[0:global_task_id],p1[global_task_id].unsqueeze(0),p1[global_task_id + 1:]), dim=0)
                    '''
                    '''
                    K_client = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    A_client = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    p_client = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed_forclient].detach().clone()), dim=0)
                    weight_client = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed_forclient]), dim=0).unsqueeze(0)
                    
                    K_client = torch.mm(weight_client, K_client.reshape(K_client.shape[0], -1))
                    K_client = K_client.reshape(-1, self.emb_d)
                    A_client = torch.mm(weight_client, A_client.reshape(A_client.shape[0], -1))
                    A_client = A_client.reshape(-1, self.emb_d)
                    p_client = torch.mm(weight_client, p_client.reshape(p_client.shape[0], -1))
                    p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed].detach().clone()), dim=0)
                    A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed].detach().clone()), dim=0)
                    p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed].detach().clone()), dim=0)
                    weight_share = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed]), dim=0).unsqueeze(0)
                    
                    
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_all = torch.cat((K_client, K_share), dim=0) 
                    A_all = torch.cat((A_client, A_share), dim=0)
                    p_all = torch.cat((p_client, p_share), dim=0)
                    '''
                    
                    
                    #for i in relative_trained_task_id[self.client_index][self.task_id]:
                    _, idx = weight[global_task_id][self.trained_task_id].topk(topk_com)
                    idx = [self.trained_task_id[id] for id in idx]
                    if '_prompt1' in self.args.method:
                        weight = torch.zeros((int(self.args.numclass/self.args.class_per_task), int(self.args.numclass/self.args.class_per_task)), device=self.device)
                        for i in range(int(self.args.numclass/self.args.class_per_task)):
                            weight[i][i] = 1.0
                    elif '_prompt2' in self.args.method:
                        weight = torch.zeros((int(self.args.numclass/self.args.class_per_task), int(self.args.numclass/self.args.class_per_task)), device=self.device)
                        for i in range(int(self.args.numclass/self.args.class_per_task)):
                            weight[i][i] = 1.0
                        idx = [global_task_id, global_task_id, global_task_id]

                    
                
                    K_global_task = K1[global_task_id].unsqueeze(0)
                    A_global_task = A1[global_task_id].unsqueeze(0)
                    p_global_task = p1[global_task_id].unsqueeze(0)
                    #print(idx)
                    for i in idx:
                    #for i in self.trained_task_id_forchoosing:
                        #print(self.trained_task_id)
                    #for i in [global_task_id, global_task_id, global_task_id]:
                        if i != global_task_id:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            trained_task_id_removed_again.remove(i)
                            if "CLpartprompt" in self.args.method and i == 1:
                                K_global_task_other = torch.cat((K1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                A_global_task_other = torch.cat((A1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                p_global_task_other = torch.cat((p1[i][0:2],torch.zeros((8,8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                K_share = torch.cat((K_global_task,K_global_task_other,K1[trained_task_id_removed_again]), dim=0)
                                A_share = torch.cat((A_global_task,A_global_task_other,A1[trained_task_id_removed_again]), dim=0)
                                p_share = torch.cat((p_global_task,p_global_task_other,p1[trained_task_id_removed_again]), dim=0)
                            else:
                                K_share = torch.cat((K_global_task,K1[i].detach().clone().unsqueeze(0),K1[trained_task_id_removed_again].detach().clone()), dim=0)
                                A_share = torch.cat((A_global_task,A1[i].detach().clone().unsqueeze(0),A1[trained_task_id_removed_again].detach().clone()), dim=0)
                                p_share = torch.cat((p_global_task,p1[i].detach().clone().unsqueeze(0),p1[trained_task_id_removed_again].detach().clone()), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                        else:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            if "CLpartprompt" in self.args.method and i == 1:
                                K_global_task_other = torch.cat((K1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                A_global_task_other = torch.cat((A1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                p_global_task_other = torch.cat((p1[i][0:2],torch.zeros((8,8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                K_share = torch.cat((K_global_task_other,K1[trained_task_id_removed_again]), dim=0)
                                A_share = torch.cat((A_global_task_other,A1[trained_task_id_removed_again]), dim=0)
                                p_share = torch.cat((p_global_task_other,p1[trained_task_id_removed_again]), dim=0)
                            else:
                                K_share = torch.cat((K_global_task,K1[trained_task_id_removed_again].detach().clone()), dim=0)
                                A_share = torch.cat((A_global_task,A1[trained_task_id_removed_again].detach().clone()), dim=0)
                                p_share = torch.cat((p_global_task,p1[trained_task_id_removed_again].detach().clone()), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                    p_all = p_all / topk_com
                    #p_all = p_all / 3
                    
                    '''
                    K_all = K_share
                    A_all = A_share
                    p_all = p_share
                    '''
                    
                    
                else:
                    '''
                    K_client = torch.cat((K0[global_task_id].unsqueeze(0),K0[trained_task_id_removed_forclient]), dim=0)
                    A_client = torch.cat((A0[global_task_id].unsqueeze(0),A0[trained_task_id_removed_forclient]), dim=0)
                    p_client = torch.cat((p0[global_task_id].unsqueeze(0),p0[trained_task_id_removed_forclient]), dim=0)
                    weight_client = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed_forclient]), dim=0).unsqueeze(0)
                    
                    K_client = torch.mm(weight_client, K_client.reshape(K_client.shape[0], -1))
                    K_client = K_client.reshape(-1, self.emb_d)
                    A_client = torch.mm(weight_client, A_client.reshape(A_client.shape[0], -1))
                    A_client = A_client.reshape(-1, self.emb_d)
                    p_client = torch.mm(weight_client, p_client.reshape(p_client.shape[0], -1))
                    p_client = p_client.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                    A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                    p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                    weight_share = torch.cat((weight[global_task_id][global_task_id].unsqueeze(0), weight[global_task_id][trained_task_id_removed]), dim=0).unsqueeze(0)
                    K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                    K_share = K_share.reshape(-1, self.emb_d)
                    A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                    A_share = A_share.reshape(-1, self.emb_d)
                    p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                    p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                    '''
                    '''
                    K_all = torch.cat((K_client, K_share), dim=0)
                    A_all = torch.cat((A_client, A_share), dim=0)
                    p_all = torch.cat((p_client, p_share), dim=0)
                    '''
                    
                    K_global_task = K1[global_task_id].unsqueeze(0)
                    A_global_task = A1[global_task_id].unsqueeze(0)
                    p_global_task = p1[global_task_id].unsqueeze(0)
                    #print(idx)
                    #for i in relative_trained_task_id[self.client_index][self.task_id]:
                    _, idx = weight[global_task_id][self.trained_task_id].topk(topk_com)
                    idx = [self.trained_task_id[id] for id in idx]
                    #print(idx)
                    for i in idx:
                    #for i in self.trained_task_id_forchoosing:
                    #for i in [global_task_id, global_task_id, global_task_id]:
                        if i != global_task_id:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            trained_task_id_removed_again.remove(i)
                            if "CLpartprompt" in self.args.method and i == 1:
                                K_global_task_other = torch.cat((K1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                A_global_task_other = torch.cat((A1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                p_global_task_other = torch.cat((p1[i][0:2],torch.zeros((8,8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                K_share = torch.cat((K_global_task,K_global_task_other,K1[trained_task_id_removed_again]), dim=0)
                                A_share = torch.cat((A_global_task,A_global_task_other,A1[trained_task_id_removed_again]), dim=0)
                                p_share = torch.cat((p_global_task,p_global_task_other,p1[trained_task_id_removed_again]), dim=0)
                            else:
                                K_share = torch.cat((K_global_task,K1[i].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                                A_share = torch.cat((A_global_task,A1[i].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                                p_share = torch.cat((p_global_task,p1[i].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                        else:
                            trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                            if "CLpartprompt" in self.args.method and i == 1:
                                K_global_task_other = torch.cat((K1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                A_global_task_other = torch.cat((A1[i][0:2],torch.zeros((8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                p_global_task_other = torch.cat((p1[i][0:2],torch.zeros((8,8,768), device=self.device).detach().clone()), dim=0).unsqueeze(0)
                                K_share = torch.cat((K_global_task_other,K1[trained_task_id_removed_again]), dim=0)
                                A_share = torch.cat((A_global_task_other,A1[trained_task_id_removed_again]), dim=0)
                                p_share = torch.cat((p_global_task_other,p1[trained_task_id_removed_again]), dim=0)
                            else:
                                K_share = torch.cat((K_global_task,K1[trained_task_id_removed_again]), dim=0)
                                A_share = torch.cat((A_global_task,A1[trained_task_id_removed_again]), dim=0)
                                p_share = torch.cat((p_global_task,p1[trained_task_id_removed_again]), dim=0)
                            weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                            K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                            K_share = K_share.reshape(-1, self.emb_d)
                            A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                            A_share = A_share.reshape(-1, self.emb_d)
                            p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                            p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)  
                            if K_all is None:
                                K_all = K_share
                                A_all = A_share
                                p_all = p_share 
                            else:
                                K_all = torch.cat((K_all, K_share), dim=0)
                                A_all = torch.cat((A_all, A_share), dim=0)
                                p_all = torch.cat((p_all, p_share), dim=0)
                    p_all = p_all / topk_com
                    #p_all = p_all / 3

                    '''
                    K_all = K_share
                    A_all = A_share
                    p_all = p_share
                    '''

            #TODO: task choosing


            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A_all) 
            #a_querry_client = torch.einsum('bd,kd->bkd', x_querry, A_client)
            #a_querry_share = torch.einsum('bd,kd->bkd', x_querry, A_share)
            #a_querry_share = x_querry.unsqueeze(1).repeat(1, A_share.shape[0], 1)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K_all, dim=1)
            
            #n_K_share_taskchoosing = nn.functional.normalize(K1, dim=1)
            #n_K_share_taskchoosing = nn.functional.normalize(K1_representation, dim=2)
            
            q = nn.functional.normalize(a_querry, dim=2)
            #q_client = nn.functional.normalize(a_querry_client, dim=2)
            #q_share = nn.functional.normalize(a_querry_share, dim=2)
            if aq_k is None:
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                #aq_k_client = torch.einsum('bkd,kd->bk', q_client, n_K_client)
                #aq_k_share = torch.einsum('bkd,kd->bk', q_share, n_K_share)
                #aq_k_share_taskchoosing = torch.einsum('bd,nd->bn', nn.functional.normalize(x_querry, dim=1), n_K_share_taskchoosing)
                #print(K1_representation.size())
                #x_querry_mean = torch.mean(x_querry, dim=0).unsqueeze(0).repeat(x_querry.shape[0], 1)
                #aq_k_share_taskchoosing = torch.einsum('bkd,nkd->bnk', nn.functional.normalize(x_querry_mean.unsqueeze(1).repeat(1, n_K_share_taskchoosing.shape[1], 1), dim=2), n_K_share_taskchoosing)
                #aq_k_share_taskchoosing = torch.mean(aq_k_share_taskchoosing, dim=2)
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, global_task_id])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, :])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, 31])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, 13])
                #print(torch.mean(aq_k_share_taskchoosing, dim=2)[0, 44])
                #print(aq_k_share_taskchoosing.size())

                #aq_k_share_taskchoosing = aq_k_share_taskchoosing - 0.1 * K2_representation.unsqueeze(0).repeat(aq_k_share_taskchoosing.shape[0], 1, 1) 
                #aq_k_share_taskchoosing, _ = torch.max(aq_k_share_taskchoosing, dim=2)
                #print(aq_k_share_taskchoosing.size())
                #aq_k_share_taskchoosing = aq_k_share_taskchoosing
                #topk_for_taskchoosing = torch.topk(aq_k_share_taskchoosing, 1, dim=1)
                #K2_representation[topk_for_taskchoosing.indices]
                #topk_for_taskchoosing = torch.topk(aq_k_share_taskchoosing, 5, dim=1)
                #indices_taskchoosing = topk_for_taskchoosing.indices
                #for i in range(indices_taskchoosing.shape[0]):
                    #for j in range(indices_taskchoosing.shape[1]):
                        #indices_taskchoosing[i][j] = self.trained_task_id[int(indices_taskchoosing[i][j].cpu().numpy())]
                #print(indices_taskchoosing[0])
                #aq_k_all = torch.cat((aq_k_client, aq_k_share), dim=1)
                
                #aq_k = torch.softmax(aq_k, dim=1)
                '''
                if train and self.ep_g is not None and (self.ep_g + 1) % 3 == 0:
                    weight_matrix = aq_k.clone().detach().cpu().numpy()
                    fig, ax = plt.subplots()
                    im = ax.imshow(weight_matrix, cmap='Blues')
                    fig.tight_layout()
                    plt.xticks([])
                    plt.yticks([])
                    plt.colorbar(im)
                    plt.savefig('/home/piaohongming/FCL/Baselines/src/Picture/Prompt_map_wp/{}_{}_{}.png'.format(self.client_index, l, self.ep_g), dpi=300)
                    plt.close()
                '''
            else:
                aq_k = aq_k[l]
                #print(aq_k.size())
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p_all)
            #P_ = torch.einsum('bk,kld->bld', aq_k_all, p_all)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            loss = 0
            
            #K_client_nor = nn.functional.normalize(K_client, dim=1)
            #K_share_nor = nn.functional.normalize(K_share, dim=1)
            #loss = (1.0 - torch.abs(aq_k_share_taskchoosing.mean())) * 1
            
            '''
            q_log_prob = torch.log(torch.mean(nn.functional.normalize(x_querry, dim=1), dim=0))
            p_log_prob = torch.log(torch.mean(n_K_share_specific, dim=0))
            print(q_log_prob)
            print(p_log_prob)
            loss = F.kl_div(p_log_prob, q_log_prob, reduction='sum')
            '''
            #loss += self.ortho_penalty(K_client_nor)
            #loss += self.ortho_penalty(K_share_nor)
            
            # 4 = ablate p, 5 = ablate a, 6 = ablate k
            if train and "classincremental" in self.args.method:
                global_task_id = self.task_id * self.num_clients + self.client_index
                global_task_id = self.global_task_id_real[global_task_id]
                K_all = None
                A_all = None
                p_all = None
                K0 = getattr(self,f'e_k_divide_{l}')
                A0 = getattr(self,f'e_a_divide_{l}')
                p0 = getattr(self,f'e_p_divide_{l}')
                for i in self.client_learned_global_task_id:
                    if i != global_task_id:
                        if K_all is None:
                            K_all = K0[i].detach().clone().unsqueeze(0)
                            A_all = A0[i].detach().clone().unsqueeze(0)
                            p_all = p0[i].detach().clone().unsqueeze(0)
                        else:
                            K_all = torch.cat((K_all, K0[i].detach().clone().unsqueeze(0)), dim=0)
                            A_all = torch.cat((A_all, A0[i].detach().clone().unsqueeze(0)), dim=0)
                            p_all = torch.cat((p_all, p0[i].detach().clone().unsqueeze(0)), dim=0)
                    else:
                        if K_all is None:
                            K_all = K0[global_task_id].unsqueeze(0)
                            A_all = A0[global_task_id].unsqueeze(0)
                            p_all = p0[global_task_id].unsqueeze(0)
                        else:
                            K_all = torch.cat((K_all, K0[global_task_id].unsqueeze(0)), dim=0)
                            A_all = torch.cat((A_all, A0[global_task_id].unsqueeze(0)), dim=0)
                            p_all = torch.cat((p_all, p0[global_task_id].unsqueeze(0)), dim=0)  
                
                loss = self.ortho_penalty(K_all.reshape(-1, C))
                loss += self.ortho_penalty(A_all.reshape(-1, C))
                loss += self.ortho_penalty(p_all.reshape(-1, self.e_p_length_2 * C))
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
        else:
            p_return = None
            P_ = None
            indices_taskchoosing = None

        # return
        if train:
            if aq_k is not None:
                return p_return, loss, x_block, P_, indices_taskchoosing, torch.mean(aq_k, dim=0)
            else:
                return p_return, loss, x_block, P_, indices_taskchoosing, None
        else:
            return p_return, 0, x_block, indices_taskchoosing

    def get_weight(self):
        
        weight = self.weight.detach().clone()
        weight = nn.functional.normalize(weight, dim=1)
        weight = torch.mm(weight, weight.T)
        #weight = torch.einsum('bj,tj->bt', weight, weight)
        #weight = torch.div(weight, torch.max(torch.abs(weight)))
        #weight = torch.sigmoid(weight)
        #print(weight)
        #weight = torch.sigmoid(torch.div(weight + weight.T, 2))
        return weight.cpu().numpy()
        
        '''
        K0 = getattr(self,f'e_k_specific_0')
        K0 = K0.detach().clone()
        K0 = K0.view(K0.shape[0], -1)
        weight = nn.functional.normalize(K0, dim=1)
        print(weight.size())
        weight = torch.mm(weight, weight.T)
        print(weight.size())
        return weight.cpu().numpy()[0:10, 0:10]
        '''

        
    def reorder_prompt(self, idx):
        global_task_id = self.task_id * self.num_clients + self.client_index
        for l in self.e_layers:
            
            K0 = getattr(self,f'e_k_specific_{l}')
            A0 = getattr(self,f'e_a_specific_{l}')
            p0 = getattr(self,f'e_p_specific_{l}')
            if global_task_id == 0:
                K0 = torch.cat((K0[global_task_id][idx[2 * l]].unsqueeze(0), K0[global_task_id + 1: ]), dim=0)
                A0 = torch.cat((A0[global_task_id][idx[2 * l]].unsqueeze(0), A0[global_task_id + 1: ]), dim=0)
                p0 = torch.cat((p0[global_task_id][idx[2 * l]].unsqueeze(0), p0[global_task_id + 1: ]), dim=0)
            elif global_task_id == (self.e_task_number - 1):
                K0 = torch.cat((K0[0: global_task_id], K0[global_task_id][idx[2 * l]].unsqueeze(0)), dim=0)
                A0 = torch.cat((A0[0: global_task_id], A0[global_task_id][idx[2 * l]].unsqueeze(0)), dim=0)
                p0 = torch.cat((p0[0: global_task_id], p0[global_task_id][idx[2 * l]].unsqueeze(0)), dim=0)
            else:
                #print(K0[global_task_id][idx[2 * l]].unsqueeze(0).size())
                #print(idx[2 * l])
                K0 = torch.cat((K0[0: global_task_id], K0[global_task_id][idx[2 * l]].unsqueeze(0), K0[global_task_id + 1: ]), dim=0)
                A0 = torch.cat((A0[0: global_task_id], A0[global_task_id][idx[2 * l]].unsqueeze(0), A0[global_task_id + 1: ]), dim=0)
                p0 = torch.cat((p0[0: global_task_id], p0[global_task_id][idx[2 * l]].unsqueeze(0), p0[global_task_id + 1: ]), dim=0)
            setattr(self, f'e_k_specific_{l}',torch.nn.Parameter(K0))
            setattr(self, f'e_a_specific_{l}',torch.nn.Parameter(A0))
            setattr(self, f'e_p_specific_{l}',torch.nn.Parameter(p0))
            K1 = getattr(self,f'e_k_share_{l}')
            A1 = getattr(self,f'e_a_share_{l}')
            p1 = getattr(self,f'e_p_share_{l}')
            if global_task_id == 0:
                K1 = torch.cat((K1[global_task_id][idx[2 * l + 1]].unsqueeze(0), K1[global_task_id + 1: ]), dim=0)
                A1 = torch.cat((A1[global_task_id][idx[2 * l + 1]].unsqueeze(0), A1[global_task_id + 1: ]), dim=0)
                p1 = torch.cat((p1[global_task_id][idx[2 * l + 1]].unsqueeze(0), p1[global_task_id + 1: ]), dim=0)
            elif global_task_id == (self.e_task_number - 1):
                K1 = torch.cat((K1[0: global_task_id], K1[global_task_id][idx[2 * l + 1]].unsqueeze(0)), dim=0)
                A1 = torch.cat((A1[0: global_task_id], A1[global_task_id][idx[2 * l + 1]].unsqueeze(0)), dim=0)
                p1 = torch.cat((p1[0: global_task_id], p1[global_task_id][idx[2 * l + 1]].unsqueeze(0)), dim=0)
            else:
                K1 = torch.cat((K1[0: global_task_id], K1[global_task_id][idx[2 * l + 1]].unsqueeze(0), K1[global_task_id + 1: ]), dim=0)
                A1 = torch.cat((A1[0: global_task_id], A1[global_task_id][idx[2 * l + 1]].unsqueeze(0), A1[global_task_id + 1: ]), dim=0)
                p1 = torch.cat((p1[0: global_task_id], p1[global_task_id][idx[2 * l + 1]].unsqueeze(0), p1[global_task_id + 1: ]), dim=0)
            setattr(self, f'e_k_share_{l}',torch.nn.Parameter(K1))
            setattr(self, f'e_a_share_{l}',torch.nn.Parameter(A1))
            setattr(self, f'e_p_share_{l}',torch.nn.Parameter(p1))
            
            '''
            K0 = getattr(self,f'e_k_specific_{l}')
            A0 = getattr(self,f'e_a_specific_{l}')
            p0 = getattr(self,f'e_p_specific_{l}')
            if global_task_id == 0:
                K0 = torch.cat((K0[global_task_id][idx[l]].unsqueeze(0), K0[global_task_id + 1: ]), dim=0)
                A0 = torch.cat((A0[global_task_id][idx[l]].unsqueeze(0), A0[global_task_id + 1: ]), dim=0)
                p0 = torch.cat((p0[global_task_id][idx[l]].unsqueeze(0), p0[global_task_id + 1: ]), dim=0)
            elif global_task_id == (self.e_task_number - 1):
                K0 = torch.cat((K0[0: global_task_id], K0[global_task_id][idx[l]].unsqueeze(0)), dim=0)
                A0 = torch.cat((A0[0: global_task_id], A0[global_task_id][idx[l]].unsqueeze(0)), dim=0)
                p0 = torch.cat((p0[0: global_task_id], p0[global_task_id][idx[l]].unsqueeze(0)), dim=0)
            else:
                #print(K0[global_task_id][idx[2 * l]].unsqueeze(0).size())
                #print(idx[2 * l])
                K0 = torch.cat((K0[0: global_task_id], K0[global_task_id][idx[l]].unsqueeze(0), K0[global_task_id + 1: ]), dim=0)
                A0 = torch.cat((A0[0: global_task_id], A0[global_task_id][idx[l]].unsqueeze(0), A0[global_task_id + 1: ]), dim=0)
                p0 = torch.cat((p0[0: global_task_id], p0[global_task_id][idx[l]].unsqueeze(0), p0[global_task_id + 1: ]), dim=0)
            setattr(self, f'e_k_specific_{l}',torch.nn.Parameter(K0))
            setattr(self, f'e_a_specific_{l}',torch.nn.Parameter(A0))
            setattr(self, f'e_p_specific_{l}',torch.nn.Parameter(p0))
            '''
        
    def get_previous(self, taskid_local, client_index, idxs, pre=None):
        P = []
        #weight = self.weight + 0
        #weight = torch.div(weight + weight.T, 2)
        #weight = torch.sigmoid(torch.div(weight + weight.T, 2))
        #weight = weight + torch.eye(self.e_task_number, device=self.device)
        weight = self.weight
        weight = nn.functional.normalize(weight, dim=1)
        #weight = torch.einsum('bj,tj->bt', weight, weight)
        weight = torch.mm(weight, weight.T)
        #weight = torch.div(weight, torch.max(torch.abs(weight)))
        #weight = torch.sigmoid(weight)
        #weight = torch.sigmoid(torch.cov(weight))
        for i in range(len(taskid_local)):
            global_task_id = taskid_local[i] * self.num_clients + client_index[i]
            trained_task_id_removed = copy.deepcopy(self.trained_task_id)
            trained_task_id_removed.remove(global_task_id)
            for l in self.e_layers:
                K1 = getattr(self,f'e_k_share_{l}')
                if pre is None:
                    K1 = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed]), dim=0)
                else:
                    K1_pre = getattr(pre,f'e_k_share_{l}')
                    K1 = torch.cat((K1[global_task_id].unsqueeze(0),K1_pre[trained_task_id_removed]), dim=0)
                K_all = None
                #for i in self.trained_task_id:
                for i in [global_task_id]:
                    if i != global_task_id:
                        trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                        trained_task_id_removed_again.remove(i)
                        K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[i].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                        weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                        K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                        K_share = K_share.reshape(-1, self.emb_d)
                        if K_all is None:
                            K_all = K_share
                        else:
                            K_all = torch.cat((K_all, K_share), dim=0)
                    else:
                        trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                        K_share = torch.cat((K1[global_task_id].unsqueeze(0),K1[trained_task_id_removed_again]), dim=0)
                        weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                        K_share = torch.mm(weight_share, K_share.reshape(K_share.shape[0], -1))
                        K_share = K_share.reshape(-1, self.emb_d)
                        if K_all is None:
                            K_all = K_share
                        else:
                            K_all = torch.cat((K_all, K_share), dim=0)
                P.append(K_all)
            for l in self.e_layers:
                A1 = getattr(self,f'e_a_share_{l}')
                if pre is None:
                    A1 = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed]), dim=0)
                else:
                    A1_pre = getattr(pre,f'e_a_share_{l}')
                    A1 = torch.cat((A1[global_task_id].unsqueeze(0),A1_pre[trained_task_id_removed]), dim=0)
                A_all = None
                #for i in self.trained_task_id:
                for i in [global_task_id]:
                    if i != global_task_id:
                        trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                        trained_task_id_removed_again.remove(i)
                        A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[i].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                        weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                        A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                        A_share = A_share.reshape(-1, self.emb_d)
                        if A_all is None:
                            A_all = A_share
                        else:
                            A_all = torch.cat((A_all, A_share), dim=0)
                    else:
                        trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                        A_share = torch.cat((A1[global_task_id].unsqueeze(0),A1[trained_task_id_removed_again]), dim=0)
                        weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                        A_share = torch.mm(weight_share, A_share.reshape(A_share.shape[0], -1))
                        A_share = A_share.reshape(-1, self.emb_d)
                        if A_all is None:
                            A_all = A_share
                        else:
                            A_all = torch.cat((A_all, A_share), dim=0)
                P.append(A_all)
            for l in self.e_layers:
                p1 = getattr(self,f'e_p_share_{l}')
                if pre is None:
                    p1 = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed]), dim=0)
                else:
                    p1_pre = getattr(pre,f'e_p_share_{l}')
                    p1 = torch.cat((p1[global_task_id].unsqueeze(0),p1_pre[trained_task_id_removed]), dim=0)
                p_all = None
                #for i in self.trained_task_id:
                for i in [global_task_id]:
                    if i != global_task_id:
                        trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                        trained_task_id_removed_again.remove(i)
                        p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[i].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                        weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][i].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                        p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                        p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                        if p_all is None:
                            p_all = p_share
                        else:
                            p_all = torch.cat((p_all, p_share), dim=0)
                    else:
                        trained_task_id_removed_again = copy.deepcopy(trained_task_id_removed)
                        p_share = torch.cat((p1[global_task_id].unsqueeze(0),p1[trained_task_id_removed_again]), dim=0)
                        weight_share = torch.cat((weight[i][global_task_id].unsqueeze(0), weight[i][trained_task_id_removed_again]), dim=0).unsqueeze(0)
                        p_share = torch.mm(weight_share, p_share.reshape(p_share.shape[0], -1))
                        p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                        if p_all is None:
                            p_all = p_share
                        else:
                            p_all = torch.cat((p_all, p_share), dim=0)
                P.append(p_all)
                #p_share = p_share.reshape(-1, self.e_p_length, self.emb_d)
                
        return P
    
    def getClientPrompt(self, client_learned_task_id, class_distribution):
        prompt_pool = []
        prompt_pool_layer = []
        prompt_pool_label = []
        for task_id in client_learned_task_id:
            for l in self.e_layers:
                p0 = getattr(self,f'e_p_specific_{l}')
                p_specific = p0[task_id * self.num_clients + self.client_index]
                for i in range(self.e_pool_size_0): 
                    for j in range(int(self.e_p_length / 2)):
                        prompt_pool.append(p_specific[i][(int(self.e_p_length / 2)):self.e_p_length][j])
                        prompt_pool_layer.append(l)
                        prompt_pool_label.append(class_distribution[self.client_index][task_id])
                '''
                p1 = getattr(self,f'e_p_share_{l}')
                p_share = p1[task_id * self.num_clients + self.client_index]
                for i in range(self.e_pool_size_1): 
                    prompt_pool.append(p_share[i][(self.e_p_length / 2):self.e_p_length])
                    prompt_pool_layer.append(l)
                    prompt_pool_label.append(class_distribution[self.client_index][task_id])
                '''
        return prompt_pool, prompt_pool_layer, prompt_pool_label
    
    def get_K_penalty(self, task):
        K_penalty = []
        for l in self.e_layers:
            K = getattr(self,f'e_k_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            f = int((task + 1) * pt)
            K = K[0:f]
            loss = self.ortho_penalty(K)
            K_penalty.append(loss.cpu().detach().numpy())
        return K_penalty
    
    def get_A_penalty(self, task):
        A_penalty = []
        for l in self.e_layers:
            A = getattr(self,f'e_a_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            f = int((task + 1) * pt)
            A = A[0:f]
            loss = self.ortho_penalty(A)
            A_penalty.append(loss.cpu().detach().numpy())
        return A_penalty
    
    def get_P_penalty(self, task):
        P_penalty = []
        for l in self.e_layers:
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            f = int((task + 1) * pt)
            p = p[0:f]
            loss = self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
            P_penalty.append(loss.cpu().detach().numpy())
        return P_penalty
    
    def getPrompt(self, client=0):
        classes = []
        prompts = []
        #pt = int(self.e_pool_size / (self.n_tasks))
        #print(self.n_tasks)
        for l in range(1):
        #for l in self.e_layers:
            p = getattr(self,f'e_k_share_{l}')
            p = p.view(p.shape[0], -1)
            for i in range(10):
                #print(p.size())
                prompts.extend(p[i].squeeze().unsqueeze(0).cpu().detach().numpy())
                #prompts.extend(p[i, 1, 0:4, :].squeeze().cpu().detach().numpy())
                #prompts.extend(p[i, 2, 0:4, :].squeeze().cpu().detach().numpy())
                #prompts.extend(p[i, 3, 0:4, :].squeeze().cpu().detach().numpy())
                #prompts.extend(p[i, 4, 0:4, :].squeeze().cpu().detach().numpy())
                classes.extend(np.full(p[i].squeeze().unsqueeze(0).shape[0], i))
                #classes.extend(np.full(p[i, 1, 0:4, :].squeeze().shape[0], i))
                #classes.extend(np.full(p[i, 2, 0:4, :].squeeze().shape[0], i))
                #classes.extend(np.full(p[i, 3, 0:4, :].squeeze().shape[0], i))
                #classes.extend(np.full(p[i, 4, 0:4, :].squeeze().shape[0], i))
        return prompts, classes
    
    def getK(self):
        classes = []
        Ks = []
        pt = int(self.e_pool_size / (self.n_tasks))
        print(self.n_tasks)
        for l in range(1):
        #for l in self.e_layers:
            K = getattr(self,f'e_k_{l}')
            for i in range(5):
                s = int(i * pt)
                f = int((i + 1) * pt)
                Ks.extend(K[s:f].reshape(-1, 768).cpu().detach().numpy())
                classes.extend(np.full(K[s:f].reshape(-1, 768).shape[0], l * self.n_tasks + i))
        return Ks, classes
    
    def getA(self):
        classes = []
        As = []
        pt = int(self.e_pool_size / (self.n_tasks))
        print(self.n_tasks)
        for l in range(1):
        #for l in self.e_layers:
            A = getattr(self,f'e_a_{l}')
            for i in range(5):
                s = int(i * pt)
                f = int((i + 1) * pt)
                As.extend(A[s:f].reshape(-1, 768).cpu().detach().numpy())
                classes.extend(np.full(A[s:f].reshape(-1, 768).shape[0], l * self.n_tasks + i))
        return As, classes
                

    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda(self.device))**2).mean() * 10000.0


class CodaPrompt_weight(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, device='cuda:0', clients_local=10, num_clients = 10):
        print(" in CODA prompt")
        super().__init__()
        self.task_id = 0
        self.max_classes = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        self.counter = 0
        self.next_task_locs= None
        self.device = device
        self.task_count_f = 0
        self.clients_local = clients_local
        self.num_clients = num_clients
        self.client_index = -1

        
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance
            e_l = self.e_p_length
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)
            for i in range(self.clients_local):
                setattr(self, f'e_p_{e}_{i}',torch.nn.Parameter(p.data))
                setattr(self, f'e_k_{e}_{i}',torch.nn.Parameter(k.data))
                setattr(self, f'e_a_{e}_{i}',torch.nn.Parameter(a.data))


        for i in range(self.num_clients):
            for t in range(self.n_tasks):
                weight = torch.nn.Parameter(torch.ones(self.clients_local), requires_grad=True) 
                setattr(self, f'weight_{i}_{t}',weight)

           
    #def set_the_weight():

    
    
    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = prompt_param[0]
        self.e_p_length = prompt_param[1]

        # prompt locations
        self.e_layers = [0,1,2,3,4]

        # location of ortho penalty
        self.ortho_mu = prompt_param[3]
        print("ortho_mu ", self.ortho_mu)

        # ablations
        '''
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
        '''
        
    def process_frequency(self, next_task_locs = None):
        for i in range(self.num_clients):
            for t in range(self.n_tasks):
                if isinstance(self.device, int):
                    weight = torch.nn.Parameter(torch.ones(self.clients_local).cuda(self.device), requires_grad=True) 
                else:
                    weight = torch.nn.Parameter(torch.ones(self.clients_local).cuda(), requires_grad=True)
                setattr(self, f'weight_{i}_{t}',weight)
        '''
        if self.task_id > 0:
            self.weight_previous_prompt()
        self.task_count_f += 1
        '''
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_id * pt)
        f = int((self.task_id + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            piao = 0
            while redo and piao < 20:
                piao = piao + 1
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
        
        return torch.nn.Parameter(uu) 
        
        # print("prceoss freq changed task_count to ", self.task_count_f)
    def next_task_locs_f(self, next_task_locs = None):
        if next_task_locs:
            self.next_task_locs = next_task_locs

    def weight_previous_prompt(self):
        for l in self.e_layers:
            K_list = []
            A_list = []
            P_list = []
            for i in range(self.clients_local):
                K_client = getattr(self,f'e_k_{l}_{i}')
                A_client = getattr(self,f'e_a_{l}_{i}')
                P_client = getattr(self,f'e_p_{l}_{i}')
                K_list.append(K_client.detach())
                A_list.append(A_client.detach())
                P_list.append(P_client.detach())
                #print(K_client.detach().unsqueeze(0).size())

            
            K = getattr(self,f'e_k_{l}').detach()
            A = getattr(self,f'e_a_{l}').detach()
            P = getattr(self,f'e_p_{l}').detach()

            pt = int(self.e_pool_size / (self.n_tasks))

            for t in range(self.task_id):
                s = int(t * pt)
                f = int((t + 1) * pt)
                if self.client_index >= 0:
                    weight = getattr(self,f'weight_{self.client_index}_{t}')
                    #weight = [1,1,1,1,1,1,1,1,1,1]
                else:
                    weight = getattr(self,f'weight_{0}_{t}')
                    #weight = [1,1,1,1,1,1,1,1,1,1]

                weight = weight / weight.sum()
                #print(weight)
                #weight = weight / sum(weight)
                #print(weight.size())
                #print(weight.squeeze().unsqueeze(0).size())
                #print(torch.stack(K_list, dim=0).transpose(0,1).size())
                K_weight = torch.matmul(torch.stack(K_list, dim=0).permute(1,2,0), weight.squeeze().unsqueeze(1)).squeeze()
                #print(K_weight[0][0].cpu().numpy())
                A_weight = torch.matmul(torch.stack(A_list, dim=0).permute(1,2,0), weight.squeeze().unsqueeze(1)).squeeze()
                P_weight = torch.matmul(torch.stack(P_list, dim=0).permute(1,2,3,0), weight.squeeze().unsqueeze(1)).squeeze()
                
                


                K[s:f] = K_weight[s:f]
                A[s:f] = A_weight[s:f]
                P[s:f] = P_weight[s:f]

            setattr(self,f'e_k_{l}',torch.nn.Parameter(K.data, requires_grad=True))
            setattr(self,f'e_a_{l}',torch.nn.Parameter(A.data, requires_grad=True))
            setattr(self,f'e_p_{l}',torch.nn.Parameter(P.data, requires_grad=True))


    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        #print(getattr(self,f'e_k_{0}'))
        if self.task_id > 0:
            self.weight_previous_prompt()
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
                    if self.max_classes > f:
                        K = torch.cat((K[:s].detach().clone(),K[s:f],K[f:self.max_classes].detach().clone()), dim=0)
                        A = torch.cat((A[:s].detach().clone(),A[s:f],A[f:self.max_classes].detach().clone()), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f],p[f:self.max_classes].detach().clone()), dim=0)
                    else:
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
                    if self.max_classes > f:
                        K = torch.cat((K[s:f],K[f:self.max_classes].detach().clone()), dim=0)
                        A = torch.cat((A[s:f],A[f:self.max_classes].detach().clone()), dim=0)
                        p = torch.cat((p[s:f],p[f:self.max_classes].detach().clone()), dim=0)
                    else:
                        K = K[s:f]
                        A = A[s:f]
                        p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
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
                    loss = self.ortho_penalty(K)
                elif self.ortho_mu == 2:
                    loss = self.ortho_penalty(A)
                elif self.ortho_mu == 3:
                    loss = self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 4:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                elif self.ortho_mu == 5:
                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 6:
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
                elif self.ortho_mu == 7:
                    # print("using all ortho penalty")

                    loss = self.ortho_penalty(K)
                    loss += self.ortho_penalty(A)
                    loss += self.ortho_penalty(p.flatten(start_dim=1,end_dim=2))
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
            #print(p_return)
        else:
            p_return = None
            P_ = None

        # return
        if train:
            return p_return, loss, x_block, P_
        else:
            return p_return, 0, x_block

    def ortho_penalty(self, t):
        return ((t @t.T - torch.eye(t.shape[0]).cuda(self.device))**2).mean() * 1e-6

def tensor_prompt(a, b, c=None, d=None, ortho=False, same=False):
    if same == False:
        if c is None and d is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
        elif d is None:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a,b,c,d), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.uniform_(p)
    else:
        if c is None and d is None:
            p = torch.nn.Parameter(torch.FloatTensor(1,b).uniform_(0, 1).repeat(a, 1), requires_grad=True)
        elif d is None:
            p = torch.nn.Parameter(torch.FloatTensor(1,b,c).uniform_(0, 1).repeat(a, 1, 1), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(1,b,c,d).uniform_(0, 1).repeat(a, 1, 1, 1), requires_grad=True)
    return p        

class ResNetZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, mode=1, prompt_flag=False, prompt_param=None, task_size=10, device='cuda:0', local_clients=10, num_clients=10):
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
                                           drop_path_rate=0, device=device
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

        elif self.prompt_flag == 'codap' or self.prompt_flag == 'cprompt':
            self.prompt = CodaPrompt(768, task_size, prompt_param, device=device)
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(768, task_size, prompt_param)
        elif self.prompt_flag == 'codap_weight' or self.prompt_flag == 'cprompt_weight':
            self.prompt = CodaPrompt_weight(768, task_size, prompt_param, device=device, clients_local=local_clients, num_clients=num_clients)
        elif self.prompt_flag == 'l2p_weight':
            pass
        elif self.prompt_flag == 'dual_weight':
            pass
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
        self.task_id = int(numclasses / self.task_size) - 1
        self.prompt.task_id = self.task_id
        self.prompt.process_frequency()

        

    def forward(self, x, pen=False, train=False, aq_k=None):
        if self.prompt is not None:
            with torch.no_grad():
                q, _, _ = self.feat(x)
                q = q[:,0,:] 
            out, prompt_loss, prompt_client = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, aq_k=aq_k)
            out = out[:,0,:]
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
        
    def feature_extractor(self, inputs):
        feature, _, _ = self.feat(inputs)
        return feature[:,0,:]

def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, task_size=10, device='cuda:0', local_clients = 10, num_clients=10):
    return ResNetZoo(num_classes=out_dim, pt=True, mode=0, prompt_flag=prompt_flag, prompt_param=prompt_param, task_size=task_size, device=device, local_clients=local_clients, num_clients=num_clients)