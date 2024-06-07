from GLFC_hard import GLFC_model_hard
from ResNet import resnet18_cbam
import torch
import copy
import random
import sys
import os.path as osp
import os
from myNetwork_hard import network_hard, LeNet_hard
from Fed_utils import * 
from ProxyServer import * 
from mini_imagenet import *
from tiny_imagenet import *
from iMNIST import *
from iDOMAINNET import *
from option import args_parser
from models_Cprompt.vit_coda_p import vit_pt_imnet
from models_Cprompt.vit_coda_p_hard import vit_pt_imnet_hard
from models_FCeD.cfed_network import cfed_network
from models_FCeD.cfed_network_hard import cfed_network_hard
from models_fedspace.fedspace_network import fedspace_network
from models_fedspace.fedspace_network_hard import fedspace_network_hard
from prompt_hard import DualPrompt_hard
from models_DWL.vit_coda_p_dwl import vit_pt_imnet_dwl
from dwl import DWL
from models_Cprompt.vision_transformer import VisionTransformer
import dataloaders
from dataloaders.utils import *
from cfed_hard import CFeD_hard
from fedspace_hard import FedSpace_model_hard
from iCOMMNIST import *
import nni

class ProtoQueue:

    def __init__(self, n_classes, max_length):
        self.n_classes = n_classes
        self.queue = {i: [] for i in range(n_classes)}
        self.max_length = max_length
        self.global_proto = {i: 0 for i in range(n_classes)}

    def insert(self, local_proto, local_radius, num_samples):
        for class_id in local_proto.keys():
            self.queue[class_id].append((local_proto[class_id], local_radius, num_samples[class_id]))

            while len(self.queue[class_id]) > self.max_length:
                self.queue[class_id].pop(0)

    def get_num_samples(self, q_id,  class_id):
        return self.queue[q_id][2][class_id]

    def compute_mean(self):
        for class_id in range(self.n_classes):
            if len(self.queue[class_id]) > 1:
                sum = 0
                ws = 0
                for item in self.queue[class_id]:
                    ws += item[2]
                    sum += item[0] * item[2]

                self.global_proto[class_id] = sum / ws

        return self.global_proto

params = {'task_index':30, 'topk_for_task':3, 'topk_for_task_selection':3,
          'class_index':30, 'topk_for_class':3,
          'learning_rate':0.001, 'epochs_local': 2, 'batch_size': 32, 'seed': 0}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

args = args_parser()
setup_seed(args.seed)
if len(args.device) == 1:
    args.device = args.device[0]
else:
    torch.cuda.set_device(args.device[0])

class_distribution_client_di = None

if "extencl" in args.method and args.dataset == 'ImageNet_R':
    high_frequency_class = list(random.sample(list(range(200)), 20))
    low_frequency_class = list(set(list(range(200))) - set(high_frequency_class))
elif "extencl" in args.method and args.dataset == 'DomainNet':
    high_frequency_class = list(random.sample(list(range(345)), 35))
    low_frequency_class = list(set(list(range(345))) - set(high_frequency_class))

if "centralized" in args.method or "one" in args.method or args.num_clients == 1:
    class_list = list(range(args.numclass))
    class_list_real = random.sample(list(range(args.numclass)), args.numclass)
    class_distribution_client = {0:[]}
    class_distribution_client_real = {0:[]}
    class_distribution_client_proportion = {0:[]}
    class_per_task = int(args.numclass / args.epochs_global * args.tasks_global)
    for i in range(int(args.epochs_global / args.tasks_global)):
        class_distribution_client[0].append(class_list[i*class_per_task:(i+1)*class_per_task])
        class_distribution_client_real[0].append(class_list_real[i*class_per_task:(i+1)*class_per_task])
        class_distribution_client_proportion[0].append([0,1])
elif "dil" in args.method:
    class_list = random.sample(list(range(200)), 200)[0:20]
    class_distribution_client = {}
    class_distribution_client_real = {}
    class_distribution_client_proportion = {}
    class_distribution_client_di = {}
    label_distribution = np.random.dirichlet([0.1]*25, 20)
    for i in range(args.num_clients):
        task_list = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 20, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 20)))
        class_distribution_client[i] = task_list
    for i in range(args.num_clients):
        task_list_real = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            task_list_real.append(class_list)
        class_distribution_client_real[i] = task_list_real
    for i in range(args.num_clients):
        task_list_pro = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            task_list_pro.append([0,0.2])
        class_distribution_client_proportion[i] = task_list_pro
    for i in range(args.num_clients):
        task_list_di = []
        for j in range(int(args.epochs_global / args.tasks_global)):
            class_list_di = []
            for k in range(20):
                class_list_di.append(label_distribution[k][i * int(args.epochs_global / args.tasks_global) + j])
            task_list_di.append(class_list_di)
        class_distribution_client_di[i] = task_list_di
else:
    if args.dataset == 'ImageNet_R':
        class_distribution_client = {}
        class_distribution_client_real = {}
        class_distribution_client_proportion = {}
        if "full" not in args.method:
            for i in range(args.num_clients):
                task_list = []
                for j in range(int(args.epochs_global / args.tasks_global)):
                    task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 20, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 20)))
                class_distribution_client[i] = task_list
        else:    
            task_list = []
            for i in range(20):
                task_list.append(list(range(i * 20, (i + 1) * 20)))
            class_distribution_client[0] = task_list
            for i in range(1,50):
                class_distribution_client[i] = [list(range(i * 20 + 400, (i + 1) * 20 + 400))]
                
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                if i * int(args.epochs_global / args.tasks_global) + j > 0:
                    similar_global_task_id = random.sample(list(range(i * int(args.epochs_global / args.tasks_global) + j)), 1)[0]
                    similar_level = random.sample(list(range(args.sim, 21)), 1)[0]
                    if similar_level != 0:
                        similar_client_id = int(similar_global_task_id // int(args.epochs_global / args.tasks_global))
                        similar_task_id = int(similar_global_task_id % int(args.epochs_global / args.tasks_global))
                        if similar_client_id == i:
                            task_list.append(list(random.sample(task_list[similar_task_id], similar_level)) + list(random.sample(list(set(list(range(200)))-set(task_list[similar_task_id])), 20-similar_level)))
                        else:
                            task_list.append(list(random.sample(class_distribution_client_real[similar_client_id][similar_task_id], similar_level)) + list(random.sample(list(set(list(range(200)))-set(class_distribution_client_real[similar_client_id][similar_task_id])), 20-similar_level)))                    
                    else:
                        if "extencl" in args.method:
                            high_frequency_part = random.sample(high_frequency_class, 10)
                            low_frequency_part = random.sample(low_frequency_class, 10)
                            total_part = high_frequency_part + low_frequency_part
                            task_list.append(total_part)
                        else:
                            task_list.append(list(random.sample(list(range(200)), 20))) 
                else:
                    if "extencl" in args.method:
                        high_frequency_part = random.sample(high_frequency_class, 10)
                        low_frequency_part = random.sample(low_frequency_class, 10)
                        total_part = high_frequency_part + low_frequency_part
                        task_list.append(total_part)
                    else:
                        task_list.append(list(random.sample(list(range(200)), 20))) 
            class_distribution_client_real[i] = task_list

        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                proportion_list = [0, 0.2, 0.4, 0.6, 0.8]
                start = proportion_list[int(i * int(args.epochs_global / args.tasks_global) + j) % 5]
                task_list.append([start, start + 0.2])
                
            class_distribution_client_proportion[i] = task_list
    elif args.dataset == 'DomainNet':
        class_distribution_client = {}
        class_distribution_client_real = {}
        class_distribution_client_proportion = {}
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                task_list.append(list(range((i * int(args.epochs_global / args.tasks_global) + j) * 35, (i * int(args.epochs_global / args.tasks_global) + j + 1) * 35)))
            class_distribution_client[i] = task_list
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                if i * int(args.epochs_global / args.tasks_global) + j > 0:
                    similar_global_task_id = random.sample(list(range(i * int(args.epochs_global / args.tasks_global) + j)), 1)[0]
                    similar_level = random.sample(list(range(args.sim, 36)), 1)[0]
                    if similar_level != 0:
                        similar_client_id = int(similar_global_task_id // int(args.epochs_global / args.tasks_global))
                        similar_task_id = int(similar_global_task_id % int(args.epochs_global / args.tasks_global))
                        if similar_client_id == i:
                            task_list.append(list(random.sample(task_list[similar_task_id], similar_level)) + list(random.sample(list(set(list(range(345)))-set(task_list[similar_task_id])), 35-similar_level)))
                        else:
                            task_list.append(list(random.sample(class_distribution_client_real[similar_client_id][similar_task_id], similar_level)) + list(random.sample(list(set(list(range(345)))-set(class_distribution_client_real[similar_client_id][similar_task_id])), 35-similar_level)))
                    else:
                        if "extencl" in args.method:
                            high_frequency_part = random.sample(high_frequency_class, 18)
                            low_frequency_part = random.sample(low_frequency_class, 17)
                            total_part = high_frequency_part + low_frequency_part
                            task_list.append(total_part)
                        else:
                            task_list.append(list(random.sample(list(range(345)), 35))) 
                else:
                    if "extencl" in args.method:
                        high_frequency_part = random.sample(high_frequency_class, 18)
                        low_frequency_part = random.sample(low_frequency_class, 17)
                        total_part = high_frequency_part + low_frequency_part
                        task_list.append(total_part)
                    else:
                        task_list.append(list(random.sample(list(range(345)), 35))) 
            class_distribution_client_real[i] = task_list
        for i in range(args.num_clients):
            task_list = []
            for j in range(int(args.epochs_global / args.tasks_global)):
                proportion_list = [i * 0.02 for i in range(50)]
                start = proportion_list[int(i * int(args.epochs_global / args.tasks_global) + j)]
                task_list.append([start, start + 0.02])
            class_distribution_client_proportion[i] = task_list


if "cprompt" not in args.method:
    pass

print("********* FULL TEST **********")
print(class_distribution_client)
print(class_distribution_client_real)
print(class_distribution_client_proportion)
print(class_distribution_client_di)

global_class_output = []
global_trained_task_id = []
global_trained_task_id_nosame = []
global_not_trained_task_id = []

if "fcil" in args.method:
    feature_extractor = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0, args=args
                                          )
    from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
    load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
    del load_dict['head.weight']; del load_dict['head.bias']
    feature_extractor.load_state_dict(load_dict)
    print(" freezing original model")
    for n,p  in feature_extractor.named_parameters():
        if not "prompt" in n:
            print(f"freezing {n}")
            p.requires_grad = False
elif "cprompt" in args.method:
    feature_extractor = None
elif "cfed" in args.method:
    feature_extractor = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0, args=args
                                          )
    from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
    load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
    del load_dict['head.weight']; del load_dict['head.bias']
    feature_extractor.load_state_dict(load_dict)
    print(" freezing original model")
    for n,p  in feature_extractor.named_parameters():
        if not "prompt" in n:
            print(f"freezing {n}")
            p.requires_grad = False
elif "fedspace" in args.method:
    feature_extractor = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0, args=args
                                          )
    from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
    load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
    del load_dict['head.weight']; del load_dict['head.bias']
    feature_extractor.load_state_dict(load_dict)
    print(" freezing original model")
    for n,p  in feature_extractor.named_parameters():
        if not "prompt" in n:
            print(f"freezing {n}")
            p.requires_grad = False
elif "dwl" in args.method:
    feature_extractor = None
else:
    feature_extractor = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                           num_heads=12, use_grad_checkpointing=False, ckpt_layer=0,
                                           drop_path_rate=0, args=args
                                          )
    from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
    load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()
    del load_dict['head.weight']; del load_dict['head.bias']
    feature_extractor.load_state_dict(load_dict)
    print(" freezing original model")
    for n,p  in feature_extractor.named_parameters():
        if not "prompt" in n:
            print(f"freezing {n}")
            p.requires_grad = False

num_clients = args.num_clients
old_client_0 = []
old_client_0_review = []
old_client_1 = [i for i in range(args.num_clients)]
new_client = []
models = []
pre_model_trainer = None
if "promptchain" in args.method:
    pass
elif "fcil" in args.method:
    model_g = network_hard(args.numclass, feature_extractor, class_distribution_client, class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, args=args)
    model_g = model_to_device(model_g, False, args.device)
    model_old = None
elif "cprompt" in args.method:
    model_g = vit_pt_imnet_hard(out_dim=args.numclass, prompt_flag=args.prompt_flag, prompt_param=args.prompt_param, task_size=args.task_size, device=args.device, local_clients=args.local_clients, num_clients=args.num_clients, class_distribution=class_distribution_client, tasks_global=args.tasks_global, class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, class_distribution_client_di=class_distribution_client_di, params=params, args=args)
    model_g = model_to_device(model_g, False, args.device)
    model_old = None
elif "cfed" in args.method:
    model_g = cfed_network_hard(args.numclass, feature_extractor, class_distribution_client, class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, args=args)
    model_g = model_to_device(model_g, False, args.device)
    model_old = None
elif "fedspace" in args.method:
    model_g = fedspace_network_hard(args.numclass, feature_extractor, class_distribution_client, class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, args=args)
    model_g = model_to_device(model_g, False, args.device)
    model_old = None
else: 
    model_g = network_hard(args.numclass, feature_extractor, class_distribution_client, class_distribution_real=class_distribution_client_real, class_distribution_proportion=class_distribution_client_proportion, args=args)
    model_g = model_to_device(model_g, False, args.device)
    model_old = None
train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=True, resize_imnet=True)
test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=True, resize_imnet=True)

if "codap_2d_v2" in args.prompt_flag:
    model_g.prompt.topk_com = params['topk_for_task_selection']

if args.dataset == 'CIFAR100':
    train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
    test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

elif args.dataset == 'TINY_IMAGENET':
    train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset
elif args.dataset == 'MNIST':
    train_dataset = iMNIST('dataset', transform=train_transform, download=True)
    test_dataset = iMNIST('dataset', test_transform=test_transform, train_1=False, train_2='train', download=True)   
elif args.dataset == 'COM_MNIST':
    train_dataset = iCOMMNIST('dataset', transform=train_transform, download=True)
    test_dataset = iCOMMNIST('dataset', test_transform=test_transform, train_1=False, train_2='test', download=True)
elif args.dataset == 'DomainNet':
    train_dataset = iDOMAIN_NET(args.dataroot, train=True, transform=train_transform, download_flag=True, seed=args.seed, validation=args.validation, domain=args.easy)
    test_dataset = iDOMAIN_NET(args.dataroot, train=False, transform=test_transform, download_flag=False, seed=args.seed, validation=args.validation)
elif args.dataset == 'ImageNet_R':
    train_dataset = iIMAGENET_R(args.dataroot, train=True, transform=train_transform, download_flag=True, seed=args.seed, validation=args.validation)
    test_dataset = iIMAGENET_R(args.dataroot, train=False, transform=test_transform, download_flag=False, seed=args.seed, validation=args.validation)
else:
    train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
    train_dataset.get_data()
    test_dataset = train_dataset
if "fcil" in args.method:
    encode_model = LeNet_hard(num_classes=100)
    encode_model.apply(weights_init)
elif "cprompt" in args.method:
    encode_model = None
elif "dwl" in args.method:
    encode_model = None
elif "cfed" in args.method:
    encode_model = None
elif "fedspace" in args.method:
    encode_model = None
else:
    encode_model = LeNet_hard(num_classes=100)
    encode_model.apply(weights_init)

if "fcil" in args.method:
    for i in range(args.num_clients):
        model_temp = GLFC_model_hard(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                    args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model, args.dataset)
        models.append(model_temp)
elif "cprompt" in args.method:
    for i in range(args.num_clients):
        model_temp = DualPrompt_hard(args.numclass, args.prompt_flag, args.prompt_param, args.task_size, 
                                args.batch_size, args.device, args.epochs_local, args.learning_rate, train_dataset, model_g, args.imbalance)
        models.append(model_temp)
elif "dwl" in args.method:
    for i in range(args.num_clients):
        model_temp = DWL(args.numclass, args.prompt_flag, args.prompt_param, args.task_size, 
                                args.batch_size, args.device, args.epochs_local, args.learning_rate, train_dataset, model_g)
        models.append(model_temp)
elif "cfed" in args.method:
    for i in range(args.num_clients):
        model_temp = CFeD_hard(args.batch_size, args.task_size, args.epochs_local, args.learning_rate, train_dataset, args.device, args.numclass, feature_extractor, args.dataset)
        models.append(model_temp)
elif "fedspace" in args.method:
    for i in range(args.num_clients):
        model_temp = FedSpace_model_hard(args.numclass, feature_extractor, args.batch_size, args.task_size,
                    args.epochs_local, args.learning_rate, train_dataset, args.device, args.optimizer, args.centralized_pretrain, args.centralized_fractal_pretrain_steps, args.temp, args.repr_loss_temp, args.lambda_proto_aug, args.lambda_repr_loss, args.dataset)
        models.append(model_temp)
else:
    for i in range(args.num_clients):
        model_temp = GLFC_model_hard(args.numclass, feature_extractor, args.batch_size, args.task_size, args.memory_size,
                    args.epochs_local, args.learning_rate, train_dataset, args.device, encode_model, args.dataset)
        models.append(model_temp)

## the proxy server
if "fcil" in args.method:
    proxy_server = proxyServer(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform)
elif "cprompt" in args.method:
    proxy_server = None
elif "dwl" in args.method:
    proxy_server = None
elif "cfed" in args.method:
    proxy_server = None
elif "fedspace" in args.method:
    proxy_server = None
else:
    proxy_server = proxyServer(args.device, args.learning_rate, args.numclass, feature_extractor, encode_model, train_transform)

## training log
output_dir = osp.join('./training_log', args.method, 'seed' + str(args.seed))
if not osp.exists(output_dir):
    os.system('mkdir -p ' + output_dir)
if not osp.exists(output_dir):
    os.mkdir(output_dir)

out_file = open(osp.join(output_dir, 'log_tar_' + str(args.learning_rate) + '.txt'), 'w')
log_str = 'method_{}, task_size_{}, learning_rate_{}'.format(args.method, args.task_size, args.learning_rate)
out_file.write(log_str + '\n')
out_file.flush()

old_task_id = -1
if "fedspace" in args.method:
    proto_global = None
    radius_global = None

choosing = {}
finished_task = {}
finished_task_forchoosing = {}
choosing_class = {}
finished_class = {}

client_finish_task_num = {}
for c in range(args.num_clients):
    client_finish_task_num[c] = 0

unpush_correlation = {}
for c in range(args.num_clients):
    unpush_correlation[c] = 0

unpull_correlation = {}
for c in range(args.num_clients):
    unpull_correlation[c] = 0

global_task_id_real = {}
for i in range(800):
    global_task_id_real[i] = i
if "cprompt" in args.prompt_flag:
    model_g.prompt.global_task_id_real = global_task_id_real

class_real = {}
for i in range(args.numclass):
    class_real[i] = i

if "classincremental" in args.method or "extension" in args.method or "extencl" in args.method:
    class_real = {}
    for c in class_distribution_client.keys():
        client_existing_class = []
        client_existing_class_real = []
        for i in range(len(class_distribution_client[c])):
            for j in range(len(class_distribution_client[c][i])):
                if class_distribution_client_real[c][i][j] not in client_existing_class_real:
                    client_existing_class.append(class_distribution_client[c][i][j])
                    client_existing_class_real.append(class_distribution_client_real[c][i][j])
                    class_real[class_distribution_client[c][i][j]] = class_distribution_client[c][i][j]
                else:
                    class_real[class_distribution_client[c][i][j]] = client_existing_class[client_existing_class_real.index(class_distribution_client_real[c][i][j])]

clients_index_pull = list(range(num_clients))
clients_index_push = list(range(num_clients))
acc_global_list = []
old_client_1_temp = []
for ep_g in range(args.epochs_global):
    
    pool_grad = []
    num_samples_list = []

    if "fcil" in args.method:
        model_old = proxy_server.model_back()
    elif "cprompt" in args.method:
        pass
    elif "dwl" in args.method:
        pass
    elif "cfed" in args.method:
        pass
    elif "fedspace" in args.method:
        model_old = None
    else:
        model_old = proxy_server.model_back()
    
    task_id = ep_g // args.tasks_global
    new_task = (task_id != old_task_id)
    print(new_task)
    
    if task_id != old_task_id and old_task_id != -1 and "extension" not in args.method and "extencl" not in args.method:
        overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
        new_client = []
        if "full" in args.method:
            old_client_1 = [0]
        elif "centralized" in args.method or "one" in args.method or args.num_clients == 1:
            old_client_1 = [0]
        else:
            old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.5))
        old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
        num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
        print(old_client_0)
        for c in old_client_1:
            client_finish_task_num[c] = client_finish_task_num[c] + 1
    elif task_id != old_task_id and old_task_id != -1 and ("extension" in args.method or "extencl" in args.method):
        overall_client = overall_client_temp
        new_client = new_client_temp
        old_client_1 = old_client_1_temp
        old_client_0 = old_client_0_temp
        num_clients = num_clients_temp
        print(old_client_0)
        for c in old_client_1:
            client_finish_task_num[c] = client_finish_task_num[c] + 1

    if "cfed_" in args.method and task_id > 0:
        old_client_0_review = random.sample(old_client_1, int(len(old_client_1) * 0.5))
        print(old_client_0_review)
    elif "cfednotran" in args.method and task_id > 0:
        old_client_0_review = []
        print(old_client_0_review)

    if task_id != old_task_id:
        model_g.Incremental_learning(task_id)

    if task_id != old_task_id and old_task_id != -1:   
        model_g = model_to_device(model_g, False, args.device)
        if "cprompt" in args.method:
            model_old = copy.deepcopy(model_g)
        if "dwl" in args.method:
            model_old = copy.deepcopy(model_g)
        elif "cfed" in args.method:
            model_old = copy.deepcopy(model_g)
        elif "fedspace" in args.method:
            if "weit" not in args.method:
                pass
            else:
                model_old = copy.deepcopy(model_g)
        else:
            pass
    
    if ep_g > 0 and "fedspace" in args.method and "weit" not in args.method:
        model_old = 1
    
    print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

    w_local = []
    m_local = []
    taskid_local = []
    clients_learned_task_id = []
    clients_learned_class = []

    idxs = None
    clients_index = random.sample(range(num_clients), args.local_clients)
    print('select part of clients to conduct local training')
    print(clients_index)

    if "fedspace" in args.method:
        #proto_queue = ProtoQueue(n_classes=args.total_classes, max_length=args.proto_queue_length)
        proto_locals = dict()
        radius_locals = dict()
        client_losses = []
    
    local_client_index = 0
    if "fedspace" in args.method:
        pass
    elif "cfed" in args.method:
        pass
    elif "fcil" in args.method:
        pass
    elif "codap_2d_v2" in args.prompt_flag:
        choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real = model_g.updateweight_with_promptchoosing(clients_index, clients_index_push, old_client_0, train_dataset, new_task, task_id, models, global_trained_task_id_nosame, choosing, choosing_class, finished_task, finished_task_forchoosing, finished_class, global_task_id_real, class_real, args, ep_g)
    if "cprompt" in args.method or "sharedcodap" in args.method:
        model_g.prompt.global_task_id_real = global_task_id_real
    else:
        pass

    w_g_last = copy.deepcopy(model_g.state_dict())

    for c in clients_index:
        if c in old_client_0:
            continue
        else:
            new_classes = []
            for i in class_distribution_client[c][task_id]:
                new_classes.append(class_real[i])
            global_class_output.extend(new_classes)
            if "full" not in args.method:
                global_trained_task_id.append(global_task_id_real[task_id * args.num_clients + c])
                global_trained_task_id_nosame.append(task_id * args.num_clients + c)
            else:
                if task_id == 0:
                    global_trained_task_id.append(global_task_id_real[c])
                    global_trained_task_id_nosame.append(c)
                else:
                    global_trained_task_id.append(global_task_id_real[49 + task_id])
                    global_trained_task_id_nosame.append(49 + task_id)
    if "weit" in args.method:
        global_class_output = list(set(global_class_output))
        global_trained_task_id = list(set(global_trained_task_id))
        global_trained_task_id_nosame = list(set(global_trained_task_id_nosame))
    else:
        global_class_output = sorted(list(set(global_class_output)))
        global_trained_task_id = sorted(list(set(global_trained_task_id)))
        global_trained_task_id_nosame = sorted(list(set(global_trained_task_id_nosame)))
    
    for t in range(len(global_trained_task_id)):
        global_trained_task_id[t] = global_task_id_real[global_trained_task_id[t]]
    global_trained_task_id = sorted(list(set(global_trained_task_id)))
    global_class_output_now = []
    for c in clients_index:
        if c in old_client_0:
            continue
        else:
            new_classes = []
            for i in class_distribution_client[c][task_id]:
                new_classes.append(class_real[i])
            global_class_output_now.extend(new_classes)
    global_class_output_now = sorted(list(set(global_class_output_now)))

    print(global_class_output)
    print(global_class_output_now)
    if "cprompt" in args.method:
        model_g.prompt.trained_task_id_forchoosing = global_trained_task_id
        model_g.prompt.trained_task_id = global_trained_task_id
    elif "sharedprompt" in args.method or "sharedcodap" in args.method:
        model_g.prompt.trained_task_id_forchoosing = global_trained_task_id
        model_g.prompt.trained_task_id = global_trained_task_id
        model_g.trained_task_id = global_trained_task_id
    else:
        model_g.trained_task_id = global_trained_task_id
    model_g.set_global_class_min_output(global_class_output, global_class_output_now)

    if (ep_g + 1) % args.tasks_global == 0 and ("extension" in args.method or "extencl" in args.method):
        overall_client_temp = len(old_client_0) + len(old_client_1) + len(new_client)
        new_client_temp = []
        if "full" in args.method:
            old_client_1_temp = [0]
        elif "fre" in args.method:
            new_task_list = [[1,2], [0,4], [1,3], [1,2]]
            old_client_1_temp = new_task_list[((ep_g + 1) // args.tasks_global) - 1]
        else:
            old_client_1_temp = random.sample([i for i in range(overall_client_temp)], int(overall_client_temp * 0.5))
        old_client_0_temp = [i for i in range(overall_client_temp) if i not in old_client_1_temp]
        num_clients_temp = len(new_client_temp) + len(old_client_1_temp) + len(old_client_0_temp)
        print(old_client_0_temp)
    
    w_g_not_trained = copy.deepcopy(model_g.state_dict())
    for c in clients_index:
        if "fcil" in args.method:
            local_model, proto_grad, num_samples = local_train_fcil(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, c, global_task_id_real=global_task_id_real, class_real=class_real)
            if "sharedcodap" in args.method:
                taskid_local.append(models[c].model.prompt.task_id)
        elif "cprompt" in args.method:
            if (ep_g + 1) % args.tasks_global == 0 and ("extension" in args.method or "extencl" in args.method) and c in old_client_1_temp and client_finish_task_num[c] >= 3 and args.prompt_flag == 'codap_2d_v2':
                local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_classes, idx, client_learned_task_id, global_task_id_real, global_trained_task_id = local_train_cprompt(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, global_task_id_real=global_task_id_real, class_real=class_real, consolidation=True)
            else:
                local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_classes, idx, client_learned_task_id, global_task_id_real, global_trained_task_id = local_train_cprompt(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, global_task_id_real=global_task_id_real, class_real=class_real, consolidation=False)
            taskid_local.append(models[c].model.prompt.task_id)
            clients_learned_task_id.append(client_learned_task_id)
            clients_learned_class.append(models[c].model.learned_classes)
        elif "dwl" in args.method:
            local_model, proto_grad, num_samples = local_train_dwl(models, c, model_g, task_id, model_old, ep_g, old_client_0)           
        elif "cfed" in args.method:
            local_model, proto_grad, average_loss, num_samples = local_train_cfed(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, old_client_0_review, global_task_id_real, class_real=class_real)       
            m_local.append(models[c].model)
            taskid_local.append(models[c].model.task_id)            
        elif "fedspace" in args.method:
            local_model, proto_grad, num_samples, num_sample_class, radius, prototype = local_train_fedspace(clients_index_push, models, c, model_g, task_id, model_old, old_client_0, proto_global, radius_global, global_task_id_real, ep_g, class_real=class_real)
            proto_locals[c] = {'sample_num': num_samples,
                                'prototype': prototype,
                                'num_samples_class': num_sample_class}
            radius_locals[c] = {'sample_num': num_samples,
                                      'radius': radius}
            taskid_local.append(models[c].model.task_id)
        else:
            local_model, proto_grad, num_samples = local_train_fcil(clients_index_push, models, c, model_g, task_id, model_old, ep_g, old_client_0, c)
        if ((ep_g + 1) % args.tasks_global == 0 and ("direct" in args.method or "FLorigin" in args.method or "notran" in args.method)):
            acc_global, accs_global = model_global_eval_hard(models[c].model, test_dataset, task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), models[c].current_class, models[c].current_class_real)
            log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                                        .format(c, models[c].model.task_id, ep_g, acc_global, accs_global)
            out_file.write(log_str + '\n')
            out_file.flush()
        
        w_local.append(local_model)
        if num_samples != None:
            num_samples_list.append(num_samples)
        local_client_index += 1

    if "extension" not in args.method:
        clients_index_pull = clients_index
        clients_index_push = clients_index
    else:
        if (ep_g + 1) % args.tasks_global == 0:
            clients_index_pull = set(list(random.sample(clients_index, int(len(clients_index) * 1))) + old_client_1_temp)
            clients_index_push = set(list(random.sample(clients_index, int(len(clients_index) * 1))) + old_client_1_temp)
        else:
            clients_index_pull = set(list(random.sample(clients_index, int(len(clients_index) * 1))))
            clients_index_push = set(list(random.sample(clients_index, int(len(clients_index) * 1))))
    print("clients_index_pull")
    print(clients_index_pull)
    print("clients_index_push")
    print(clients_index_push)
    print('every participant start updating their exemplar set and old model...')
    if "fcil" in args.method:
        participant_exemplar_storing_fcil(models, num_clients, model_g, old_client_0, task_id, clients_index)
    elif "cprompt" in args.method:
        participant_exemplar_storing_cprompt(models, num_clients, model_g, old_client_0, task_id, clients_index)
    elif "dwl" in args.method:
        participant_exemplar_storing_dwl(models, num_clients, model_g, old_client_0, task_id, clients_index)
    elif "cfed" in args.method:
        participant_exemplar_storing_cfed(models, num_clients, model_g, old_client_0, task_id, clients_index)
    elif "fedspace" in args.method:
        participant_exemplar_storing_fedspace(models, num_clients, model_g, old_client_0, task_id, clients_index, proto_global, radius_global) 
        proto_global = aggregate_proto_by_class(proto_locals, proto_global, model_g.fc.in_features, args.ema_global)
        radius_global = aggregate_radius(radius_locals)
        print(proto_global.keys())
    else:
        participant_exemplar_storing_fcil(models, num_clients, model_g, old_client_0, task_id, clients_index)
    
    print('updating finishes')
    print('federated aggregation...')
    
    if "fcil" in args.method:
        w_g_new = FedAvg_weit(w_local, w_g_last, num_samples_list, args.global_weight, clients_index, class_distribution_client, taskid_local, models, task_id, clients_index_pull, args.num_clients, global_task_id_real)
    elif "cprompt" in args.method:
        if 'codap_weight' == args.prompt_flag:
            w_g_new = FedAvg_withweights_codapweight(w_local, num_samples_list, clients_index, task_id, int(args.prompt_param[0]/int(args.epochs_local/args.tasks_global)))      
        elif 'codap_2d' == args.prompt_flag:
            w_g_new = FedAvg_withweights_and_taskfc_and_taskprompt_v2(w_local, num_samples_list, clients_index, class_distribution_client, task_id, taskid_local, int(args.prompt_param[0]/args.task_size), old_client_0)
        elif 'codap_2d_v2' == args.prompt_flag or 'codap' == args.prompt_flag:
            w_g_new = FedAvg_our_v1(w_local, num_samples_list, clients_index, class_distribution_client, task_id, taskid_local, old_client_0, args.num_clients, copy.deepcopy(model_g), args.global_update_lr, args.device, idxs, clients_learned_task_id, clients_learned_class, global_task_id_real, class_real, global_trained_task_id, global_class_output, models, clients_index_pull, w_g_last)
        else:
            w_g_new = FedAvg_our_v2(w_local, num_samples_list, clients_index, class_distribution_client, task_id, taskid_local, old_client_0, args.num_clients, copy.deepcopy(model_g), args.global_update_lr, args.device, idxs, clients_learned_task_id, clients_learned_class, clients_index_pull, w_g_last)
    elif "dwl" in args.method:
        w_g_new = FedAvg_with_dwl(w_local, num_samples_list)
    elif "cfed" in args.method:
        w_g_new = FedAvg_weit(w_local, w_g_last, num_samples_list, args.global_weight, clients_index, class_distribution_client, taskid_local, models, task_id, clients_index_pull, args.num_clients, global_task_id_real)
    elif "fedspace" in args.method:
        w_g_new = FedAvg_weit(w_local, w_g_last, num_samples_list, args.global_weight, clients_index, class_distribution_client, taskid_local, models, task_id, clients_index_pull, args.num_clients, global_task_id_real)
    else:
        w_g_new = FedAvg(w_local)

    
    model_g.load_state_dict(w_g_new)
    if "classincremental" in args.method and 'codap_2d_v2' == args.prompt_flag:
        for c in clients_index_pull:
            task_embedding_index = models[c].model.prompt.global_task_id_real[models[c].model.prompt.task_id * models[c].model.prompt.num_clients + models[c].model.prompt.client_index]
            model_g.prompt.task_embedding[task_embedding_index, :] = models[c].model.prompt.task_embedding[task_embedding_index, :]
    
    global_optimizer = 0
    global_lr_schedule = 0
    

    if "fcil" in args.method:
        proxy_server.model = copy.deepcopy(model_g)
        proxy_server.dataloader(pool_grad, new_task)
    elif "cprompt" in args.method:
        pass
    elif "dwl" in args.method:
        pass
    elif "cfed" in args.method:
        pass
    elif "fedspace" in args.method:
        pass
    else:
        proxy_server.model = copy.deepcopy(model_g)
        proxy_server.dataloader(pool_grad, new_task)
    
    if "cprompt" in args.method or "sharedcodap" in args.method:
        model_g.prompt.global_task_id_real = global_task_id_real
        for c in models:
            c.model.prompt.global_task_id_real = global_task_id_real
    else:
        pass

    if (ep_g + 1) % args.tasks_global == 0 and "direct" not in args.method and "notran" not in args.method:
        acc_global_list = []
        model_for_eval = None
        for i in global_trained_task_id_nosame:
            if "full" not in args.method:
                current_class = class_distribution_client[int(i % args.num_clients)][int(i // args.num_clients)]
            else:
                if i < 50:
                    current_class = class_distribution_client[i][0]
                else:
                    current_class = class_distribution_client[0][i-49]
            classes_list = []
            for j in current_class:
                classes_list.append(class_real[j])
            current_class = classes_list
            if "full" not in args.method:
                current_class_real = class_distribution_client_real[int(i % args.num_clients)][int(i // args.num_clients)]
                client_index = int(i % args.num_clients)
                model_task_id = int(i // args.num_clients)
            else:
                if i < 50:
                    client_index = i
                    model_task_id = 0
                else:
                    client_index = 0
                    model_task_id = i-49
                current_class_real = class_distribution_client_real[client_index][model_task_id]
            client_class_min_output = sorted(list(set(list(range(args.numclass)))-set(current_class)))
            client_class_max_output = current_class
            if "extension" in args.method or "extencl" in args.method:
                if client_index in clients_index_push:
                    model_for_eval = copy.deepcopy(models[client_index].model)
                else:
                    model_for_eval = copy.deepcopy(models[client_index].model)
            else:
                if client_index in clients_index_push:
                    model_for_eval = copy.deepcopy(model_g)
                else:
                    model_for_eval = copy.deepcopy(models[client_index].model)
            model_for_eval.client_index = client_index
            if "cprompt" in args.method or "sharedcodap" in args.method:
                model_for_eval.prompt.client_index = client_index
                model_for_eval.prompt.task_id = model_task_id
            if ("codap_2d_v2" in args.prompt_flag) and "cprompt" in args.method:
                model_for_eval.prompt.trained_task_id_forchoosing = finished_task_forchoosing[i]
                model_for_eval.prompt.trained_task_id = global_trained_task_id
            if "weit" in args.method:
                model_for_eval.trained_task_id = [ph for ph in global_trained_task_id if (ph // args.num_clients) <= (i // args.num_clients)]
                model_for_eval.set_global_class_min_output(global_class_output[0: len(model_for_eval.trained_task_id) * args.class_per_task], global_class_output[0: len(model_for_eval.trained_task_id) * args.class_per_task])
            model_for_eval.client_class_min_output = client_class_min_output
            model_for_eval.client_class_max_output = client_class_max_output
            if "cprompt" in args.method or "sharedcodap" in args.method:
                model_for_eval.prompt.client_learned_global_task_id = models[client_index].model.prompt.client_learned_global_task_id
            if "privacy" in args.method:
                acc_global, accs_global = model_global_eval_hard_privacy(model_for_eval, train_dataset, test_dataset, model_task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), class_distribution_client, class_distribution_client_real, class_distribution_client_proportion)
                acc_global_list.append(acc_global)
            elif "centralized" in args.method:
                model_for_eval.client_index = -1
                model_for_eval.prompt.client_index = -1 
                acc_global, accs_global = model_global_eval_hard(model_for_eval, test_dataset, model_task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), current_class, current_class_real)
                acc_global_list.append(acc_global)
                log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                                                    .format(client_index, model_task_id, ep_g, acc_global, accs_global)
                out_file.write(log_str + '\n')
                out_file.flush()
                print('powder Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'.format(client_index, model_task_id, ep_g, acc_global, accs_global))
            else:
                acc_global, accs_global = model_global_eval_hard(model_for_eval, test_dataset, model_task_id, args.task_size, args.device, args.method, int(args.epochs_global / args.tasks_global), current_class, current_class_real)
                acc_global_list.append(acc_global)
                log_str = 'Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'\
                                                    .format(client_index, model_task_id, ep_g, acc_global, accs_global)
                out_file.write(log_str + '\n')
                out_file.flush()
                print('powder Client: {}, Task: {}, Round: {} Accuracy = {:.2f}% = Accuracys = {}'.format(client_index, model_task_id, ep_g, acc_global, accs_global))
        del model_for_eval
        nni.report_intermediate_result({'default': sum(acc_global_list) / len(acc_global_list)})
    
    for c in clients_index_push:
        if "cprompt" in args.method or "sharedcodap" in args.method:
            client_learned_global_task_id_saved = models[c].model.prompt.client_learned_global_task_id
        models[c].model = copy.deepcopy(model_g)
        if "cprompt" in args.method or "sharedcodap" in args.method:
            models[c].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
    
    old_task_id = task_id
    if "notran" in args.method or "direct" in args.method:
        model_g.load_state_dict(w_g_not_trained)
        
print(sum(acc_global_list) / len(acc_global_list))
print(type(sum(acc_global_list) / len(acc_global_list)))
nni.report_final_result(float(sum(acc_global_list) / len(acc_global_list)))