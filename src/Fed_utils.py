import torch.nn as nn
import torch
import copy
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def model_to_device(model, parallel, device):
    
    if isinstance(device, int):
        card = torch.device("cuda:{}".format(device))
        model.to(card)
    else:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=device, output_device=device[0])
    return model



def participant_exemplar_storing_fcil(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, index)
            else:
                clients[index].beforeTrain(task_id, 1, index)
            clients[index].update_new_set(task_id, index)

def participant_exemplar_storing_cprompt(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, None, index)
            else:
                clients[index].beforeTrain(task_id, 1, None, index)

def participant_exemplar_storing_dwl(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0)
            else:
                clients[index].beforeTrain(task_id, 1)

def participant_exemplar_storing_cfed(clients, num, model_g, old_client, task_id, clients_index):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, index)
            else:
                clients[index].beforeTrain(task_id, 1, index)

def participant_exemplar_storing_fedspace(clients, num, model_g, old_client, task_id, clients_index, proto_global, radius_global):
    for index in range(num):
        #clients[index].model = copy.deepcopy(model_g)
        if index not in clients_index:
            if index in old_client:
                clients[index].beforeTrain(task_id, 0, proto_global, radius_global, index)
            else:
                clients[index].beforeTrain(task_id, 1, proto_global, radius_global, index)


        

def local_train_fcil(clients_index_push, clients, index, model_g, task_id, model_old, ep_g, old_client, client_index, global_task_id_real, class_real=None):
    if index in clients_index_push:
        if "sharedcodap" in model_g.args.method and clients[index].model is not None:
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
            clients[index].model = copy.deepcopy(model_g)
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
        else:
            clients[index].model = copy.deepcopy(model_g)
    else:
        if "sharedcodap" in model_g.args.method and clients[index].model is not None:
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
            temp_model = copy.deepcopy(model_g)
            temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
            temp_model.load_state_dict(temp_state_dict)
            clients[index].model = temp_model
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
        else:
            temp_model = copy.deepcopy(model_g)
            temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
            temp_model.load_state_dict(temp_state_dict)
            clients[index].model = temp_model
            model_old = [clients[index].old_model, clients[index].old_model]

    if index in old_client:
        clients[index].beforeTrain(task_id, 0, client_index, global_task_id_real, class_real)
    else:
        clients[index].beforeTrain(task_id, 1, client_index, global_task_id_real, class_real)

    clients[index].update_new_set(task_id, client_index)
    print(clients[index].signal)
    num_samples = clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = None
    #proto_grad = clients[index].proto_grad_sharing()
    #num_samples = None

    print('*' * 60)

    return local_model, proto_grad, num_samples

def local_train_cprompt(clients_index_push, clients, index, model_g, task_id, model_old, ep_g, old_client, classes=None, global_task_id_real=None, class_real=None, consolidation=False):
    if index in clients_index_push:
        if model_g.args.prompt_flag == 'codap_2d_v2':
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
        clients[index].model = copy.deepcopy(model_g)
        if model_g.args.prompt_flag == 'codap_2d_v2':
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved
    else:
        if model_g.args.prompt_flag == 'codap_2d_v2':
            client_learned_global_task_id_saved = clients[index].model.prompt.client_learned_global_task_id
        temp_model = copy.deepcopy(model_g)
        temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
        temp_model.load_state_dict(temp_state_dict)
        clients[index].model = temp_model
        if model_g.args.prompt_flag == 'codap_2d_v2':
            clients[index].model.prompt.client_learned_global_task_id = client_learned_global_task_id_saved

    if index in old_client:
        clients[index].beforeTrain(task_id, 0, classes, index, global_task_id_real, class_real)
    else:
        clients[index].beforeTrain(task_id, 1, classes, index, global_task_id_real, class_real)
    if "extension" in model_g.args.method or "extencl" in model_g.args.method:
        clients[index].update_new_set(task_id, index, ep_g)
    if ep_g < 0:   
        
        with torch.no_grad():
            clients[index].model.fc.fc.weight.data[clients[index].current_class] = 0 * clients[index].model.fc.fc.weight.data[clients[index].current_class]

    if consolidation:
        print("save consolidation")
        clients[index].model.prompt.save_extra_prompt()

    if index in old_client:
        num_samples, local_optimizer, local_lr_schedule, current_class = clients[index].train(ep_g, model_old, model_g, 0)
    else:
        num_samples, local_optimizer, local_lr_schedule, current_class = clients[index].train(ep_g, model_old, model_g)
    #prompt_importance = clients[index].compute_prompt_importance()
    #clients[index].model.eval()
    #idx = clients[index].reorder_prompt(prompt_importance)
    if consolidation:
        print("begin consolidation")
        target_task_id_data_list, origin_task_id, target_class_min_output_list, target_class_max_output_list = clients[index].generate_consolidation_dataset(class_real)
        clients[index].model.prompt.load_extra_prompt()
        clients[index].model.prompt.delete_extra_prompt()
        for t in target_task_id_data_list:
            global_task_id_real[t] = origin_task_id
        for t in range(len(clients[index].model.prompt.client_learned_global_task_id)):
            clients[index].model.prompt.client_learned_global_task_id[t] = global_task_id_real[clients[index].model.prompt.client_learned_global_task_id[t]]
        clients[index].model.prompt.client_learned_global_task_id = sorted(list(set(clients[index].model.prompt.client_learned_global_task_id)))
        clients[index].client_learned_global_task_id = clients[index].model.prompt.client_learned_global_task_id
        for t in range(len(clients[index].model.prompt.trained_task_id)):
            clients[index].model.prompt.trained_task_id[t] = global_task_id_real[clients[index].model.prompt.trained_task_id[t]]   
        clients[index].model.prompt.trained_task_id = sorted(list(set(clients[index].model.prompt.trained_task_id)))
        clients[index].consolidation_train(target_class_min_output_list, target_class_max_output_list)
    local_model = clients[index].model.state_dict()
    
    proto_grad = None

    print('*' * 60)
    return local_model, proto_grad, num_samples, local_optimizer, local_lr_schedule, current_class, None, clients[index].client_learned_task_id, global_task_id_real, clients[index].model.prompt.trained_task_id

def local_train_dwl(clients, index, model_g, task_id, model_old, ep_g, old_client):
    clients[index].model = copy.deepcopy(model_g)

    if index in old_client:
        clients[index].beforeTrain(task_id, 0)
    else:
        clients[index].beforeTrain(task_id, 1)
    
    num_samples = clients[index].train(ep_g, model_old, model_g)
    local_model = clients[index].model.state_dict()
    proto_grad = None

    print('*' * 60)
    return local_model, proto_grad, num_samples

def local_train_cfed(clients_index_push, clients, index, model_g, task_id, model_old, ep_g, old_client, old_client_0_review, global_task_id_real, class_real=None):
    if index in clients_index_push:
        clients[index].model = copy.deepcopy(model_g)
    else:
        temp_model = copy.deepcopy(model_g)
        temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
        temp_model.load_state_dict(temp_state_dict)
        clients[index].model = temp_model
        model_old = clients[index].old_model

    if index in old_client:
        clients[index].beforeTrain(task_id, 0, index, global_task_id_real, class_real)
    elif index in old_client_0_review:
        clients[index].beforeTrain(task_id, 2, index, global_task_id_real, class_real)
    else:
        clients[index].beforeTrain(task_id, 1, index, global_task_id_real, class_real)

    average_loss, num_samples = clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = None

    print('*' * 60)
    return local_model, proto_grad, average_loss, num_samples
#, clients[index].model

def centralized_fractual_pretraining(pre_model_trainer):
    pre_model_trainer.fractal_pretrain()


def federated_fractual_pretraining():
    return


def local_train_fedspace(clients_index_push, clients, index, model_g, task_id, model_old, old_client, proto_global, radius_global, global_task_id_real, ep_g, class_real=None):
    if index in clients_index_push:
        clients[index].model = copy.deepcopy(model_g)
    else:
        temp_model = copy.deepcopy(model_g)
        temp_state_dict = copy.deepcopy(clients[index].model.state_dict())
        temp_model.load_state_dict(temp_state_dict)
        clients[index].model = temp_model
        model_old = clients[index].old_model

    if index in old_client:
        clients[index].beforeTrain(task_id, 0, proto_global, radius_global, index, global_task_id_real, class_real)
    else:
        clients[index].beforeTrain(task_id, 1, proto_global, radius_global, index, global_task_id_real, class_real)


    loss_term, num_sample_class, num_samples = clients[index].train(ep_g, model_old)
    local_model = clients[index].model.state_dict()
    proto_grad = None
    radius, prototype, class_label = clients[index].proto_save()

    print('*' * 60)
    return local_model, proto_grad, num_samples, num_sample_class, radius, prototype


    

def FedAvg(models):
    w_avg = copy.deepcopy(models[0])
    print(w_avg.keys())
    for k in w_avg.keys():
        for i in range(1, len(models)):
            w_avg[k] += models[i][k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg

def FedAvg_withweights(models, num_samples_list):
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            weight = num_samples_list[i] / sum(num_samples_list)
            if weighted_sum is None:
                weighted_sum = weight * models[i][key]
            else:
                weighted_sum += weight * models[i][key]
        w_avg[key] = weighted_sum
    checkpoint = {
            'net': w_avg,
            }
    #torch.save(checkpoint, './checkpoint/test1.pth')
    return w_avg

def FedAvg_withweights_and_taskfc(models, num_samples_list, client_index, class_distribution_client, taskid_local):
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            weight = num_samples_list[i] / sum(num_samples_list)
            if key == 'fc.weight' or key == "fc.bias":
                if weighted_sum is None:
                    weighted_sum = models[i][key]
                else:
                    weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
            else:
                if weighted_sum is None:
                    weighted_sum = weight * models[i][key]
                else:
                    weighted_sum += weight * models[i][key]
            
        w_avg[key] = weighted_sum
    
    return w_avg

def FedAvg_withweights_and_taskfc_v1(models, num_samples_list, client_index, class_distribution_client, taskid_local):
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            if f'client{client_index[i]}' in key:
                w_avg[key] = models[i][key]
            elif 'client' not in key:
                weight = num_samples_list[i] / sum(num_samples_list)
                if key == 'fc.weight' or key == "fc.bias":
                    if weighted_sum is None:
                        weighted_sum = models[i][key]
                    else:
                        weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
            
        if weighted_sum is not None:
            w_avg[key] = weighted_sum
    
    return w_avg

def FedAvg_withweights_and_taskfc_and_taskprompt(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, pt):
    #print(client_index)
    #print(newest_task_id)
    #print(taskid_local)
    w_avg = copy.deepcopy(models[0])
    num_samples_newest_task = 0
    for i in range(len(num_samples_list)):
        if taskid_local[i] == newest_task_id:
            num_samples_newest_task = num_samples_newest_task + num_samples_list[i]
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        #task_samples_sum = {}
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            weight = num_samples_list[i] / sum(num_samples_list)
            weight_newest = num_samples_list[i] / num_samples_newest_task
            if key == 'fc.fc.weight' or key == "fc.fc.bias":
                if weighted_sum is None:
                    weighted_sum = models[i][key]
                else:
                    weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
            elif "prompt" in key:
                if taskid_local[i] == newest_task_id:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                        weighted_sum[taskid_local[i] * pt: (taskid_local[i] + 1) * pt] = weight_newest * models[i][key][taskid_local[i] * pt: (taskid_local[i] + 1) * pt]
                    else:
                        weighted_sum[0: taskid_local[i] * pt] += weight * models[i][key][0: taskid_local[i] * pt]
                        weighted_sum[taskid_local[i] * pt: (taskid_local[i] + 1) * pt] += weight_newest * models[i][key][taskid_local[i] * pt: (taskid_local[i] + 1) * pt]
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                        weighted_sum[newest_task_id * pt: (newest_task_id + 1) * pt] = 0 * models[i][key][newest_task_id * pt: (newest_task_id + 1) * pt]
                    else:
                        weighted_sum[0: newest_task_id * pt] += weight * models[i][key][0: newest_task_id * pt]
            else:
                if weighted_sum is None:
                    weighted_sum = weight * models[i][key]
                else:
                    weighted_sum += weight * models[i][key]
        
        w_avg[key] = weighted_sum
    return w_avg

def FedAvg_withweights_and_taskfc_and_taskprompt_v1(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, pt):
    print(client_index)
    print(newest_task_id)
    print(taskid_local)
    w_avg = copy.deepcopy(models[0])
    num_samples_newest_task = 0
    for i in range(len(num_samples_list)):
        if taskid_local[i] == newest_task_id:
            num_samples_newest_task = num_samples_newest_task + num_samples_list[i]
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        #task_samples_sum = {}
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            if f'client{client_index[i]}' in key:
                w_avg[key] = models[i][key]
            elif 'client' not in key:
                weight = num_samples_list[i] / sum(num_samples_list)
                weight_newest = num_samples_list[i] / num_samples_newest_task
                if key == 'fc.fc.weight' or key == "fc.fc.bias":
                    if weighted_sum is None:
                        weighted_sum = models[i][key]
                    else:
                        weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
                elif "prompt" in key:
                    if taskid_local[i] == newest_task_id:
                        if weighted_sum is None:
                            weighted_sum = weight * models[i][key]
                            weighted_sum[taskid_local[i] * pt: (taskid_local[i] + 1) * pt] = weight_newest * models[i][key][taskid_local[i] * pt: (taskid_local[i] + 1) * pt]
                        else:
                            weighted_sum[0: taskid_local[i] * pt] += weight * models[i][key][0: taskid_local[i] * pt]
                            weighted_sum[taskid_local[i] * pt: (taskid_local[i] + 1) * pt] += weight_newest * models[i][key][taskid_local[i] * pt: (taskid_local[i] + 1) * pt]
                    else:
                        if weighted_sum is None:
                            weighted_sum = weight * models[i][key]
                            weighted_sum[newest_task_id * pt: (newest_task_id + 1) * pt] = 0 * models[i][key][newest_task_id * pt: (newest_task_id + 1) * pt]
                        else:
                            weighted_sum[0: newest_task_id * pt] += weight * models[i][key][0: newest_task_id * pt]
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
        if weighted_sum is not None:
            w_avg[key] = weighted_sum
    return w_avg

def FedAvg_withweights_and_taskfc_and_taskprompt_v2(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, pt, old_client_0):
    print(client_index)
    print(newest_task_id)
    print(taskid_local)
    print(old_client_0)
    print(pt)
    w_avg = copy.deepcopy(models[0])
    num_samples_newest_task = 0
    for i in range(len(num_samples_list)):
        if client_index[i] not in old_client_0:
            num_samples_newest_task = num_samples_newest_task + num_samples_list[i]

    print(sum(num_samples_list) / num_samples_newest_task)
    for i in range(len(num_samples_list)):
        if client_index[i] not in old_client_0:
            for key in w_avg.keys():
                if "prompt" in key:
                    models[i][key][newest_task_id * pt: (newest_task_id + 1) * pt] = (sum(num_samples_list) / num_samples_newest_task) * models[i][key][newest_task_id * pt: (newest_task_id + 1) * pt]
        else:
            for key in w_avg.keys():
                if "prompt" in key:
                    models[i][key][newest_task_id * pt: (newest_task_id + 1) * pt] = 0 * models[i][key][newest_task_id * pt: (newest_task_id + 1) * pt]
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        #task_samples_sum = {}
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            if f'client{client_index[i]}' in key:
                w_avg[key] = models[i][key]
            elif 'client' not in key:
                weight = num_samples_list[i] / sum(num_samples_list)
                #weight_newest = num_samples_list[i] / num_samples_newest_task
                if key == 'fc.fc.weight' or key == "fc.fc.bias":
                    
                    if weighted_sum is None:
                        weighted_sum = models[i][key]
                    else:
                        weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
        if weighted_sum is not None:
            w_avg[key] = weighted_sum
    return w_avg


def FedAvg_our_v3(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, old_client_0, num_clients, model_g, global_update_lr, device, idxs, clients_learned_task_id, clients_learned_class, global_task_id_real, class_real, global_trained_task_id, global_class_output, models_model):
    #print(taskid_local)
    task_frequency = {}
    class_frequency = {}
    for m in models_model:
        global_task_id = global_task_id_real[m.model.prompt.task_id * num_clients + m.model.prompt.client_index]
        if global_task_id not in task_frequency.keys():
            task_frequency[global_task_id] = 1
        else:
            task_frequency[global_task_id] = task_frequency[global_task_id] + 1
        current_classes = m.model.current_class
        for c in current_classes:
            if c not in class_frequency.keys():
                class_frequency[c] = 1
            else:
                class_frequency[c] = class_frequency[c] + 1

    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys():
        weighted_sum = None
        weighted_sum_1 = None
        for i in range(len(num_samples_list)):
            weight = num_samples_list[i] / sum(num_samples_list)
            if "prompt" in key:
                if client_index[i] != 20:
                    if weighted_sum is None:
                        weighted_sum = models[i][key]
                    else:
                        weighted_sum[taskid_local[i] * num_clients + client_index[i]] = models[i][key][taskid_local[i] * num_clients + client_index[i]]
                else:
                    pass
            #elif key == "fc.fc.weight" or key == "fc.fc.bias" or key == "fc.fc_ova.weight" or key == "fc.fc_ova.bias":
            elif key == "fc.fc.weight" or key == "fc.fc.bias":    
                #print(key)
                
                print(list(class_frequency.keys()))
                print(class_frequency)
                if weighted_sum is None:
                    weighted_sum = copy.deepcopy(models[i][key])
                    weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                    #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                    for c in models_model[client_index[i]].model.current_class:
                        weighted_sum[c] = models[i][key][c] / class_frequency[c]
                else:
                    #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                    for c in models_model[client_index[i]].model.current_class:
                        weighted_sum[c] += models[i][key][c] / class_frequency[c]
                if key == "fc.fc.weight":
                    print(weighted_sum[0][0:10])
                    print(weighted_sum[40][0:10])
                    print(weighted_sum[80][0:10])
                
                
            else:
                if weighted_sum is None:
                    weighted_sum = weight * models[i][key]
                else:
                    weighted_sum += weight * models[i][key]
            
            
            
        if weighted_sum is not None:
            w_avg[key] = weighted_sum 
    model_g_before = copy.deepcopy(model_g)   
    model_g.load_state_dict(w_avg)
    print("*************update global**************")
    #update_global(model_g_before, model_g, models, taskid_local, client_index, num_clients, global_update_lr, device, idxs)
    return model_g.state_dict()

def FedAvg_our_v1(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, old_client_0, num_clients, model_g, global_update_lr, device, idxs, clients_learned_task_id, clients_learned_class, global_task_id_real, class_real, global_trained_task_id, global_class_output, models_model, clients_index_pull, w_g_last):
    #print(taskid_local)
    task_frequency = {}
    class_frequency = {}
    for m in models_model:
        if m.model.prompt.client_index in clients_index_pull:
            if "full" not in m.model.prompt.args.method:
                global_task_id = global_task_id_real[m.model.prompt.task_id * num_clients + m.model.prompt.client_index]
            else:
                if m.model.prompt.task_id == 0:
                    global_task_id = m.model.prompt.client_index
                else:
                    global_task_id = m.model.prompt.task_id + 49
            if global_task_id not in task_frequency.keys():
                task_frequency[global_task_id] = 1
            else:
                task_frequency[global_task_id] = task_frequency[global_task_id] + 1
            current_classes = m.model.current_class
            print(current_classes)
            for c in current_classes:
                if c not in class_frequency.keys():
                    class_frequency[c] = 1
                else:
                    class_frequency[c] = class_frequency[c] + 1
    
    summation = sum([num_samples_list[i] for i in range(len(num_samples_list)) if client_index[i] in clients_index_pull])
    if "extension" not in models_model[0].model.prompt.args.method:
        w_avg = copy.deepcopy(models[0])
    else:
        w_avg = copy.deepcopy(w_g_last)
    for key in w_avg.keys():
        weighted_sum = None
        for i in range(len(num_samples_list)):
            weight = num_samples_list[i] / summation
            if "prompt" in key:
                if client_index[i] in clients_index_pull:
                    if client_index[i] != 10000:
                        if "full" not in models_model[0].model.prompt.args.method:
                            global_task_id = global_task_id_real[taskid_local[i] * num_clients + client_index[i]]
                        else:
                            if taskid_local[i] == 0:
                                global_task_id = client_index[i]
                            else:
                                global_task_id = taskid_local[i] + 49
                        if weighted_sum is None:
                            weighted_sum = copy.deepcopy(models[i][key])
                            weighted_sum[list(task_frequency.keys())] = 0 * models[i][key][list(task_frequency.keys())]
                            weighted_sum[global_task_id] = models[i][key][global_task_id] / task_frequency[global_task_id]
                        else:
                            weighted_sum[global_task_id] += models[i][key][global_task_id] / task_frequency[global_task_id]
                    else:
                        pass
            #elif key == "fc.fc.weight" or key == "fc.fc.bias" or key == "fc.fc_ova.weight" or key == "fc.fc_ova.bias":
            elif key == "fc.fc.weight" or key == "fc.fc.bias":    
                #print(key)
                #print(list(class_frequency.keys()))
                if client_index[i] in clients_index_pull:
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] = models[i][key][c] / class_frequency[c]
                    else:
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] += models[i][key][c] / class_frequency[c]
                        #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
            else:
                if client_index[i] in clients_index_pull:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
            
            
            
        if weighted_sum is not None:
            w_avg[key] = weighted_sum 
    model_g_before = copy.deepcopy(model_g)   
    model_g.load_state_dict(w_avg)
    print("*************update global**************")
    #update_global(model_g_before, model_g, models, taskid_local, client_index, num_clients, global_update_lr, device, idxs)
    return model_g.state_dict()

def FedAvg_our_v2(models, num_samples_list, client_index, class_distribution_client, newest_task_id, taskid_local, old_client_0, num_clients, model_g, global_update_lr, device, idxs, clients_learned_task_id, clients_learned_class, clients_index_pull, w_g_last):
    #print(taskid_local)
    summation = sum([num_samples_list[i] for i in range(len(num_samples_list)) if client_index[i] in clients_index_pull])
    #w_avg = copy.deepcopy(models[0])
    if "extension" not in model_g.prompt.args.method:
        w_avg = copy.deepcopy(models[0])
    else:
        w_avg = copy.deepcopy(w_g_last)
    for key in w_avg.keys():
        weighted_sum = None
        for i in range(len(num_samples_list)):
            if client_index[i] in clients_index_pull:
                weight = num_samples_list[i] / summation
                if key == "fc.fc.weight" or key == "fc.fc.bias":    
                    #print(key)
                    if client_index[i] != 20:
                        if weighted_sum is None:
                            weighted_sum = models[i][key]
                            #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                            weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
                        else:
                            #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                            weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]]
                            #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
                    else:
                        pass
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
            
            
        if weighted_sum is not None:
            w_avg[key] = weighted_sum 
    model_g_before = copy.deepcopy(model_g)   
    model_g.load_state_dict(w_avg)
    print("*************update global**************")
    #update_global(model_g_before, model_g, models, taskid_local, client_index, num_clients, global_update_lr, device, idxs)
    return model_g.state_dict()

def update_global(model_g_before, model_g, models, taskid_local, client_index, num_clients, global_update_lr, device, idxs):
    #diff_K = [, forclient]
    P = model_g_before.prompt.get_previous(taskid_local, client_index, idxs)
    P_now = model_g.prompt.get_previous(taskid_local, client_index, None, pre=model_g_before.prompt)
    P1 = []
    P2 = []
    P3 = []
    diff_P1 = []
    diff_P2 = []
    diff_P3 = []
    i = 0
    for j in range(len(models)):
        client = models[j]
        #local_task_id = taskid_local[j]
        #local_client_index = client_index[j]
        for key in client.keys():
            if "k_share" in key:
                #print(P[i][0][0])
                #print(client[key][local_task_id * num_clients + local_client_index][0][0])
                P1.append(P[i])
                diff_P1.append(P_now[i].clone().detach() - P[i].clone().detach())
                i = i + 1
        for key in client.keys():
            if "a_share" in key:
                P2.append(P[i])
                diff_P2.append(P_now[i].clone().detach() - P[i].clone().detach())
                i = i + 1
        for key in client.keys():
            if "p_share" in key:
                P3.append(P[i])
                diff_P3.append(P_now[i].clone().detach() - P[i].clone().detach())
                i = i + 1
    
    P1 = torch.cat([item.unsqueeze(0) for item in P1], dim=0)
    P2 = torch.cat([item.unsqueeze(0) for item in P2], dim=0)
    P3 = torch.cat([item.unsqueeze(0) for item in P3], dim=0)
    diff_P1 = torch.cat([item.unsqueeze(0) for item in diff_P1], dim=0)
    diff_P2 = torch.cat([item.unsqueeze(0) for item in diff_P2], dim=0)
    diff_P3 = torch.cat([item.unsqueeze(0) for item in diff_P3], dim=0)
    #P1 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in P1])).cuda(device)
    #P2 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in P2])).cuda(device)
    #P3 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in P3])).cuda(device)
    #diff_P1 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in diff_P1])).cuda(device)
    #diff_P2 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in diff_P2])).cuda(device)
    #diff_P3 = torch.FloatTensor(np.array([item.cpu().detach().numpy() for item in diff_P3])).cuda(device)
    print(model_g.prompt.weight.requires_grad)
    grads1 = torch.autograd.grad(
        outputs = P1,
        inputs=model_g_before.prompt.weight,
        grad_outputs=diff_P1,
        allow_unused=True,
        retain_graph=True
    )
    grads2 = torch.autograd.grad(
        outputs = P2,
        inputs=model_g_before.prompt.weight,
        grad_outputs=diff_P2,
        allow_unused=True,
        retain_graph=True
    )
    grads3 = torch.autograd.grad(
        outputs = P3,
        inputs=model_g_before.prompt.weight,
        grad_outputs=diff_P3,
        allow_unused=True
    )

    #print(grads1[0].size())
    #print(len(grads1))
    
    if (grads1 is not None) and (grads2 is not None) and (grads3 is not None):
        weight = nn.functional.normalize(copy.deepcopy(model_g.prompt.weight), dim=1)
        weight = torch.mm(weight, weight.T)
        print(weight)
        model_g.prompt.weight.data -= global_update_lr * (grads1[0] + grads2[0] + grads3[0])
        weight = nn.functional.normalize(copy.deepcopy(model_g.prompt.weight), dim=1)
        weight = torch.mm(weight, weight.T)
        print(weight)



def FedAvg_withweights_codapweight(models, num_samples_list, clients_index, task_id, pt):
    w_avg = copy.deepcopy(models[0])
    #print(w_avg.keys())
    for key in w_avg.keys(): # iterate over the keys of the model
        if 'prompt' in key:
            continue
        else:
            weighted_sum = None
            for i in range(len(num_samples_list)): # iterate over the cleint weights
                # print(all_client_weights)
                weight = num_samples_list[i] / sum(num_samples_list)
                if weighted_sum is None:
                    weighted_sum = weight * models[i][key]
                else:
                    weighted_sum += weight * models[i][key]
            w_avg[key] = weighted_sum
    
    
    for l in [0, 1, 2, 3, 4]:
        weighted_sum1 = None
        weighted_sum2 = None
        weighted_sum3 = None
        key1 = f'prompt.e_k_{l}'
        key2 = f'prompt.e_a_{l}'
        key3 = f'prompt.e_p_{l}'
        for i in range(len(num_samples_list)):
            w_avg[f'prompt.e_k_{l}_{i}'][task_id * pt: (task_id + 1) * pt] = models[i][f'prompt.e_k_{l}'][task_id * pt: (task_id + 1) * pt]
            w_avg[f'prompt.e_a_{l}_{i}'][task_id * pt: (task_id + 1) * pt] = models[i][f'prompt.e_a_{l}'][task_id * pt: (task_id + 1) * pt]
            w_avg[f'prompt.e_p_{l}_{i}'][task_id * pt: (task_id + 1) * pt] = models[i][f'prompt.e_p_{l}'][task_id * pt: (task_id + 1) * pt]

            
            weight = num_samples_list[i] / sum(num_samples_list)
            if weighted_sum1 is None:
                weighted_sum1 = weight * models[i][key1]
            else:
                weighted_sum1 += weight * models[i][key1]
            if weighted_sum2 is None:
                weighted_sum2 = weight * models[i][key2]
            else:
                weighted_sum2 += weight * models[i][key2]
            if weighted_sum3 is None:
                weighted_sum3 = weight * models[i][key3]
            else:
                weighted_sum3 += weight * models[i][key3]
        w_avg[key1] = weighted_sum1
        w_avg[key2] = weighted_sum2
        w_avg[key3] = weighted_sum3
    
    
    if task_id > 0:
        i = 0
        for index in clients_index:
            for t in range(task_id):
                w_avg[f'prompt.weight_{index}_{t}'] = models[i][f'prompt.weight_{index}_{t}']

            i = i + 1
    checkpoint = {
            'net': w_avg,
            }
    #torch.save(checkpoint, './checkpoint/test2.pth')
    return w_avg



def FedAvg_with_dwl(models, num_samples_list):
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            weight = num_samples_list[i] / sum(num_samples_list)
            if weighted_sum is None:
                weighted_sum = weight * models[i][key]
            else:
                weighted_sum += weight * models[i][key]
        w_avg[key] = weighted_sum
    return w_avg

def FedAvg_cfed(models, model_last_round, num_samples_list, global_weight, client_index, class_distribution_client, taskid_local, models_model, task_id):
    
    global_weight = 1
    
    class_frequency = {}
    for m in models_model:
        current_classes = m.model.current_class
        for c in current_classes:
            if c not in class_frequency.keys():
                class_frequency[c] = 1
            else:
                class_frequency[c] = class_frequency[c] + 1
    
    w_avg = copy.deepcopy(models[0])
    print(num_samples_list)
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            weight = num_samples_list[i] / sum(num_samples_list)
            if key == "fc.weight" or key == "fc.bias":    
                #print(key)
                if client_index[i] != 20:
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] = (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = global_weight * models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]] + model_last_round[key][class_distribution_client[client_index[i]][taskid_local[i]]] * (1 - global_weight)
                    else:
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] += (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = global_weight * models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]] + model_last_round[key][class_distribution_client[client_index[i]][taskid_local[i]]] * (1 - global_weight)
                        #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
                else:
                    pass
            else:
                if weighted_sum is None:
                    weighted_sum =  global_weight * weight * models[i][key] + (1 - global_weight) * model_last_round[key]
                else:
                    weighted_sum += global_weight * weight * models[i][key] + (1 - global_weight) * model_last_round[key]
        w_avg[key] = weighted_sum
        if weighted_sum is None:
            pass
        else:
            pass
    return w_avg

def FedAvg_fedspace(models, model_last_round, num_samples_list, global_weight, client_index, class_distribution_client, taskid_local, models_model, task_id):
    if task_id == 0:
        global_weight = 1
    
    class_frequency = {}
    for m in models_model:
        current_classes = m.model.current_class
        for c in current_classes:
            if c not in class_frequency.keys():
                class_frequency[c] = 1
            else:
                class_frequency[c] = class_frequency[c] + 1
    
    w_avg = copy.deepcopy(models[0])
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            weight = num_samples_list[i] / sum(num_samples_list)
            if key == "fc.weight" or key == "fc.bias":    
                #print(key)
                if client_index[i] != 20:
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] = (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = global_weight * models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]] + model_last_round[key][class_distribution_client[client_index[i]][taskid_local[i]]] * (1 - global_weight)
                    else:
                        for c in models_model[client_index[i]].model.current_class:
                            weighted_sum[c] += (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = global_weight * models[i][key][class_distribution_client[client_index[i]][taskid_local[i]]] + model_last_round[key][class_distribution_client[client_index[i]][taskid_local[i]]] * (1 - global_weight)
                        #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
                else:
                    pass
            else:
                if weighted_sum is None:
                    weighted_sum =  global_weight * weight * models[i][key] + (1 - global_weight) * model_last_round[key]
                else:
                    weighted_sum += global_weight * weight * models[i][key] + (1 - global_weight) * model_last_round[key]
        w_avg[key] = weighted_sum

    return w_avg

def FedAvg_weit(models, model_last_round, num_samples_list, global_weight, client_index, class_distribution_client, taskid_local, models_model, task_id, clients_index_pull, num_clients, global_task_id_real):
    if task_id == 0:
        global_weight = 1
    
    summation = sum([num_samples_list[i] for i in range(len(num_samples_list)) if client_index[i] in clients_index_pull])
    #w_avg = copy.deepcopy(models[0])
    if "extension" not in models_model[0].model.args.method:
        w_avg = copy.deepcopy(models[0])
    else:
        w_avg = copy.deepcopy(model_last_round)
    for key in w_avg.keys(): # iterate over the keys of the model
        weighted_sum = None
        for i in range(len(num_samples_list)): # iterate over the cleint weights
            # print(all_client_weights)
            if client_index[i] in clients_index_pull:
                weight = num_samples_list[i] / summation
                if key == "fc.weight" or key == "fc.bias":    
                    #print(key)
                    if client_index[i] != 20:
                        if weighted_sum is None:
                            weighted_sum = copy.deepcopy(models[i][key])
                            #weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                            #for c in models_model[client_index[i]].model.current_class:
                                #weighted_sum[c] = (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                            #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                            weighted_sum[models_model[client_index[i]].model.current_class] = models[i][key][models_model[client_index[i]].model.current_class]
                        else:
                            #for c in models_model[client_index[i]].model.current_class:
                                #weighted_sum[c] += (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                            #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                            weighted_sum[models_model[client_index[i]].model.current_class] = models[i][key][models_model[client_index[i]].model.current_class]
                            #weighted_sum[clients_learned_class[i]] = models[i][key][clients_learned_class[i]]
                    else:
                        pass
                elif key == "aggregate_weight" and "weit" in models_model[0].model.args.method:
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        #weighted_sum[list(class_frequency.keys())] = 0 * models[i][key][list(class_frequency.keys())]
                        #for c in models_model[client_index[i]].model.current_class:
                            #weighted_sum[c] = (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        weighted_sum[:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index] = models[i][key][:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index]
                    else:
                        #for c in models_model[client_index[i]].model.current_class:
                            #weighted_sum[c] += (global_weight * models[i][key][c] + model_last_round[key][c] * (1 - global_weight)) / class_frequency[c]
                        #weighted_sum[class_distribution_client[client_index[i]][taskid_local[i]]] = models[i][key][list(range(120, 140))]
                        weighted_sum[:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index] = models[i][key][:, models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index]
                elif "feature" in key and '{}'.format(models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index) in key and "weit" in models_model[0].model.args.method and "sharedencoder" in models_model[0].model.args.method:
                    weighted_sum = models[i][key]
                    break
                elif "prompt" in key and '{}'.format(models_model[client_index[i]].model.task_id * len(models) + models_model[client_index[i]].model.client_index) in key and "weit" in models_model[0].model.args.method and "sharedprompt" in models_model[0].model.args.method:
                    weighted_sum = models[i][key]
                    break
                elif "sharedcodap" in models_model[0].model.args.method and "prompt" in key and "global" not in key:
                    if "full" not in models_model[0].model.prompt.args.method:
                        #print(global_task_id_real)
                        #print(taskid_local[i] * num_clients + client_index[i])
                        global_task_id = global_task_id_real[taskid_local[i] * num_clients + client_index[i]]
                    else:
                        if taskid_local[i] == 0:
                            global_task_id = client_index[i]
                        else:
                            global_task_id = taskid_local[i] + 49
                    if weighted_sum is None:
                        weighted_sum = copy.deepcopy(models[i][key])
                        #weighted_sum[list(task_frequency.keys())] = 0 * models[i][key][list(task_frequency.keys())]
                        weighted_sum[global_task_id] = models[i][key][global_task_id]
                    else:
                        weighted_sum[global_task_id] = models[i][key][global_task_id]
                else:
                    if weighted_sum is None:
                        weighted_sum = weight * models[i][key]
                    else:
                        weighted_sum += weight * models[i][key]
        w_avg[key] = weighted_sum

    return w_avg

def aggregate_proto_by_class(proto_locals, proto_global_old, feature_size, ema_global):
    global_classes = set()

    for client in proto_locals.keys():
        global_classes = set.union(global_classes, set(proto_locals[client]["prototype"].keys()))
    global_classes = list(global_classes)
    proto_global = {k: np.zeros(feature_size) for k in global_classes}

    weights_sums = {k: 0 for k in global_classes}

    for client in proto_locals.keys():
        local_proto = proto_locals[client]['prototype']
        for j in global_classes:
            if j in local_proto.keys() and not np.all(local_proto[j] == 0):
                w = proto_locals[client]["num_samples_class"][j]
                proto_global[j] += local_proto[j] * w
                weights_sums[j] += w

    for j in global_classes:
        if 0 < weights_sums[j] < 1:
            proto_global[j] /= weights_sums[j]

    if proto_global_old is not None:
        for k in proto_global_old.keys():
            if k in proto_global.keys():
                proto_global[k] = proto_global[k] * ema_global + proto_global_old[k] * (
                        1 - ema_global)
            else:
                proto_global[k] = proto_global_old[k]

    return proto_global

def aggregate_radius(radius_locals):
    radius_global = 0
    training_num = 0
    for client in radius_locals.keys():
        training_num += radius_locals[client]['sample_num']

    for client in radius_locals.keys():
        local_sample_number = radius_locals[client]['sample_num']
        local_radius = radius_locals[client]['radius']
        w = local_sample_number / training_num
        radius_global += local_radius * w

    return radius_global

def model_global_eval_proto(model_g, proto_global, radius_global, task_id, task_size, device, method):
    model_to_device(model_g, False, device)
    model_g.eval()
    proto_aug = []
    proto_aug_label = []
    index = [k for k, v in proto_global.items()]
    if index:
        for i in range(128):
            np.random.shuffle(index)
            #temp = prototype[index[0]] + np.random.normal(0, 1, prototype[index[0]].shape[0]) * radius
            temp = proto_global[index[0]]
            #print(i)
            proto_aug.append(temp)
            proto_aug_label.append(index[0])
            #proto_aug_label.append(4 * index[0])
        proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(device)
        proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).long()
        soft_feat_aug = model_g.predict(proto_aug)
        predicts = torch.max(soft_feat_aug, dim=1)[1]
        correct = (predicts.cpu() == proto_aug_label).sum()
        #total += len(labels)
        accuracy = (100 * correct / 128)


    model_g.train()
    return accuracy
        



def model_global_eval(model_g, test_dataset, task_id, task_size, device, method, task_num):
    model_to_device(model_g, False, device)
    model_g.eval()
    test_dataset.getTestData([0, task_size * (task_id + 1)])
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=64, num_workers=2, pin_memory=True)
    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            if "fcil" in method:
                outputs = model_g(imgs)
            elif "cprompt" in method:
                outputs = model_g(imgs)
            elif "cfed" in method:
                outputs = model_g(imgs)
            elif "fedspace" in method:
                outputs = model_g(imgs)
                #outputs = (outputs[:, 0::4] + outputs[:, 1::4] + outputs[:, 2::4] + outputs[:, 3::4]) / 4
                #outputs = outputs[:, ::4]
            else:
                outputs = model_g(imgs)
                
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        if step == 1:
            predicts_1 = predicts.cpu()
            labels_1 = labels.cpu()
        total += len(labels)
    accuracy = 100 * correct / total

    accuracys = []
    for i in range(task_num):
        test_dataset.getTestData([task_size * i, task_size * (i + 1)])
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128, num_workers=2, pin_memory=True)
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            if isinstance(device, int):
                imgs, labels = imgs.cuda(device), labels.cuda(device)
            else:
                imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                if "fcil" in method:
                    outputs = model_g(imgs)
                elif "cprompt" in method:
                    outputs = model_g(imgs)
                elif "cfed" in method:
                    outputs = model_g(imgs)
                elif "fedspace" in method:
                    outputs = model_g(imgs)
                    #outputs = (outputs[:, 0::4] + outputs[:, 1::4] + outputs[:, 2::4] + outputs[:, 3::4]) / 4
                else:
                    outputs = model_g(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracys.append(100 * correct / total)


    model_g.train()
    return accuracy, accuracys, predicts_1, labels_1

def model_global_eval_hard(model_g, test_dataset, task_id, task_size, device, method, task_num, global_class_output, global_class_output_real):
    model_to_device(model_g, False, device)
    model_g.eval()
    #print(model_g.global_class_min_output)
    #print(model_g.client_class_min_output)
    #print(model_g.client_index)
    #print(global_class_output)
    test_dataset.getTestData_hard(global_class_output, global_class_output_real)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, total = 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            if "fcil" in method:
                outputs = model_g(imgs)
            elif "cprompt" in method:
                outputs = model_g(imgs, device=device)
                #print(outputs[0])
            elif "cfed" in method:
                outputs = model_g(imgs)
            elif "fedspace" in method:
                outputs = model_g(imgs)
                #print(outputs[0])
                #print(labels[0])
                #outputs = (outputs[:, 0::4] + outputs[:, 1::4] + outputs[:, 2::4] + outputs[:, 3::4]) / 4
                #outputs = outputs[:, ::4]
            else:
                outputs = model_g(imgs)
                
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        if step == 1:
            predicts_1 = predicts.cpu()
            labels_1 = labels.cpu()
        total += len(labels)
    accuracy = 100 * correct / total

    accuracys = []
    '''
    for i in global_class_output_real:
        test_dataset.getTestData_hard([global_class_output[global_class_output_real.index(i)]], [i])
        test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=128, num_workers=8, pin_memory=True)
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            if isinstance(device, int):
                imgs, labels = imgs.cuda(device), labels.cuda(device)
            else:
                imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                if "fcil" in method:
                    outputs = model_g(imgs)
                elif "cprompt" in method:
                    outputs = model_g(imgs, device=device)
                elif "cfed" in method:
                    outputs = model_g(imgs)
                elif "fedspace" in method:
                    outputs = model_g(imgs)
                    #outputs = (outputs[:, 0::4] + outputs[:, 1::4] + outputs[:, 2::4] + outputs[:, 3::4]) / 4
                else:
                    outputs = model_g(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracys.append(100 * correct / total)
    '''

    model_g.train()
    return accuracy, accuracys

def model_global_eval_hard_privacy(model_g, train_dataset, test_dataset, task_id, task_size, device, method, task_num, class_distribution_client, class_distribution_client_real, class_distribution_client_proportion):
    model_to_device(model_g, False, device)
    model_g.eval()
    client_index = model_g.prompt.client_index
    task_id = model_g.prompt.task_id
    #current_class = class_distribution_client[0][task_id]
    #current_class_real = class_distribution_client_real[client_index][task_id]
    
    if client_index == 0:
        client_other = 1
        known_current_class = class_distribution_client[0][task_id][10:20]
        known_current_class_real = class_distribution_client_real[0][task_id][10:20]
        known_current_class_proportion = class_distribution_client_proportion[0][task_id]
        unknown_current_class = class_distribution_client[0][task_id][10:20]
        unknown_current_class_real = class_distribution_client_real[1][task_id][0:10]
        unknown_current_class_proportion = class_distribution_client_proportion[1][task_id]
    else:
        client_other = 0
        known_current_class = class_distribution_client[1][task_id][0:10]
        known_current_class_real = class_distribution_client_real[1][task_id][0:10]
        known_current_class_proportion = class_distribution_client_proportion[1][task_id]
        unknown_current_class = class_distribution_client[1][task_id][0:10]
        unknown_current_class_real = class_distribution_client_real[0][task_id][10:20]
        unknown_current_class_proportion = class_distribution_client_proportion[0][task_id]

    train_dataset.getTrainData(known_current_class, [], [], client_index, classes_real=list(range(50,60)), classes_proportion=known_current_class_proportion)
    test_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, confidence, entropy, modified_entropy, total = 0, 0, 0, 0, 0
    topk_confidence, topk_entropy = None, None
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model_g(imgs, device=device)

        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()

        softmax_out = nn.Softmax(dim=1)(outputs)
        confidence += torch.max(softmax_out, dim=1)[0].sum()
        if topk_confidence is None:
            topk_confidence = torch.max(softmax_out, dim=1)[0]
        else:
            topk_confidence = torch.cat((topk_confidence, torch.max(softmax_out, dim=1)[0]), dim=0)

        entropy += cal_entropy(softmax_out).sum()
        if topk_entropy is None:
            topk_entropy = cal_entropy(softmax_out)
        else:
            topk_entropy = torch.cat((topk_entropy, cal_entropy(softmax_out)), dim=0)

        modified_entropy += cal_modified_entropy(softmax_out, labels)

        total += len(labels)
    
    average_correct = 100 * correct / total
    average_confidence = confidence / total
    average_entropy = entropy / total
    average_modified_entropy = modified_entropy / total
    k_confidence, _ = topk_confidence.topk(int(total * 0.1))
    #print(k_confidence)
    k_confidence = k_confidence[-1]
    k_entropy, _ = topk_entropy.topk(int(total * 0.9))
    k_entropy = k_entropy[-1]
    print("For random set of client {} on client {}".format(client_index, client_index))
    print("average_correct: {}".format(average_correct))
    print("average_confidence: {}".format(average_confidence))
    print("average_entropy: {}".format(average_entropy))
    print("average_modified_entropy: {}".format(average_modified_entropy))
    print("k_confidence: {}".format(k_confidence))
    print("k_entropy: {}".format(k_entropy))

    
    train_dataset.getTrainData(known_current_class, [], [], client_index, classes_real=known_current_class_real, classes_proportion=known_current_class_proportion)
    test_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, confidence, entropy, modified_entropy, total, train_num_confidence, train_num_entropy = 0, 0, 0, 0, 0, 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model_g(imgs, device=device)

        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()

        softmax_out = nn.Softmax(dim=1)(outputs)
        confidence += torch.max(softmax_out, dim=1)[0].sum()
        train_num_out_confidence = torch.max(softmax_out, dim=1)[0]
        #print(train_num_out_confidence[train_num_out_confidence[:] >= k_confidence.repeat(train_num_out_confidence.size(0))].size(0))
        train_num_confidence += train_num_out_confidence[train_num_out_confidence[:] >= k_confidence.repeat(train_num_out_confidence.size(0))].size(0)
        
        entropy += cal_entropy(softmax_out).sum()
        train_num_out_entropy = cal_entropy(softmax_out)
        train_num_entropy += train_num_out_entropy[train_num_out_entropy[:] <= k_entropy.repeat(train_num_out_entropy.size(0))].size(0)

        modified_entropy += cal_modified_entropy(softmax_out, labels)

        total += len(labels)
    
    average_correct = 100 * correct / total
    average_confidence = confidence / total
    average_entropy = entropy / total
    average_modified_entropy = modified_entropy / total
    print("For training set of client {} on client {}".format(client_index, client_index))
    print("average_correct: {}".format(average_correct))
    print("average_confidence: {}".format(average_confidence))
    print("average_entropy: {}".format(average_entropy))
    print("average_modified_entropy: {}".format(average_modified_entropy))
    print("{} / {}".format(train_num_confidence, total))
    print("{} / {}".format(train_num_entropy, total))

    test_dataset.getTestData_hard(known_current_class, known_current_class_real)
    test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, confidence, entropy, modified_entropy, total, train_num_confidence, train_num_entropy = 0, 0, 0, 0, 0, 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model_g(imgs, device=device)

        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()

        softmax_out = nn.Softmax(dim=1)(outputs)
        confidence += torch.max(softmax_out, dim=1)[0].sum()
        train_num_out_confidence = torch.max(softmax_out, dim=1)[0]
        #print(train_num_out_confidence[train_num_out_confidence[:] >= k_confidence.repeat(train_num_out_confidence.size(0))].size(0))
        train_num_confidence += train_num_out_confidence[train_num_out_confidence[:] >= k_confidence.repeat(train_num_out_confidence.size(0))].size(0)

        entropy += cal_entropy(softmax_out).sum()
        train_num_out_entropy = cal_entropy(softmax_out)
        train_num_entropy += train_num_out_entropy[train_num_out_entropy[:] <= k_entropy.repeat(train_num_out_entropy.size(0))].size(0)

        modified_entropy += cal_modified_entropy(softmax_out, labels)

        total += len(labels)
    
    average_correct = 100 * correct / total
    average_confidence = confidence / total
    average_entropy = entropy / total
    average_modified_entropy = modified_entropy / total
    print("For test set of client {} on client {}".format(client_index, client_index))
    print("average_correct: {}".format(average_correct))
    print("average_confidence: {}".format(average_confidence))
    print("average_entropy: {}".format(average_entropy))
    print("average_modified_entropy: {}".format(average_modified_entropy))
    print("{} / {}".format(train_num_confidence, total))
    print("{} / {}".format(train_num_entropy, total))

    train_dataset.getTrainData(unknown_current_class, [], [], client_index, classes_real=unknown_current_class_real, classes_proportion=unknown_current_class_proportion)
    test_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    correct, confidence, entropy, modified_entropy, total, train_num_confidence, train_num_entropy = 0, 0, 0, 0, 0, 0, 0
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        if isinstance(device, int):
            imgs, labels = imgs.cuda(device), labels.cuda(device)
        else:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model_g(imgs, device=device)

        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()

        softmax_out = nn.Softmax(dim=1)(outputs)
        confidence += torch.max(softmax_out, dim=1)[0].sum()
        train_num_out_confidence = torch.max(softmax_out, dim=1)[0]
        #print(train_num_out_confidence[train_num_out_confidence[:] >= k_confidence.repeat(train_num_out_confidence.size(0))].size(0))
        train_num_confidence += train_num_out_confidence[train_num_out_confidence[:] >= k_confidence.repeat(train_num_out_confidence.size(0))].size(0)

        entropy += cal_entropy(softmax_out).sum()
        train_num_out_entropy = cal_entropy(softmax_out)
        train_num_entropy += train_num_out_entropy[train_num_out_entropy[:] <= k_entropy.repeat(train_num_out_entropy.size(0))].size(0)

        modified_entropy += cal_modified_entropy(softmax_out, labels)

        total += len(labels)
    
    average_correct = 100 * correct / total
    average_confidence = confidence / total
    average_entropy = entropy / total
    average_modified_entropy = modified_entropy / total
    print("For training set of client {} on client {}".format(client_other, client_index))
    print("average_correct: {}".format(average_correct))
    print("average_confidence: {}".format(average_confidence))
    print("average_entropy: {}".format(average_entropy))
    print("average_modified_entropy: {}".format(average_modified_entropy))
    print("{} / {}".format(train_num_confidence, total))
    print("{} / {}".format(train_num_entropy, total))

    return average_correct, []

def cal_entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def cal_modified_entropy(input_, label_, num_classes=500):
    nt_positions = torch.arange(0, num_classes).to(input_.device)
    nt_positions = nt_positions.repeat(input_.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != label_.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)
    logits = torch.gather(input_, 1, nt_positions)

    nt_positions_true = torch.arange(0, num_classes).to(input_.device)
    nt_positions_true = nt_positions_true.repeat(input_.size(0), 1)
    nt_positions_true = nt_positions_true[nt_positions_true[:, :] == label_.view(-1, 1)]
    nt_positions_true = nt_positions_true.view(-1, 1)
    logits_true = torch.gather(input_, 1, nt_positions_true)

    term_1 = -logits * torch.log(1 - logits + 1e-5)
    term_2 = -(1 - logits_true) * torch.log(logits_true + 1e-5)
    term_1 = term_1.sum()
    term_2 = term_2.sum()
    return term_1 + term_2
