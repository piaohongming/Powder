import numpy as np
pathes = []
pathes_notran = []

task_list_forward = []
task_list_backward = []

def acc():
    acc_list = []
    acc_list_list = []
    for i in range(len(pathes)):
        path = pathes[i]
        with open(path) as f:
            line_index = 0
            content = f.read()
            x = []
            y = []
            y_round = []
            last_round = -1
            for line in content.split('\n'):
                if line_index == len(content.split('\n')) - 1:
                    x.append(round)
                    y.append(sum(y_round) / len(y_round))
                    break
                if line_index == 0:
                    line_index = line_index + 1
                    continue
                else:
                    line_index = line_index + 1
                round = eval(line.split(' ')[5].replace(',', ''))
                if last_round != round and last_round != -1:
                    if len(y_round) > 0:
                        x.append(round - 1)
                        y.append(sum(y_round) / len(y_round))
                        y_round = []
                    else:
                        x.append(round - 1)
                        y.append(0)
                        y_round = []
                if (round + 1) % 3 == 0:
                    y_round.append(eval(line.split('=')[1].replace(' ', '').replace('%', '')))
                else:
                    y_round = []
                last_round = round
            acc_list.append(sum(y) / len(y))
            acc_list_list.append(np.array(y))
    print("********ACC********")
    print(np.mean(acc_list))
    print(np.std(acc_list))
    avg_acc_list, std_acc_list = list(np.stack(acc_list_list).mean(axis=0)), list(np.std(np.stack(acc_list_list), axis=0))
    print(avg_acc_list)  
    print(std_acc_list) 


def forget(): 
    forget_list = []
    forget_list_list = []
    for i in range(len(pathes)):
        path = pathes[i]   
        with open(path) as f:
            line_index = 0
            content = f.read()
            x_forget = []
            y_forget = []
            y_forget_round = []
            task_acc_dict = {}
            last_round = -1
            for line in content.split('\n'):
                if line_index == len(content.split('\n')) - 1:
                    x_forget.append(round)
                    y_forget.append(sum(y_forget_round) / len(y_forget_round))
                    break
                if line_index == 0:
                    line_index = line_index + 1
                    continue
                else:
                    line_index = line_index + 1
                round = eval(line.split(' ')[5].replace(',', ''))
                client_id = eval(line.split(' ')[1].replace(',', ''))
                task_id = eval(line.split(' ')[3].replace(',', ''))
                global_task_id = task_id * 5 + client_id
                if last_round != round:
                    if len(y_forget_round) > 0:
                        x_forget.append(round - 1)
                        y_forget.append(sum(y_forget_round) / len(y_forget_round))
                        y_forget_round = []
                    else:
                        y_forget_round = []
                if (round + 1) % 3 == 0:
                    if global_task_id in task_acc_dict.keys():
                        y_max_previous = max(task_acc_dict[global_task_id])
                        y_now = eval(line.split('=')[1].replace(' ', '').replace('%', ''))
                        y_forget_round.append(y_max_previous - y_now)
                        task_acc_dict[global_task_id].append(y_now)
                    else:
                        y_forget_round.append(0)
                        task_acc_dict[global_task_id] = []
                        task_acc_dict[global_task_id].append(eval(line.split('=')[1].replace(' ', '').replace('%', '')))
                else:
                    y_forget_round = []
                last_round = round
            forget_list.append(sum(y_forget) / len(y_forget))
            forget_list_list.append(np.array(y_forget))
    print("********FORGET********")
    print(np.mean(forget_list))
    print(np.std(forget_list))
    avg_forget_list, std_forget_list = list(np.stack(forget_list_list).mean(axis=0)), list(np.std(np.stack(forget_list_list), axis=0))
    print(avg_forget_list)  
    print(std_forget_list)


def forward_transfer():
    forward_list = []
    forward_list_list = []
    for i in range(len(pathes)):
        path = pathes[i]
        path_notran = pathes_notran[i]
        with open(path) as f1:
            with open(path_notran) as f2:
                line_index = 0
                content = f1.read()
                x_forward = []
                y_forward = []
                y_forward_round = []
                new_task = task_list_forward[i]
                last_round = -1
                for line in content.split('\n'):
                    if line_index == len(content.split('\n')) - 1:
                        x_forward.append(round)
                        y_forward.append(sum(y_forward_round) / len(y_forward_round))
                        break
                    if line_index == 0:
                        line_index = line_index + 1
                        continue
                    else:
                        line_index = line_index + 1
                    round = eval(line.split(' ')[5].replace(',', ''))
                    client_id = eval(line.split(' ')[1].replace(',', ''))
                    task_id = eval(line.split(' ')[3].replace(',', ''))
                    global_task_id = task_id * 5 + client_id
                    if last_round != round:
                        if len(y_forward_round) > 0:
                            x_forward.append(round - 1)
                            y_forward.append(sum(y_forward_round) / len(y_forward_round))
                            y_forward_round = []
                        else:
                            y_forward_round = []
                    
                    if (round + 1) % 3 == 0 and global_task_id in new_task[round]:
                        y_forward_round.append(eval(line.split('=')[1].replace(' ', '').replace('%', '')))
                    
                    last_round = round

                new_task = task_list_forward[i]
                line_index = 0
                content = f2.read()
                x_normal = []
                y_normal = []
                y_normal_round = []
                last_round = -1
                
                for line in content.split('\n'):
                    if line_index == len(content.split('\n')) - 1:
                        x_normal.append(round)
                        y_normal.append(sum(y_normal_round) / len(y_normal_round))
                        break
                    if line_index == 0:
                        line_index = line_index + 1
                        continue
                    else:
                        line_index = line_index + 1
                    round = eval(line.split(' ')[5].replace(',', ''))
                    client_id = eval(line.split(' ')[1].replace(',', ''))
                    task_id = eval(line.split(' ')[3].replace(',', ''))
                    global_task_id = task_id * 5 + client_id
                    if last_round != round:
                        if len(y_normal_round) > 0:
                            x_normal.append(round - 1)
                            y_normal.append(sum(y_normal_round) / len(y_normal_round))
                            y_normal_round = []
                        else:
                            y_normal_round = []
                    if (round + 1) % 3 == 0 and global_task_id in new_task[round]:
                        y_normal_round.append(eval(line.split('=')[1].replace(' ', '').replace('%', '')))
                    
                    last_round = round
                
                y_transfer = []
                for i in range(len(y_normal)):
                    if i == len(y_normal) - 1:
                        y_transfer.append(y_forward[i] - y_normal[0])
                    else:
                        y_transfer.append(y_forward[i] - y_normal[i])
                
                forward_list.append(sum(y_transfer) / len(y_transfer))
                forward_list_list.append(y_transfer)
    print("********FORWARD********")  
    print(np.mean(forward_list))
    print(np.std(forward_list))
    avg_forward_list, std_forward_list = list(np.stack(forward_list_list).mean(axis=0)), list(np.std(np.stack(forward_list_list), axis=0))
    print(avg_forward_list)  
    print(std_forward_list)
    return forward_list, forward_list_list
        

def backward_transfer():
    backward_list = []
    backward_list_list = []
    for i in range(len(pathes)):
        path = pathes[i]
        with open(path) as f:
            line_index = 0
            content = f.read()
            x_backward = []
            y_backward = []
            y_backward_round = []
            task_acc_dict = {}
            
            finished_task = task_list_backward[i]
            last_round = -1
            for line in content.split('\n'):
                if line_index == len(content.split('\n')) - 1:
                    x_backward.append(round)
                    y_backward.append(sum(y_backward_round) / len(y_backward_round))
                    break
                if line_index == 0:
                    line_index = line_index + 1
                    continue
                else:
                    line_index = line_index + 1
                round = eval(line.split(' ')[5].replace(',', ''))
                client_id = eval(line.split(' ')[1].replace(',', ''))
                task_id = eval(line.split(' ')[3].replace(',', ''))
                global_task_id = task_id * 5 + client_id
                if last_round != round:
                    if len(y_backward_round) > 0:
                        x_backward.append(round - 1)
                        y_backward.append(sum(y_backward_round) / len(y_backward_round))
                        y_backward_round = []
                    else:
                        y_backward_round = []
                if (round + 1) % 3 == 0:
                    if global_task_id in task_acc_dict.keys():
                        y_previous = task_acc_dict[global_task_id]
                        y_now = eval(line.split('=')[1].replace(' ', '').replace('%', ''))
                        y_backward_round.append(y_now - y_previous)
                    else:
                        pass
                    if global_task_id in finished_task[round]:
                        task_acc_dict[global_task_id] = eval(line.split('=')[1].replace(' ', '').replace('%', ''))             
                else:
                    y_backward_round = []
                last_round = round
            backward_list.append(sum(y_backward) / len(y_backward))
            backward_list_list.append(y_backward)
    print("********BACKWARD********")
    print(np.mean(backward_list))
    print(np.std(backward_list))
    avg_backward_list, std_backward_list = list(np.stack(backward_list_list).mean(axis=0)), list(np.std(np.stack(backward_list_list), axis=0))
    print(avg_backward_list)  
    print(std_backward_list)
    return backward_list, backward_list_list

def calculate_total_transfer(forward_list_list, backward_list_list):
    total_list = []
    total_list_list = []
    for i in range(len(backward_list_list)):
        total_list_list_0 = []
        for j in range(len(backward_list_list[i])):
            total_list_list_0.append(forward_list_list[i][j] + backward_list_list[i][j])
        total_list_list_0.append(forward_list_list[i][-1])
        total_list.append(sum(total_list_list_0) / len(total_list_list_0))
        total_list_list.append(total_list_list_0)
    print("********TOTAL TRANSFER********")
    print(np.mean(total_list))
    print(np.std(total_list))
    avg_total_list, std_total_list = list(np.stack(total_list_list).mean(axis=0)), list(np.std(np.stack(total_list_list), axis=0))
    print(avg_total_list)  
    print(std_total_list)
    return



acc()
forget()
forward_list, forward_list_list = forward_transfer()
backward_list, backward_list_list = backward_transfer()
calculate_total_transfer(forward_list_list, backward_list_list)