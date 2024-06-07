import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="name of dataset")
    parser.add_argument('--method', type=str, default='glfc', help="name of method")
    parser.add_argument('--iid_level', type=int, default=6, help='non-iid level')
    parser.add_argument('--sim', type=int, default=0, help="control task similarity level")
    parser.add_argument('--numclass', type=int, default=10, help="number of classes in FCL process")
    parser.add_argument('--class_per_task', type=int, default=10, help="number of classes per task")
    parser.add_argument('--img_size', type=int, default=32, help="size of images")
    parser.add_argument('--device', nargs="+", type=int, default=[0, 1, 2, 3], help="GPU ID, -1 for CPU")
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--task_size', type=int, default=10, help='number of classes each task')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--memory_size', type=int, default=500, help='size of exemplar memory')
    parser.add_argument('--epochs_local', type=int, default=20, help='local epochs of each global round')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='learning rate')
    parser.add_argument('--global_update_lr', type=float, default=0.0001, help='global_update_lr')
    parser.add_argument('--num_clients', type=int, default=30, help='initial number of clients')
    parser.add_argument('--local_clients', type=int, default=10, help='number of selected clients each round')
    parser.add_argument('--epochs_global', type=int, default=100, help='total number of global rounds')
    parser.add_argument('--tasks_global', type=int, default=10, help='min number of rounds for a task')
    parser.add_argument('--prompt_param', nargs="+", type=int, default=[25, 10, 10, 8, 0, 0, 6], help='prompt pool size')
    #CODAP: [100, 8, 1, 0, 0, 6] Dual: [25, 20, 1, 0, 0, 10] L2P: [20, 20, 1, 0, 0, 6] CODAP_ours_v1: [25, 10, 10, 8, 0, 0, 6]
    parser.add_argument('--prompt_flag', type=str, default='codap', help='method of prompt')
    parser.add_argument('--global_weight', type=float, default=0.5, help="weight of the global model")
    #global_weight
    parser.add_argument('--centralized_pretrain', action='store_true', default=False, help="whether pretrain")   
    #centralized_pretrain
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')
    #optimizer
    parser.add_argument('--centralized_fractal_pretrain_steps', type=int, default=10000000000, help="number of centralized fractal pretrain")
    #centralized_fractal_pretrain_steps
    parser.add_argument('--temp', default=1, type=float, help='training time temperature')
    #temp
    parser.add_argument('--total_classes', type=int, default=100, help='total classes')
    #total_classes
    parser.add_argument('--repr_loss_temp', default=1., type=float, help='representation loss temp')
    #repr_loss_temp
    parser.add_argument('--lambda_proto_aug', default=1e-4, type=float, help='protoAug loss weight')
    #lambda_proto_aug
    parser.add_argument('--lambda_repr_loss', default=1e-2, type=float, help='representation loss weight')
    #lambda_repr_loss
    parser.add_argument('--ema_global', default=0.95, type=float, help='exponential moving average smoothing factor')
    #ema_global
    parser.add_argument('--update_teacher_step', default=1, type=int, help='')
    parser.add_argument('--update_teacher_ema', default=0.2, type=int, help='')
    parser.add_argument('--easy', type=int, default=0, help='')
    #schedule
    parser.add_argument("--dataroot", type=str, default='dataset', help='root of data')
    parser.add_argument("--validation", action='store_true', default=False, help="whether validate")
    parser.add_argument("--imbalance", type=str, default='none', help='methods to deal with class imbalance')
    args = parser.parse_args()
    return args