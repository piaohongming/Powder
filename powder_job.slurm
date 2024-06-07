python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap_2d_v2 --method cprompt_powder_v2 --batch_size 64 \
    --prompt_param 25 10 10 8 0 0 6 10 8\
    --epochs_local 10

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap_2d_v2 --method cprompt_powder_v2_direct --batch_size 64 \
    --prompt_param 25 10 10 8 0 0 6 10 8\
    --epochs_local 10 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 1750 --class_per_task 35 --dataset DomainNet --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 30 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap_2d_v2 --method cprompt_powder_domainnet_v2 --batch_size 32 \
    --prompt_param 25 10 10 8 0 0 6 10 8\
    --epochs_local 4 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 1750 --class_per_task 35 --dataset DomainNet --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 30 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap_2d_v2 --method cprompt_powder_domainnet_v2_direct --batch_size 32 \
    --prompt_param 25 10 10 8 0 0 6 10 8\
    --epochs_local 4 