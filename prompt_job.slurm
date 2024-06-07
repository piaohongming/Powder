python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag l2p --method cprompt_l2p_imagenet --batch_size 64 \
    --prompt_param 20 20 1 0 0 6\
    --epochs_local 10 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag l2p --method cprompt_l2p_imagenet_direct --batch_size 64 \
    --prompt_param 20 20 1 0 0 6\
    --epochs_local 10 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag dual --method cprompt_dual_imagenet --batch_size 64 \
    --prompt_param 25 20 1 0 0 10\
    --epochs_local 10 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag dual --method cprompt_dual_imagenet_direct --batch_size 64 \
    --prompt_param 25 20 1 0 0 10\
    --epochs_local 10 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap --method cprompt_codap_imagenet --batch_size 64 \
    --prompt_param 25 10 10 8 0 0 6\
    --epochs_local 10

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag codap --method cprompt_codap_imagenet_direct --batch_size 64 \
    --prompt_param 25 10 10 8 0 0 6\
    --epochs_local 10

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag cprompt --method cprompt_imagenet --batch_size 64 \
    --prompt_param 25 10 10 8 0 0 6\
    --epochs_local 10

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_clients 5 \
    --learning_rate 0.005 --prompt_flag cprompt --method cprompt_imagenet_direct --batch_size 64 \
    --prompt_param 25 10 10 8 0 0 6\
    --epochs_local 10