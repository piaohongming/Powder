#You can change the "fcil" to "cfed", "fcilweit", "fedspace" for other baselines (GLFC, CFeD, FedWeIT, FedSpace).

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcil_imagenet --batch_size 64 --epochs_local 10 \
    --global_weight 1 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcilnotran_imagenet --batch_size 64 --epochs_local 10 \
    --global_weight 1

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcil_imagenet_sharedcodap --batch_size 64 --epochs_local 10 \
    --global_weight 1 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcilnotran_imagenet_sharedcodap --batch_size 64 --epochs_local 10 \
    --global_weight 1

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcil_imagenet_sharedprompt --batch_size 64 --epochs_local 10 \
    --global_weight 1 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcilnotran_imagenet_sharedprompt --batch_size 64 --epochs_local 10 \
    --global_weight 1

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcil_imagenet_sharedfc --batch_size 64 --epochs_local 10 \
    --global_weight 1 

python fl_main_hard.py --device 0 --global_update_lr 100 \
    --img_size 224 --numclass 500 --class_per_task 20 --dataset ImageNet_R --easy 0 \
    --tasks_global 3 --num_clients 5 --epochs_global 15 --local_client 5 \
    --learning_rate 0.005 --method fcilnotran_imagenet_sharedfc --batch_size 64 --epochs_local 10 \
    --global_weight 1

