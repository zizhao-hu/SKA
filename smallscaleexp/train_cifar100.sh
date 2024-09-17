#!/bin/bash

# Command 1
python train_cifar100.py --nowandb --net vit_small --n_epochs 200

python train_cifar100.py --nowandb --net vit_small_stu --n_epochs 200

python train_cifar100.py --nowandb --net cvt --n_epochs 200

python train_cifar100.py --nowandb --net cvt_stu --n_epochs 200

python train_cifar100.py --nowandb --net swin --n_epochs 400

python train_cifar100.py --nowandb --net swin_stu --n_epochs 400

python train_cifar100.py --nowandb --net cait --n_epochs 400

python train_cifar100.py --nowandb --net cait_stu --n_epochs 400
# ... (Add more commands as needed)