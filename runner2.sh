python main.py --name target-dog-reg3-5000-lr1e-1 --wandb  --device cuda --dataset 5000 --log_freq 25 --lr 1e-1 --epochs 1000 --batch_size 128 --target_class dog --regularizer l2 --reg_weight 1e-4
python main.py --name target-dog-reg4-5000-lr1e-1 --wandb  --device cuda --dataset 5000 --log_freq 25 --lr 1e-1 --epochs 1000 --batch_size 128 --target_class dog --regularizer l2 --reg_weight 1e-3
