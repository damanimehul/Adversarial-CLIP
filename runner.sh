python main.py --name target-v1-5000-lr1e-3 --wandb  --device cuda --dataset 5000 --log_freq 25 --lr 1e-3 --epochs 1000 --batch_size 128
python main.py --name target-v1-1000-lr1e-3 --wandb  --device cuda --dataset 1000 --log_freq 25 --lr 1e-3 --epochs 1000 --batch_size 128
python main.py --name target-v1-5000-lr1e-2 --wandb  --device cuda --dataset 5000 --log_freq 25 --lr 1e-2 --epochs 1000 --batch_size 128
python main.py --name target-v1-1000-lr1e-2 --wandb  --device cuda --dataset 1000 --log_freq 25 --lr 1e-2 --epochs 1000 --batch_size 128