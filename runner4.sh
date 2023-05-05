CUDA_VISIBLE_DEVICES=1 python main.py --name maxem-v2-5000-lr1e-1 --wandb  --device cuda --dataset 5000 --log_freq 50 --lr 1e-1 --epochs 1000 --batch_size 128 --trainer max_embedding_trainer
