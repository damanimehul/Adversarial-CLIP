import argparse 
import torch 
from generate_dataset import load_dataset, make_dataset, ImageCaptionDataset
from train import TargetClassTrainer, MaxEmbeddingTrainer, MaxTargetProbTrainer
from utils import WandbLogger, preprocess_dataset
from transformers import AutoTokenizer,AutoProcessor,CLIPTextModelWithProjection,CLIPVisionModelWithProjection
import os 
import pickle 

if __name__=='__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--name',type=str,default='Simple-Run',help='Give a name to the experiment')
    parser.add_argument('--dataset',type=str,default=None,help='Specify pickle name to load dataset from. If this is provided, no new images will be generated')  
    parser.add_argument('--caption_max_index',type=int,default=2,help='Upto which index of coco should be used for captions')
    parser.add_argument('--num_images_per_caption',type=int,default=5,help='Number of images to generate per caption')
    parser.add_argument('--test_size',type=float,default=0.2,help='Test size for train test split')
    parser.add_argument('--dname',type=str,default='dataset',help='Name of the dataset to be saved')
    parser.add_argument('--wandb',action='store_true',default=False,help='Log on wandb') 
    parser.add_argument('--batch_size',type=int,default=32,help='Batch size for training')
    parser.add_argument('--epochs',type=int,default=10,help='Number of epochs for training')
    parser.add_argument('--num_iters',type=int,default=50,help='Number of iterations per epoch')
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate for training')
    parser.add_argument('--trainer',type=str,default='target_class_trainer',help='Which trainer to use')
    parser.add_argument('--target_class',type=str,default='dog',help='Target class for training, only used in target_class_trainer')
    parser.add_argument('--log_freq',type=int,default=10,help='Log frequency for training')
    parser.add_argument('--device',type=str,default='cpu',help='Device to use for training')
    parser.add_argument('--regularizer',type=str,default='none',help='Regularizer to use for training') 
    parser.add_argument('--reg_weight',type=float,default=0.01,help='Regularizer weight to use for training')
    parser.add_argument('--temperature',type=float,default=0,help='Temperature to scale raw logits')

    args = parser.parse_args() 

    if args.dataset is not None:
        train_data,test_data = load_dataset(name=args.dataset) 
        print(f'Loaded dataset with {len(train_data)} train samples and {len(test_data)} test samples')
    else:
        train_data,test_data = make_dataset(name=args.dname,caption_max_index=args.caption_max_index,
                                            num_images_per_caption=args.num_images_per_caption,test_size=args.test_size)
    
    train_data,test_data = preprocess_dataset(train_data,test_data)
    print('Preprocessed dataset by removing NSFW samples') 
    if args.trainer is not None and args.trainer!='none':
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=min(256,len(test_data)), shuffle=True)
        
        logger = WandbLogger(args.wandb,args,'results/{}'.format(args.name),args.epochs)

        # Make directory 
        if not os.path.exists('results/{}'.format(args.name)):
            os.makedirs('results/{}'.format(args.name))
        
        # text model
        text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # vision model
        vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if args.device == 'cuda':
            text_model = text_model.to(args.device)
            vision_model = vision_model.to(args.device)

        if args.trainer == 'target_class_trainer':
            trainer = TargetClassTrainer(device=args.device,dataloader=train_dataloader,test_dataloader=test_dataloader,logger=logger,text_model=text_model, visual_model=vision_model, 
                                         text_tokenizer=text_tokenizer, vision_processor=vision_processor,num_iters=args.num_iters,epochs=args.epochs,lr=args.lr,
                                         log_freq=args.log_freq,target_class=args.target_class,regularizer=args.regularizer,reg_weight=args.reg_weight)
        elif args.trainer == 'max_embedding_trainer':
            trainer = MaxEmbeddingTrainer(device=args.device,dataloader=train_dataloader,test_dataloader=test_dataloader,logger=logger,text_model=text_model, visual_model=vision_model, 
                                         text_tokenizer=text_tokenizer, vision_processor=vision_processor,num_iters=args.num_iters,epochs=args.epochs,lr=args.lr,
                                         log_freq=args.log_freq,regularizer=args.regularizer,reg_weight=args.reg_weight)

        elif args.trainer == 'max_target_prob':
            trainer = MaxTargetProbTrainer(device=args.device,dataloader=train_dataloader,test_dataloader=test_dataloader,logger=logger,text_model=text_model, visual_model=vision_model, 
                                         text_tokenizer=text_tokenizer, vision_processor=vision_processor,num_iters=args.num_iters,epochs=args.epochs,lr=args.lr,
                                         log_freq=args.log_freq,target_class=args.target_class,regularizer=args.regularizer,reg_weight=args.reg_weight,temperature=args.temperature)

        perturbations = trainer.train()
        with open('results/{}/{}'.format(args.name,'v.pickle'), 'wb') as f:
            pickle.dump(perturbations, f)

        logger.write_csv() 
    else :
        print('No trainer specified, only dataset created if that was required')

