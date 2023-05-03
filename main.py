import argparse 
import torch 
from generate_dataset import load_dataset, make_dataset, ImageCaptionDataset
from train import TargetClassTrainer 
from utils import WandbLogger
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

    args = parser.parse_args() 

    if args.dataset is not None:
        train_data,test_data = load_dataset(name=args.dataset) 
        print(f'Loaded dataset with {len(train_data)} train samples and {len(test_data)} test samples')
    else:
        train_data,test_data = make_dataset(name=args.dname,caption_max_index=args.caption_max_index,
                                            num_images_per_caption=args.num_images_per_caption,test_size=args.test_size)
        
    if args.trainer is not None and args.trainer!='none':
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=True)
        if args.device == 'cuda':
            train_dataloader = train_dataloader.to(args.device)
            test_dataloader = test_dataloader.to(args.device)

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
            text_tokenizer = text_tokenizer.to(args.device)
            text_model = text_model.to(args.device)
            vision_model = vision_model.to(args.device)
            vision_processor = vision_processor.to(args.device)

        if args.trainer == 'target_class_trainer':
            trainer = TargetClassTrainer(device=args.device,dataloader=train_dataloader,test_dataloader=test_dataloader,logger=logger,text_model=text_model, visual_model=vision_model, 
                                         text_tokenizer=text_tokenizer, vision_processor=vision_processor,num_iters=args.num_iters,epochs=args.epochs,lr=args.lr,
                                         log_freq=args.log_freq,target_class=args.target_class)
            
        perturbations = trainer.train()
        with open('results/{}/{}'.format(args.name,'v.pickle'), 'wb') as f:
            pickle.dump(perturbations, f)

        logger.write_csv() 
    else :
        print('No trainer specified, only dataset created if that was required')

