import argparse 
import torch 
from generate_dataset import load_dataset, make_dataset, ImageCaptionDataset
from train import TargetClassTrainer 
from utils import WandbLogger 
from transformers import AutoTokenizer,AutoProcessor,CLIPTextModelWithProjection,CLIPVisionModelWithProjection

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
    parser.add_argument('--lr',type=float,default=1e-3,help='Learning rate for training')
    parser.add_argument('--trainer',type=str,default='target_class_trainer',help='Which trainer to use')
    parser.add_argument('--target_class',type=str,default='dog',help='Target class for training, only used in target_class_trainer')

    args = parser.parse_args() 

    if args.dataset is not None:
        train_data,test_data = load_dataset(name=args.dataset) 
        print(f'Loaded dataset with {len(train_data)} train samples and {len(test_data)} test samples')
    else:
        train_data,test_data = make_dataset(name=args.dname,caption_max_index=args.caption_max_index,
                                            num_images_per_caption=args.num_images_per_caption,test_size=args.test_size)
    
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    logger = WandbLogger(args.wandb,args)
    
    # text model
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    text_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # vision model
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if args.trainer == 'target_class_trainer':
        trainer = TargetClassTrainer(train_dataloader,test_dataloader,logger,text_model, vision_model, text_tokenizer, 
                 vision_processor,args.epochs,args.lr,args.target_class)
        
    trainer.train()

