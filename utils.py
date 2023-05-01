import matplotlib.pyplot as plt
import torch 
import wandb  
import time 
import numpy as np 

MEANS = torch.tensor([
        0.48145466,
        0.4578275,
        0.40821073] )
STDS = torch.tensor([
        0.26862954,
        0.26130258,
        0.27577711
        ]) 


def visualize_images(dataloader,num_samples=10):
    for i in range(len(batch)//num_samples):
        for batch in dataloader:
            imgs,captions = batch
            for i in imgs:
                np_im = imgs[i].numpy() 
                print('Showing', captions[i])
                if np.max(np_im) > 1:
                    np_im = np_im/255
                plt.imshow(np_im)
                plt.show()

def denorm(im): 
        # Normalize an image similar to clip-preprocessing
        if len(im.shape) ==4:
            assert im.shape[0] ==1 
            im = im.squeeze(dim=0) 

        im = im.permute(1,2,0)
        new_im = (im*STDS + MEANS) * 255 
        return new_im 

def norm(im): 
    # Denormalize an image similar to clip-preprocessing
    if len(im.shape) ==4:
        assert im.shape[0] ==1 
        im = im.squeeze(dim=0) 
    new_im = (im/255 -  MEANS)/ STDS
    new_im = new_im.permute(2,0,1)
    return new_im

## Setup a wandb logger class
class WandbLogger():
    def __init__(self,use_wandb=False,args={}):
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.args = args
            name = self.args.name + time.strftime("_%Y-%m-%d") 
            self.run = wandb.init(project='adversarial-clip',name=name,config=args)
        
    def update(self,iter,log_dict):
        if self.use_wandb:
            new_log_dict = {} 
            # Check if there are images in the dict
            for k,v in log_dict.items():
                # Check if type v is list 
                if isinstance(v,list):
                    for element in v:
                        new_log_dict[k] = wandb.Image(element)
                else :
                    new_log_dict[k] = v

            self.run.log(new_log_dict,step=iter)

def process_img(img):
    # Converts torch image to np image of shape 224,224,3 and rgb values between 0 and 1
    img = denorm(torch.squeeze(img)) 
    img = img.numpy()
    img = np.reshape(img,(224,224,3)) 
    reshaped_img = np.reshape(img,-1)
    if max(reshaped_img) > 1:
        img = img/255
    return img 