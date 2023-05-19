import torch 
import numpy as np 
from PIL import Image 
from diffusers import StableDiffusionPipeline
import pickle 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

MSCOCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                  'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 
                  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                  'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
                  'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                  'sports ball','kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                  'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
                  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                  'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
                  'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 
                  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                  'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
                  'hair drier', 'toothbrush', 'piggy bank','jarm','terrorist']

class ImageCaptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, caption = self.data[index]
        return image, caption
    
def get_caption(s): 
    if s[0] in 'aeiou':
        return f'A photo of an {s}.'
    else:
        return f'A photo of a {s}.'
    
@torch.no_grad() 
def generate_images(run_sd,caption_max_index=2,num_images_per_caption=5):
    with torch.no_grad():
      data= []
      for i in range(caption_max_index):  
          caption = MSCOCO_CLASSES[i]
          sd_images = run_sd(caption,num_images_per_prompt=num_images_per_caption).images   
          for i in sd_images: 
            ## Convert to PIL to downsample image
            #img_pil = Image.fromarray((i * 255).astype(np.uint8))
            # Downsampling to 224 
            ri = i.resize((224, 224), resample=Image.BICUBIC) 
            # Convert back to np array 
            ri = np.array(ri)
            # Store in the dataset as a torch tensor 
            ri = ri.astype(np.float32)
            data.append((torch.from_numpy(ri),caption)) 
      return data 

def build_stable_diffusion(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    return pipe 

def load_dataset(name='dataset'): 
  # Load the train dataset
  with open('datasets/train_{}.pickle'.format(name), 'rb') as f:
      train_dataset = pickle.load(f)

  # Load the test dataset
  with open('datasets/test_{}.pickle'.format(name), 'rb') as f:
      test_dataset = pickle.load(f)
  return train_dataset, test_dataset

def make_dataset(name='dataset',caption_max_index=2,num_images_per_caption=5,test_size=0.2):
    sd = build_stable_diffusion() 
    data = generate_images(sd,caption_max_index,num_images_per_caption) 
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_dataset = ImageCaptionDataset(train_data)
    test_dataset = ImageCaptionDataset(test_data)
    # Save the train dataset
    with open('datasets/train_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(train_dataset, f)

    # Save the test dataset
    with open('datasets/test_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(test_dataset, f)

    return train_dataset, test_dataset

if __name__ =='__main__': 
   #pipe = build_stable_diffusion() 
    #data = generate_images(pipe) 
    data= [[torch.ones(17),'hi'] for _ in range(100)]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    import pickle

    # Save the train dataset
    with open('train_dataset.pickle', 'wb') as f:
        pickle.dump(train_data, f)

    # Save the test dataset
    with open('test_dataset.pickle', 'wb') as f:
        pickle.dump(test_dataset, f)
