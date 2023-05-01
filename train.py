import torch 
from utils import MEANS,STDS,norm,denorm,process_img
import numpy as np 

class BaseTrainer():

    def __init__(self, dataloader,test_dataloader,logger,text_model, visual_model, text_tokenizer, 
                 vision_processor,num_iters,lr):
        self.text_model = text_model 
        self.visual_model = visual_model 
        self.text_tokenizer = text_tokenizer 
        self.vision_processor = vision_processor
        self.num_iters = num_iters
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.log_freq =  5
        self.logger = logger  
        self.num_iters = num_iters
        self.lr = lr

    def train_setup(self):
        raise NotImplementedError

    def get_text_embeddings(self,inputs):
        # Get clip embeddings from a list of str inputs 
        preprocess_text = self.text_tokenizer(inputs, padding=True, return_tensors="pt")
        outputs = self.text_model(**preprocess_text) 
        return outputs.text_embeds 

    def get_vision_embeddings(self,inputs):
        # Get clip embeddings from CLIP pre-processed images
        outputs = self.visual_model(**inputs) 
        return outputs.image_embeds

    def get_vision_preprocessing(self,inputs): 
        # Get preprocessed images from raw images containing RGB vals between 0-255 
        vision_inputs = self.vision_processor(images=inputs, return_tensors="pt") 
        return vision_inputs 

    def train(self): 
        raise NotImplementedError 

    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError
     
    
class TargetClassTrainer(BaseTrainer):

    def __init__(self, dataloader,test_dataloader,logger,text_model, visual_model, text_tokenizer, 
                 vision_processor,num_iters,lr,target_class='dog'):
        super().__init__(dataloader,test_dataloader,logger,text_model, visual_model, text_tokenizer, 
                 vision_processor,num_iters,lr)
        self.target_class = target_class
        self.train_setup() 

    def train_setup(self):
        self.v = torch.zeros((3,224,224),requires_grad=True)
        self.optimizer = torch.optim.SGD([self.v],lr=self.lr)
        self.loss_criterion = torch.nn.MSELoss() 

        input_caption = f'a photo of a {self.target_class}'
        with torch.no_grad():
            self.target_embeddings = self.get_text_embeddings([input_caption])[0]
        self.curr_iter = 0 

    def train(self): 
        for i in range(self.num_iters):
            # Get a batch of images and captions 
            batch = next(iter(self.dataloader))
            imgs,_ = batch
            # Preprocess images and captions 
            vision_inputs = self.get_vision_preprocessing(imgs) 
            vision_inputs['pixel_values'] += self.v 
            vision_embeddings = self.get_vision_embeddings(vision_inputs)
            loss = self.loss_criterion(vision_embeddings,self.target_embeddings)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.logger.update(iter=self.curr_iter,log_dict={'loss':loss.item()})
            if i % self.log_freq == 0:
                print(f'Iteration {i} loss: {loss.item()}')
                self.evaluate() 
            self.curr_iter +=1

    @torch.no_grad()
    def evaluate(self):
        batch = next(iter(self.test_dataloader))
        imgs,_ = batch
        loss = 0 
        log_dict = {'original_images':[],'generations':[]} 
        for i in imgs:
            # Preprocess images and captions 
            vision_inputs = self.get_vision_preprocessing(i) 
            log_dict['original_images'].append(process_img(vision_inputs['pixel_values']))
            vision_inputs['pixel_values'] += self.v 
            log_dict['generations'].append(process_img(vision_inputs['pixel_values']))
            vision_embeddings = self.get_vision_embeddings(vision_inputs)
            loss += self.loss_criterion(vision_embeddings,self.target_embeddings) 
        log_dict['test_loss'] = loss.item()/len(imgs)
        log_dict['v'] = [process_img(self.v)]
        self.logger.update(iter=self.curr_iter,log_dict=log_dict)