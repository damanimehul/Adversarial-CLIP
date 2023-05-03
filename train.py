import torch 
from utils import MINS,MAXES,norm,denorm,process_img
import numpy as np 
from generate_dataset import MSCOCO_CLASSES

class BaseTrainer():

    def __init__(self,device,dataloader,test_dataloader,logger,text_model, visual_model, text_tokenizer, 
                 vision_processor,num_iters,epochs,lr,log_freq):
        self.device = device
        self.text_model = text_model 
        self.visual_model = visual_model 
        self.text_tokenizer = text_tokenizer 
        self.vision_processor = vision_processor
        self.num_iters = num_iters
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.log_freq =  log_freq
        self.logger = logger  
        self.num_iters = num_iters
        self.lr = lr
        self.epochs = epochs
        self.mins = MINS
        self.maxes = MAXES
        if self.device == 'cuda':
            self.mins = self.mins.cuda()
            self.maxes = self.maxes.cuda()

    def train_setup(self):
        self.v = torch.zeros((3,224,224),requires_grad=True)
        self.optimizer = torch.optim.SGD([self.v],lr=self.lr)
        if self.device == 'cuda':
            self.v = self.v.cuda()
        self.loss_criterion = torch.nn.MSELoss() 
        all_captions = [i for i in MSCOCO_CLASSES]
        with torch.no_grad():
            self.all_embeddings = self.get_text_embeddings(all_captions)
        self.curr_iter = 0 
        self.curr_epoch = 0

    def get_text_embeddings(self,inputs):
        # Get clip embeddings from a list of str inputs 
        preprocess_text = self.text_tokenizer(inputs, padding=True, return_tensors="pt")
        if self.device == 'cuda':
            preprocess_text = {k:v.cuda() for k,v in preprocess_text.items()}
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
    
    def get_image(self,inputs): 
        if self.device == 'cuda':
            img_proc = inputs.clone().cpu()
            img_proc = process_img(img_proc)
        else :
            img_proc = process_img(inputs)
        return img_proc
    
    @torch.no_grad()
    def get_classification_accuracy(self,vision_embeddings,captions):
        dot_products = torch.transpose(torch.matmul(self.all_embeddings,torch.transpose(vision_embeddings,0,1)),0,1) 
        maxes = torch.argmax(dot_products,dim=1)
        maxes = maxes.cpu().numpy()
        caption_labels = np.array([MSCOCO_CLASSES.index(i) for i in captions]) 
        flags = np.equal(maxes,caption_labels) 
        accuracy = np.sum(flags)/len(flags) 
        return accuracy

     
class TargetClassTrainer(BaseTrainer):

    def __init__(self, device,dataloader,test_dataloader,logger,text_model, visual_model, text_tokenizer, 
                 vision_processor,num_iters,epochs,lr,log_freq,target_class='dog'):
        super().__init__(device,dataloader,test_dataloader,logger,text_model, visual_model, text_tokenizer, 
                 vision_processor,num_iters,epochs,lr,log_freq)
        self.target_class = target_class
        self.target_class_index = MSCOCO_CLASSES.index(target_class)
        self.train_setup() 

    def train_setup(self):
        super().train_setup()
        input_caption = self.target_class
        with torch.no_grad():
            self.target_embeddings = self.get_text_embeddings([input_caption])[0]

    @torch.no_grad()
    def get_classification_accuracy(self,vision_embeddings,captions):
        dot_products = torch.transpose(torch.matmul(self.all_embeddings,torch.transpose(vision_embeddings,0,1)),0,1) 
        maxes = torch.argmax(dot_products,dim=1)
        maxes = maxes.cpu().numpy()
        caption_labels = np.array([MSCOCO_CLASSES.index(i) for i in captions]) 
        flags = np.equal(maxes,caption_labels) 
        accuracy = np.sum(flags)/len(flags)
        #Count number of appearences of target class in batch
        target_class_count = np.sum(np.equal(maxes,self.target_class_index))
        # Average distance to target class embedding 
        target_dists = dot_products[:,self.target_class_index].cpu().numpy()
        return accuracy, target_class_count, np.mean(target_dists)

    def train(self): 
        perturbs = {} 
        for i in range(self.epochs): 
            losses,train_accuracies,target_accuracies,distance_to_targets = [],[],[],[] 
            for j in range(self.num_iters):
                # Get a batch of images and captions 
                batch = next(iter(self.dataloader))
                imgs,captions = batch
                if self.device == 'cuda':
                    imgs = imgs.cuda()
                # Preprocess images and captions 
                vision_inputs = self.get_vision_preprocessing(imgs) 
                if self.device == 'cuda':
                    vision_inputs = {k:v.cuda() for k,v in vision_inputs.items()}
                vision_inputs['pixel_values'] += self.v 
                vision_inputs['pixel_values'] = torch.stack([torch.clamp(vision_inputs['pixel_values'][:, k, :, :], self.mins[k], self.maxes[k]) for k in range(3)], dim=1)
                vision_embeddings = self.get_vision_embeddings(vision_inputs)
                train_accuracy,target_accuracy,distance_to_target = self.get_classification_accuracy(vision_embeddings,captions)
                target_embeddings = self.target_embeddings.repeat(vision_embeddings.shape[0],1)
                loss = self.loss_criterion(vision_embeddings,target_embeddings)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
                train_accuracies.append(train_accuracy)
                target_accuracies.append(target_accuracy)
                distance_to_targets.append(distance_to_target)
                self.curr_iter +=1
            self.logger.update(iter=self.curr_epoch,log_dict={'loss':np.mean(losses),'train classification accuracy':np.mean(train_accuracies), 
                                                              'Classified as target':np.mean(target_accuracies), 'Average similarity with target':np.mean(distance_to_targets)})
            if i % self.log_freq == 0:
                test_accuracy = self.evaluate() 
                print(f'Epoch {i} loss: {loss.item()} train classification accuracy: {100*np.round(np.mean(train_accuracies),2)}% test classification accuracy: {100*np.round(test_accuracy,2)}%')
                v = self.v.clone().detach().cpu().numpy()
                perturbs[self.curr_epoch] = v
                self.logger.update(iter=self.curr_epoch,log_dict={'norm v':np.linalg.norm(v.flatten()),'test classification accuracy':test_accuracy})
            self.curr_epoch +=1

        return perturbs

    @torch.no_grad()
    def evaluate(self):
        batch = next(iter(self.test_dataloader))
        imgs,captions = batch
        if self.device == 'cuda':
            imgs = imgs.cuda()
        loss = 0 
        log_dict = {'original_images':[],'generations':[]} 
        class_accuracies, target_classified, target_dist = [] , [], [] 
        for j,i in enumerate(imgs):
            caption = captions[j]
            # Preprocess images and captions 
            vision_inputs = self.get_vision_preprocessing(i) 
            if self.device == 'cuda':
                vision_inputs = {k:v.cuda() for k,v in vision_inputs.items()}
            if j<8:
                log_dict['original_images'].append(self.get_image(vision_inputs['pixel_values'].clone().cpu())) 
            vision_inputs['pixel_values'] += self.v 
            vision_inputs['pixel_values'] = torch.stack([torch.clamp(vision_inputs['pixel_values'][:, i, :, :], self.mins[i], self.maxes[i]) for i in range(3)], dim=1)
            if j<8:
                log_dict['generations'].append(self.get_image(vision_inputs['pixel_values']))
            vision_embeddings = self.get_vision_embeddings(vision_inputs)
            target_embeddings = self.target_embeddings.repeat(vision_embeddings.shape[0],1)
            loss += self.loss_criterion(vision_embeddings,target_embeddings) 
            a,b,c = self.get_classification_accuracy(vision_embeddings,[caption]) 
            class_accuracies.append(a)
            target_classified.append(b)
            target_dist.append(c)

        class_accuracies = np.mean(class_accuracies)
        log_dict['test_loss'] = loss.item()/len(imgs)
        log_dict['v'] = [self.get_image(self.v.clone().cpu())] 
        log_dict['test classification accuracy'] = class_accuracies
        log_dict['Test-Classified as target'] = np.mean(target_classified)
        log_dict['Test-Average similarity with target'] = np.mean(target_dist)
        self.logger.update(iter=self.curr_epoch,log_dict=log_dict) 
        return class_accuracies