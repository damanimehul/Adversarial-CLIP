import pickle
from torch.utils.data import Dataset

class ImageCaptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, caption = self.data[index]
        return image, caption
    

