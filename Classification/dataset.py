from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from utils import augment_and_pad

class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, is_train: bool, mean, std):
        self.images_path = images_path
        self.images_class = images_class
        self.mean = mean
        self.std = std
        self.channels = len(mean)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if self.channels == 1 :
            img = img.convert('L')
        
        img = augment_and_pad(img, 224, self.mean, self.std, self.channels)
        label = self.images_class[item]

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
