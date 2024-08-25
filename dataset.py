from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class SACDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, is_train: bool, args, mean, std):
        self.images_path = images_path
        self.images_class = images_class

        if is_train :
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else :
            self.transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
