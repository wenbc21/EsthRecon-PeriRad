from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, is_train: bool, mean, std):
        self.images_path = images_path
        self.images_class = images_class
        self.mean = mean
        self.std = std

        if is_train :
            self.transform = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0, 0.25), contrast=(0.25, 0.5)),
                transforms.RandomInvert(p=0.2),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else :
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

    # 定义变换
    def augment_and_pad(self, image, target_size):
        # # 获取增强后的图像尺寸
        width, height = image.size

        # 计算长边缩放到目标大小后的新尺寸
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        # 定义调整大小和填充的变换
        transform = transforms.Compose([
            transforms.Resize((new_height, new_width)),
        ])
        
        image = transform(image)
        
        # 创建增强变换
        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(1.25, 1.75)),
            transforms.RandomRotation(20)
        ])
        
        # 应用增强变换
        augmented_image = augment_transform(image)
        
        # 获取增强后的图像尺寸
        width, height = augmented_image.size

        if width > height:
            # 图像较宽
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            # 图像较高
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        pad_height1 = (target_size - new_width) // 2
        pad_height2 = target_size - new_width - pad_height1
        pad_width1 = (target_size - new_height) // 2
        pad_width2 = target_size - new_height - pad_width1

        # 创建变换
        transform = transforms.Compose([
            # transforms.Resize((new_height, new_width)),  # 首先调整大小
            transforms.Pad((pad_height1, pad_width1, pad_height2, pad_width2), fill=(0)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # 应用调整大小和填充变换
        padded_image = transform(augmented_image)
        
        return padded_image

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('L')
        
        # transformed_image = self.augment_and_pad(img, 224)

        # # 查看调整后的尺寸
        # print("Transformed size:", transformed_image.size)

        # # 显示原始和变换后的图像
        # plt.figure(figsize=(8, 4))

        # # 显示原始图像
        # plt.subplot(1, 2, 1)
        # plt.imshow(img)
        # plt.title('Original Image')
        # plt.axis('off')

        # # 显示变换后的图像
        # plt.subplot(1, 2, 2)
        # plt.imshow(transformed_image)
        # plt.title('Transformed Image (256x256)')
        # plt.axis('off')

        # plt.show()

        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        # if self.transform is not None:
        #     img = self.transform(img)
        img = self.augment_and_pad(img, 224)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
