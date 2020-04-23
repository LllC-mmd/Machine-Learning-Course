from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import PIL.ImageOps as ops
import numpy as np


def invertIMG(img, p):
    u = np.random.random(size=1)
    if u < p:
        return ops.invert(img)
    else:
        return img

## Note that: here we provide a basic solution for loading data and transforming data.
## You can directly change it if you find something wrong or not good enough.

## the mean and standard variance of imagenet dataset
## mean_vals = [0.485, 0.456, 0.406]
## std_vals = [0.229, 0.224, 0.225]

def load_data(data_dir = "../data/", input_size=224, batch_size=36):
    data_transforms = {
        'train': transforms.Compose([
            # A crop of random size (default: of 0.08 to 1.0) of the original size
            # and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made
            transforms.RandomResizedCrop(input_size),
            # Horizontally flip the given PIL Image randomly with a given probability (default p=0.5)
            transforms.RandomHorizontalFlip(),
            # Add-on augmentation
            transforms.Lambda(lambda img: invertIMG(img, p=0.5)),
            transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0.2, hue=0.2),
            # Convert a PIL Image or numpy.ndarray to tensor
            transforms.ToTensor(),
            # Normalize a tensor image with mean and standard deviation
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_dataset_train = datasets.ImageFolder(os.path.join(data_dir,'train'), data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(data_dir,'valid'), data_transforms['valid'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader
