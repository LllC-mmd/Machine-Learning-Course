import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


pretrained_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224, interpolation=2),
        # Horizontally flip the given PIL Image randomly with a given probability (default p=0.5)
        transforms.RandomHorizontalFlip(),
        # Add-on augmentation
        transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0.2, hue=0.2),
        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(224, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

seg_data_transforms = {
    'train': transforms.Compose([
        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),
        # Normalize a tensor image with mean and standard deviation
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class SegDataset(Dataset):
    def __init__(self, args, root="SegDataset", mode="train"):

        self.root = root
        self.mode = mode
        self.args = args
        self.files = {}

        self.img_base = os.path.join(self.root, self.mode, "img_RGB")
        self.labels_base = os.path.join(self.root, self.mode, "label_5classes")

        self.files[mode] = self.recursive_glob(rootdir=self.img_base, suffix='.tiff')

        self.label_dict = {"built-up": (0, [255, 0, 0]), "farmland": (1, [0, 255, 0]), "forest": (2, [0, 255, 255]),
                           "meadow": (3, [255, 255, 0]), "water": (4, [0, 0, 255]), "unknown": (5, [0, 0, 0])}

        if not self.files[mode]:
            raise Exception("No files for split=[%s] found in %s" % (mode, self.img_base))

        print("Found %d %s images" % (len(self.files[mode]), mode))

    def __len__(self):
        return len(self.files[self.mode])

    def __getitem__(self, index):
        img_name = self.files[self.mode][index].split(".")[0]
        img_path = os.path.join(self.img_base, img_name + '.tiff')
        lbl_path = os.path.join(self.labels_base, img_name + '.tiff')

        _img = Image.open(img_path).convert('RGB')
        _tmp = Image.open(lbl_path).convert('RGB')
        _target = self.encode_segmap(_tmp, self.label_dict)

        if self.mode == 'train':
            _img = seg_data_transforms["train"](_img)
        elif self.mode == 'valid':
            _img = seg_data_transforms["valid"](_img)
        else:
            raise NotImplementedError

        sample = {'image': _img, 'label': _target}
        return sample

    def encode_segmap(self, label_rgb, label_dict):
        label_arr = np.array(label_rgb)
        mask = np.zeros((label_arr.shape[0], label_arr.shape[1]))
        for k, v in label_dict.items():
            coor = np.where(np.all(label_arr == v[1], axis=2))
            mask[coor] = v[0]
        return mask

    def recursive_glob(self, rootdir='.', suffix=".tiff"):
        return [filename for looproot, _, filenames in os.walk(rootdir) for filename in filenames if filename.endswith(suffix)]


def load_pretrained_data(root_dir="./dataset", batch_size=12):
    image_dataset_train = datasets.ImageFolder(os.path.join(root_dir, 'train'), pretrained_data_transforms['train'])
    image_dataset_valid = datasets.ImageFolder(os.path.join(root_dir, 'valid'), pretrained_data_transforms['valid'])

    train_loader = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(image_dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader


def load_seg_data(args, root_dir="SegDataset"):

    seg_dataset_train = SegDataset(args, root=root_dir, mode="train")
    seg_dataset_valid = SegDataset(args, root=root_dir, mode="valid")

    train_loader = DataLoader(seg_dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(seg_dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader