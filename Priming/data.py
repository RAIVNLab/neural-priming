from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import numpy as np
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        for target in np.sort(os.listdir(self.root)):
            target_path = os.path.join(self.root, target)
            if os.path.isdir(target_path):
                class_idx = len(self.class_to_idx)
                self.class_to_idx[target] = class_idx
                self.idx_to_class[class_idx] = target
                for root, _, fnames in sorted(os.walk(target_path)):
                    if len(fnames) > 0:
                        for fname in fnames:
                            path = os.path.join(root, fname)
                            self.imgs.append((path, class_idx))


    def __getitem__(self, index):
        path, target = self.imgs[index]
        try:
            img = Image.open(path).convert('RGB')
        except:
            print('Error loading image:', path)
            img = Image.new('RGB', (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_class_name(self, class_idx):
        return self.idx_to_class[class_idx]

    def get_class_idx(self, class_name):
        return self.class_to_idx[class_name]

class ImageNetV2(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        # get the folder name from the path
        folder_name = os.path.basename(os.path.dirname(path))
        # convert the folder name to an integer
        target = int(folder_name)
        # return the image and the correct label
        return super().__getitem__(index)[0], target
