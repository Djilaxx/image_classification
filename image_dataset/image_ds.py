#Data functions, allow to load the data and targets, transform into a pytorch dataset
import numpy as np
import torch

from PIL import Image
from PIL import ImageFile

class Image_dataset:
    '''
    Pytorch class to define an image dataset
    image_path : must be a list of path to individual images like "data/image_001.png"
    resize : if not None, image will be resized to this size, MUST BE A TUPLE
    label : labels for each image of their class
    transforms : if not None, transform will be applied on images
    '''

    def __init__(self, image_path, resize, label=None, transforms=None, test=False):
        self.image_path = image_path
        self.resize = resize
        self.label = label
        self.transforms = transforms
        self.test = test

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        if self.transforms:
            image = self.transforms(image)

        if self.test is False:
            label = self.label[item]
            return {
                "images": image,
                "labels": torch.tensor(label)
            }
        else:
            return {
                "images" : image
            }

    def __len__(self):
        return len(self.image_path)


class Image_loader:
    def __init__(self, image_path, resize, label=None, transforms=None, test=False):
        self.image_path = image_path
        self.resize = resize
        self.label = label,
        self.transforms = transforms,
        self.test = test
        self.image_dataset = Image_dataset(
            image_path = self.image_path,
            resize = self.resize,
            label = self.label,
            transforms = self.transforms,
            test = self.test
        )
    
    def get_loader(self, batch_size, num_workers, shuffle=True, drop_last=False):

        image_data_loader = torch.utils.data.DataLoader(
            self.image_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        return image_data_loader