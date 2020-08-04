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

    def __init__(self, image_path, resize, label, transforms=None, test=False):
        self.image_path = image_path
        self.resize = resize
        self.label = label
        self.transforms = transforms

    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        label = self.label[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        if self.transforms:
            image = self.transforms(image)

        return {
            "images": image,
            "labels": torch.tensor(label)
        }

    def __len__(self):
        return len(self.image_path)
