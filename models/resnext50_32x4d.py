import torch
import torch.nn as nn
from torchvision import models

class resnext_50_32x4d(nn.Module):
    def __init__(self, classes, pt=True):
        super(resnext_50_32x4d, self).__init__()

        self.base_model = models.resnext50_32x4d(pretrained=pt)
        in_features = self.base_model.fc.out_features
        #self.nb_features = self.base_model.fc.in_features
        self.l0 = nn.Linear(in_features, classes)
    
    def forward(self, image):
        x = self.base_model(image)
        out = self.l0(x)
        return out