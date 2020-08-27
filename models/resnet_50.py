import torch
import torch.nn as nn
from torchvision import models

class resnet_50(nn.Module):
    def __init__(self, classes=2, pt=True):
        super(resnet_50, self).__init__()

        self.base_model = models.resnet50(pretrained=pt)
        in_features = self.base_model.fc.out_features
        #self.nb_features = self.base_model.fc.in_features
        self.l0 = nn.Linear(in_features, classes)
    
    def forward(self, image):
        x = self.base_model(image)
        out = self.l0(x)
        return out