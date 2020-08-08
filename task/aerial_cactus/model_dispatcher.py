from models import resnet18, resnet34, resnet50, resnext50_32x4d
from .config import config

models = {
    "RESNET18": resnet18.resnet_18(
        classes=config.main.N_CLASS,
        pt=True),
    "RESNET34" : resnet34.resnet_34(
        classes=config.main.N_CLASS,
        pt=True),
    "RESNET50" : resnet50.resnet_50(
        classes=config.main.N_CLASS,
        pt=True),
    "RESNEXT50_32X4D" : resnext50_32x4d.resnext_50_32x4d(
        classes=config.main.N_CLASS,
        pt=True),
}