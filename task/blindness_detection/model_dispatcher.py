from models import resnet18
from .config import config

models = {
    "RESNET18": resnet18.resnet_18(
        classes=config.main.N_CLASS,
        pt=config.main.PT),
    "RESNET34" : resnet34.resnet_34(
        classes=config.main.N_CLASS,
        pt=config.main.PT),
    "RESNET50" : resnet50.resnet_50(
        classes=config.main.N_CLASS,
        pt=config.main.PT),
    "RESNEXT50_32X4D" : resnext50_32x4d.resnext_50_32x4d(
        classes=config.main.N_CLASS,
        pt=config.main.PT),
}
