from models import resnet18
from .config import config

models = {
    "RESNET18": resnet18.resnet_18(
        classes=config.main.N_CLASS,
        pt=True)
}