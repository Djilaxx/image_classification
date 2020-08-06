from models import resnet18
from .config import config

from pytorch_lightning.metrics import functional

models = {
    "RESNET18": resnet18.resnet_18(
        classes=config.main.N_CLASS,
        pt=True)
}

