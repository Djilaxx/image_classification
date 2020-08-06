from models import resnet18
from .config import config

from pytorch_lightning.metrics import functional

models = {
    "RESNET18": resnet18.resnet_18(
        classes=config.main.N_CLASS,
        pt=True)
}

metrics = {
    "ACCURACY" : functional.accuracy,
    "AUROC" : functional.auroc,
    "CONFUSION_MATRIX" : functional.confusion_matrix,
    "F1_SCORE" : functional.f1_score
}