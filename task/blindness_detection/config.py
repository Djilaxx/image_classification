import torch
from easydict import EasyDict as edict

config = edict()

config.main = edict()
config.main.PROJECT_PATH = "task/blindness_detection/"
config.main.TRAIN_PATH = "data/Blindness_Detect/train"
config.main.TEST_PATH = "data/Blindness_Detect/test"
config.main.TRAIN_FILE = "data/Blindness_Detect/train.csv"
config.main.FOLD_FILE = "data/Blindness_Detect/train_folds.csv"
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "diagnosis"
config.main.IMAGE_ID = "id_code"
config.main.IMAGE_EXT = ".png"
config.main.FOLD = 5
config.main.TRAIN_BS = 32
config.main.VALID_BS = 16
config.main.EPOCHS = 5
config.main.LR = 1e-4
config.main.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 5
config.main.image_size = (128, 128)

config.RESNET18 = edict()
