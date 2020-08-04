import torch
from easydict import EasyDict as edict

config = edict()

config.main = edict()
config.main.PROJECT_PATH = "task/melanoma_detection/"
config.main.TRAIN_PATH = "data/melanoma/train/train/"
config.main.TEST_PATH = "data/melanoma/test/test/"
config.main.TRAIN_FILE = "data/melanoma/train_concat.csv"
config.main.FOLD_FILE = "data/melanoma/train_folds.csv"
config.main.FOLD_METHOD = "SKF"
config.main.TARGET_VAR = "target"
config.main.IMAGE_ID = "image_name"
config.main.IMAGE_EXT = ".jpg"
config.main.FOLD = 5
config.main.TRAIN_BS = 16
config.main.VALID_BS = 8
config.main.EPOCHS = 5
config.main.LR = 1e-4
config.main.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 2
config.main.image_size = (128, 128)

config.RESNET18 = edict()