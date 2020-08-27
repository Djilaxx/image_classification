import torch
from easydict import EasyDict as edict

config = edict()

########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
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
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
config.main.N_CLASS = 2

###################
# HYPERPARAMETERS #
###################
config.hyper = edict()
config.hyper.TRAIN_BS = 16
config.hyper.VALID_BS = 8
config.hyper.EPOCHS = 5
config.hyper.LR = 1e-4
config.hyper.IMAGE_SIZE = (128, 128)
config.hyper.PT = False