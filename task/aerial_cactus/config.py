import torch
from easydict import EasyDict as edict

config = edict()

config.main = edict()
config.main.PROJECT_PATH = "task/aerial_cactus/"                                                    #Main project path
config.main.TRAIN_PATH = "data/aerial-cactus-identification/train"                                  #Path to training images
config.main.TEST_PATH = "data/aerial-cactus-identification/test"                                    #Path to test images
config.main.TRAIN_FILE = "data/aerial-cactus-identification/train.csv"                              #Path to training data file
config.main.FOLD_FILE = "data/aerial-cactus-identification/train_folds.csv"                         #Path to training data with fold
config.main.FOLD_METHOD = "SKF"                                                                     #Folding technique used
config.main.TARGET_VAR = "has_cactus"                                                               #Target variable
config.main.IMAGE_ID = "id"                                                                         #Image identifier
config.main.IMAGE_EXT = ".jpg"                                                                      #Image Extension type
config.main.TRAIN_BS = 32                                                                           #Batch size for training pass
config.main.VALID_BS = 16                                                                           #Batch size for validation pass
config.main.EPOCHS = 5                                                                              #Number of epochs
config.main.LR = 1e-4                                                                               #Learning rate
config.main.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")     #Device to use for training
config.main.N_CLASS = 2                                                                             #Number of class in the target variable
config.main.image_size = (32, 32)                                                                   #Image size
config.main.PT = False