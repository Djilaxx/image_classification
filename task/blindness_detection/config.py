import torch
from easydict import EasyDict as edict

config = edict()

config.main = edict()
config.main.PROJECT_PATH = "task/blindness_detection/"                                              #Main project path
config.main.TRAIN_PATH = "data/Blindness_Detect/train"                                              #Path to training images
config.main.TEST_PATH = "data/Blindness_Detect/test"                                                #Path to test images
config.main.TRAIN_FILE = "data/Blindness_Detect/train.csv"                                          #Path to training data file
config.main.FOLD_FILE = "data/Blindness_Detect/train_folds.csv"                                     #Path to training data with fold
config.main.FOLD_METHOD = "SKF"                                                                     #Folding technique used
config.main.TARGET_VAR = "diagnosis"                                                                #Target variable
config.main.IMAGE_ID = "id_code"                                                                    #Image identifier
config.main.IMAGE_EXT = ".png"                                                                      #Image Extension type
config.main.TRAIN_BS = 32                                                                           #Batch size for training pass
config.main.VALID_BS = 16                                                                           #Batch size for validation pass
config.main.EPOCHS = 5                                                                              #Number of epochs
config.main.LR = 1e-4                                                                               #Learning rate
config.main.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")     #Device to use for training
config.main.N_CLASS = 5                                                                             #Number of class in the target variable
config.main.image_size = (128, 128)                                                                 #Image size
