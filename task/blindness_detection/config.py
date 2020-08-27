import torch
from easydict import EasyDict as edict

config = edict()

########
# MAIN #
########
# main is the config section related to basic info on the project
# data repo, data format, folding etc... data preparation
config.main = edict()
config.main.PROJECT_PATH = "task/blindness_detection/"                                              #Main project path
config.main.TRAIN_PATH = "data/Blindness_Detect/train"                                              #Path to training images
config.main.TEST_PATH = "data/Blindness_Detect/test"                                                #Path to test images
config.main.TRAIN_FILE = "data/Blindness_Detect/train.csv"                                          #Path to training data file
config.main.FOLD_FILE = "data/Blindness_Detect/train_folds.csv"                                     #Path to training data with fold
config.main.TEST_FILE = "data/Blindness_Detect/test.csv"
config.main.SUBMISSION_FILE = "data/Blindness_Detect/sample_submission.csv"
config.main.FOLD_METHOD = "SKF"                                                                     #Folding technique used
config.main.TARGET_VAR = "diagnosis"                                                                #Target variable
config.main.IMAGE_ID = "id_code"                                                                    #Image identifier
config.main.IMAGE_EXT = ".png"                                                                      #Image Extension type                                                                        #Learning rate
config.main.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")     #Device to use for training
config.main.N_CLASS = 5                                                                             #Number of class in the target variable


###################
# HYPERPARAMETERS #
###################
config.hyper = edict()
config.hyper.TRAIN_BS = 32                                                                           #Batch size for training pass
config.hyper.VALID_BS = 16                                                                           #Batch size for validation pass
config.hyper.EPOCHS = 5                                                                              #Number of epochs
config.hyper.LR = 1e-4  
config.hyper.IMAGE_SIZE = (128, 128)                                                                 #Image size
config.hyper.PT = False