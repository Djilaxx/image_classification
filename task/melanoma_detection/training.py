import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc
import pandas as pd
import numpy as np
from pathlib import Path

#ML import
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn import model_selection

#My own modules
from image_dataset import image_ds
from utils import early_stopping, folding, parser
from trainer.train_fct import Trainer
from .config import config
from .model_dispatcher import models
from .augment import Augmentations

def run(folds, model):

    #Creating the folds from the training data
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)

    #Set parameters
    train_path = config.main.TRAIN_PATH
    df = pd.read_csv(config.main.FOLD_FILE)
    device = config.main.device
    epochs = config.main.EPOCHS
    train_bs = config.main.TRAIN_BS
    valid_bs = config.main.VALID_BS
    image_size = config.main.image_size

    #Initial parameters save
    model = models[model]
    Path(os.path.join(config.main.PROJECT_PATH, "model_init/")).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.main.PROJECT_PATH, "model_init/model_init.pt"))

    for fold in range(folds):
        print(f"Starting training for fold : {fold}")

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        #Load initial parameters of the model and load it on device
        init = torch.load(os.path.join(config.main.PROJECT_PATH, "model_init/model_init.pt"))
        model.load_state_dict(init)
        model.to(device)

        #Create dataset and dataloader
        trn_img = df_train[config.main.IMAGE_ID].values.tolist()
        trn_img = [os.path.join(train_path, i + config.main.IMAGE_EXT) for i in trn_img]
        trn_labels = df_train[config.main.TARGET_VAR].values

        valid_img = df_valid[config.main.IMAGE_ID].values.tolist()
        valid_img = [os.path.join(train_path, i + config.main.IMAGE_EXT) for i in valid_img]
        valid_labels = df_valid[config.main.TARGET_VAR].values

        trn_ds = image_ds.Image_dataset(
            image_path=trn_img,
            resize=image_size,
            label=trn_labels,
            transforms=Augmentations["train"]
        )

        train_loader = torch.utils.data.DataLoader(
            trn_ds, batch_size=train_bs, shuffle=True, num_workers=4
        )

        valid_ds = image_ds.Image_dataset(
            image_path=valid_img,
            resize=image_size,
            label=valid_labels,
            transforms=Augmentations["valid"]
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=valid_bs, shuffle=True, num_workers=2
        )

        #Set optimizer, scheduler, early stopping etc...
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.main.LR)
        scheduler = None
        es = early_stopping.EarlyStopping(patience=2, mode="max")
        trainer = Trainer(
            model, optimizer, device, criterion, scheduler=scheduler)
        #Starting training for nb_epoch
        for epoch in range(epochs):
            print(f"Starting epoch number : {epoch}")
            #Training phase
            print("Training the model...")
            trainer.trn_function(train_loader)
            #Evaluation phase
            print("Evaluating the model...")
            acc = trainer.eval_function(valid_loader)
            #Metrics
            print(f"Validation Accuracy = {acc}")

            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(acc, model,
               model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{fold}.bin"))
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

#Create the parser
args = parser.create_parser()

if __name__ == "__main__":
    print("Training start...")

    run(
        folds=args.folds,
        model=args.model,
        metric=args.metric
    )