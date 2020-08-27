import os, inspect, importlib
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
from .augment import Augmentations

def run(folds=5, model="resnet_18", metric="ACCURACY"):

    print(f"Training for {folds} folds with {model} model")
    
    #Creating the folds from the training data
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)

    #Load data
    df = pd.read_csv(config.main.FOLD_FILE)
    
    #Initial parameters save
    #Getting the model class and instantiate the model with config parameters
    for name, cls in inspect.getmembers(importlib.import_module("models." + model), inspect.isclass):
        if name == model:
            model = cls(
                classes = config.main.N_CLASS, 
                pt = config.hyper.PT
                )

    Path(os.path.join(config.main.PROJECT_PATH, "model_init/")).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.main.PROJECT_PATH, "model_init/model_init.pt"))

    for fold in range(folds):
        print(f"Starting training for fold : {fold}")

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        #Load initial parameters of the model and load it on device
        init = torch.load(os.path.join(config.main.PROJECT_PATH, "model_init/model_init.pt"))
        model.load_state_dict(init)
        model.to(config.main.DEVICE)

        #Create dataset and dataloader
        trn_img = df_train[config.main.IMAGE_ID].values.tolist()
        trn_img = [os.path.join(config.main.TRAIN_PATH, i) for i in trn_img]
        trn_labels = df_train[config.main.TARGET_VAR].values

        valid_img = df_valid[config.main.IMAGE_ID].values.tolist()
        valid_img = [os.path.join(config.main.TRAIN_PATH, i) for i in valid_img]
        valid_labels = df_valid[config.main.TARGET_VAR].values

        trn_ds = image_ds.Image_dataset(
            image_path=trn_img,
            resize=config.hyper.IMAGE_SIZE,
            label=trn_labels,
            transforms=Augmentations["train"]
        )

        train_loader = torch.utils.data.DataLoader(
            trn_ds, batch_size=config.hyper.TRAIN_BS, shuffle=True, num_workers=4
        )

        valid_ds = image_ds.Image_dataset(
            image_path=valid_img,
            resize=config.hyper.IMAGE_SIZE,
            label=valid_labels,
            transforms=Augmentations["valid"]
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=config.hyper.VALID_BS, shuffle=True, num_workers=2
        )

        #Set optimizer, scheduler, early stopping etc...
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper.LR)
        scheduler = None
        es = early_stopping.EarlyStopping(patience=2, mode="max")
        trainer = Trainer(
            model, optimizer, config.main.DEVICE, criterion, scheduler=scheduler)
        #Starting training for nb_epoch
        for epoch in range(config.hyper.EPOCHS):
            print(f"Starting epoch number : {epoch}")
            #Training phase
            print("Training the model...")
            trainer.training_step(train_loader)
            #Evaluation phase
            print("Evaluating the model...")
            metric_value = trainer.eval_step(valid_loader, metric)
            #Metrics
            print(f"Validation {metric} = {metric_value}")

            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(metric_value, model,
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