##################
# IMPORT MODULES #
##################
# Sys import
import os, inspect, importlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import gc
import pandas as pd
import numpy as np
from pathlib import Path
# ML import
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn import model_selection
# My own modules
from image_dataset import image_ds
from utils import early_stopping, folding
from trainer.train_fct import Trainer
################
# RUN FUNCTION #
################
def run(folds=5, task="aerial_cactus", model="resnet_18", loss="cross_entropy", metric="ACCURACY"):

    print(f"Training on task : {task} for {folds} folds with {model} model")
    print(f"{loss} loss & {metric} metric")
    # IMPORT TASK FILES
    config = getattr(importlib.import_module(f"task.{task}.config"), "config")
    Augmentations = getattr(importlib.import_module(f"task.{task}.augment"), "Augmentations") 
    # CREATING FOLDS
    folding.create_folds(datapath=config.main.TRAIN_FILE,
                        output_path=config.main.FOLD_FILE,
                        nb_folds = folds,
                        method=config.main.FOLD_METHOD,
                        target=config.main.TARGET_VAR)
    # LOADING DATA FILE
    df = pd.read_csv(config.main.FOLD_FILE)
    # SAVE MODEL PARAM (USEFUL FOR PRETRAINED MODELS) 
    for name, cls in inspect.getmembers(importlib.import_module("models." + model), inspect.isclass):
        if name == model:
            model = cls(classes = config.main.N_CLASS, pt = config.hyper.PT)

    Path(os.path.join(config.main.PROJECT_PATH, "model_init/")).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.main.PROJECT_PATH, "model_init/model_init.pt"))

    # START FOLD LOOP
    for fold in range(folds):
        print(f"Starting training for fold : {fold}")
        # CREATING TRAINING AND VALIDATION SETS
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        # LOADING MODEL PARAMETERS
        init = torch.load(os.path.join(config.main.PROJECT_PATH, "model_init/model_init.pt"))
        model.load_state_dict(init)
        model.to(config.main.DEVICE)
        ########################
        # CREATING DATALOADERS #
        ########################
        # TRAINING IDs & LABELS
        trn_img = df_train[config.main.IMAGE_ID].values.tolist()
        trn_img = [os.path.join(config.main.TRAIN_PATH, os.path.splitext(i)[0] + config.main.IMAGE_EXT) for i in trn_img]
        trn_labels = df_train[config.main.TARGET_VAR].values
        # VALIDATION IDs & LABELS
        valid_img = df_valid[config.main.IMAGE_ID].values.tolist()
        valid_img = [os.path.join(config.main.TRAIN_PATH, os.path.splitext(i)[0] + config.main.IMAGE_EXT) for i in valid_img]
        valid_labels = df_valid[config.main.TARGET_VAR].values
        # TRAINING DATASET
        trn_ds = image_ds.Image_dataset(
            image_path=trn_img,
            resize=config.hyper.IMAGE_SIZE,
            label=trn_labels,
            transforms=Augmentations["train"]
        )
        # TRAINING DATALOADER
        train_loader = torch.utils.data.DataLoader(
            trn_ds, batch_size=config.hyper.TRAIN_BS, shuffle=True, num_workers=0
        )
        # VALIDATION DATASET
        valid_ds = image_ds.Image_dataset(
            image_path=valid_img,
            resize=config.hyper.IMAGE_SIZE,
            label=valid_labels,
            transforms=Augmentations["valid"]
        )
        # VALIDATION DATALOADER
        valid_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=config.hyper.VALID_BS, shuffle=True, num_workers=0
        )

        # IMPORT LOSS FUNCTION
        loss_module = importlib.import_module(f"loss.{loss}")
        criterion = loss_module.loss_fct()

        # SET OPTIMIZER, SCHEDULER
        optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # SET EARLY STOPPING FUNCTION
        es = early_stopping.EarlyStopping(patience=2, mode="max")

        # CREATE TRAINER
        trainer = Trainer(model, optimizer, config.main.DEVICE, criterion)

        # START TRAINING FOR N EPOCHS
        for epoch in range(config.hyper.EPOCHS):
            print(f"Starting epoch number : {epoch}")

            # TRAINING PHASE
            print("Training the model...")
            trainer.training_step(train_loader)

            # VALIDATION PHASE
            print("Evaluating the model...")
            val_loss, metric_value = trainer.eval_step(valid_loader, metric)
            scheduler.step(val_loss)

            # METRICS
            print(f"Validation {metric} = {metric_value}")

            #SAVING CHECKPOINTS
            Path(os.path.join(config.main.PROJECT_PATH, "model_output/")).mkdir(parents=True, exist_ok=True)
            es(metric_value, model,
               model_path=os.path.join(config.main.PROJECT_PATH, "model_output/", f"model_{fold}.bin"))
            if es.early_stop:
                print("Early Stopping")
                break
            gc.collect()

##########
# PARSER #
##########
parser = argparse.ArgumentParser()
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--task", type=str, default="aerial_cactus")
parser.add_argument("--model", type=str, default="resnet_18")
parser.add_argument("--loss", type=str, default="cross_entropy")
parser.add_argument("--metric", type=str, default="ACCURACY")

args = parser.parse_args()
##################
# START TRAINING #
##################
if __name__ == "__main__":
    print("Training start...")
    run(
        folds=args.folds,
        task=args.task,
        model=args.model,
        loss=args.loss,
        metric=args.metric
    )