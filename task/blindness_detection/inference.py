import os
import pandas as pd 
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm

from utils import parser
from image_dataset import image_ds
from .config import config
from .augment import Augmentations
from .model_dispatcher import models

def inference(model = "RESNET18", model_path = None, test_file = True):
    final_output = []
    device = config.main.device

    if test_file is True:
        df_test = pd.read_csv(config.main.TEST_FILE)
        test_img = df_test[config.main.IMAGE_ID].values.tolist()
        test_img = [os.path.join(config.main.TEST_PATH, i + config.main.IMAGE_EXT) for i in test_img]
    else:
        files_in_test = sorted(os.listdir(config.main.TEST_PATH))
        df_test = pd.DataFrame()
        df_test[config.main.IMAGE_ID] = [str(x) for x in files_in_test]
        test_img = df_test[config.main.IMAGE_ID].values.tolist()
        test_img = [os.path.join(config.main.TEST_PATH, i + config.main.IMAGE_EXT) for i in test_img]
        
    test_ds = image_ds.Image_dataset(
        image_path=test_img,
        resize=config.main.image_size,
        transforms=Augmentations["test"],
        test=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.main.VALID_BS, shuffle=True, num_workers=6
    )

        #Code that allow ensemble of multiple model (same model arch) or inference using a specified model path
    model_dict = dict()
    if model_path is not None:
        model = models[model]
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        for obj, idx in zip(os.listdir(os.path.join(config.main.PROJECT_PATH, "model_output/")), range(0, len(os.listdir(os.path.join(config.main.PROJECT_PATH, "model_output/"))))):
            model_dict[idx] = models[model]
            model_dict[idx].to(device)
            model_dict[idx].load_state_dict(torch.load(os.path.join(config.main.PROJECT_PATH, "model_output/", obj)))
            model_dict[idx].eval()

    with torch.no_grad():
        tk0 = tqdm(test_loader, total=len(test_loader))

        for _, data in enumerate(tk0):
            images = data["images"]
            images = images.to(device)

            for model in model_dict:
                model_used = model_dict[model]
                output = model_used(images)
                final_output.append(output)
                
            final_predictions = torch.sum(final_output)/len(model_dict)
            final_predictions = torch.softmax(final_predictions, dim=1)
            _, final_predictions = torch.max(final_predictions, dim=1)

    Path(os.path.join(config.main.PROJECT_PATH, "submission/")).mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(config.main.SUBMISSION_FILE)
    sample.loc[: config.main.TARGET_VAR] = final_predictions
    sample.to_csv(os.path.join(config.main.PROJECT_PATH, "submission/submission.csv"), index = False)

#Create the parser
args = parser.create_parser()

if __name__ == "__main__":
    print("Inference start...")

    inference(
        model=args.model,
        model_path=args.model_path,
        test_file=args.test_file
    )


    


#Create the test dataset and dataloader 

#Create the model, and add the trained model parameters

#Torch loop of prediction for each loaded model

#Optional ensembling if we loaded multiple models

#Put the prediction into correct format and create submission file (if it's kaggle competition)

#Return a metric if we have the values of the test set

#Parsing argument