from utils.average_meter import AverageMeter
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from pytorch_lightning.metrics.functional import accuracy
from .model_dispatcher import metrics
class Trainer:
    '''
    trn_function train the model for one epoch
    eval_function evaluate the current model on validation data and output current loss and other evaluation metric
    '''
    def __init__(self, model, optimizer, device, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.criterion = criterion

    def trn_function(self, data_loader):
        self.model.train()
        losses = AverageMeter()

        tk0 = tqdm(data_loader, total=len(data_loader))

        for _, data in enumerate(tk0):
            images = data["images"]
            labels = data["labels"]

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.model.zero_grad()

            output = self.model(images)
            loss = self.criterion(output, labels)
            
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            losses.update(loss.item(), images.size(0))
            tk0.set_postfix(loss=losses.avg)

    def eval_function(self, data_loader):
        self.model.eval()
        losses = AverageMeter()
        ACC = AverageMeter()

        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for _, data in enumerate(tk0):
                images = data["images"]
                labels = data["labels"]

                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, labels)

                predictions = torch.softmax(output, dim=1)
                _, predictions = torch.max(predictions, dim=1)

                acc = accuracy(labels, predictions)

                losses.update(loss.item(), images.size(0))
                ACC.update(acc.item(), images.size(0))

                tk0.set_postfix(loss=losses.avg)
        print(f"Validation Loss = {losses.avg}")
        return ACC.avg
