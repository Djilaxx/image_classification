# **Image Classification**

This repository contains image classification projects i've worked on, either for fun/education or competition on Kaggle. \
Each project have it's own readme containing information about the specific problematics of each. 

I train the models locally on my pc using a Nvidia 1080 GPU. 

## **Task**
---
The task folder contains the specific code about each project, a config.py file containing  most of the hyperparameters of the model, and a training.py file that start the training cycle. \
To start training a model on any task use this command in terminal :
```
python -m task.aerial_cactus.training
```
You can replace the **aerial_cactus** with any folder in task.
Default parameters train for **5** fold using a **Resnet18** model and return validation **accuracy** after each epoch. 
You can change these parameters as such :
```
python -m task.aerial_cactus.training --folds=3 --model=RESNET34 --metric=F1_SCORE
```

The parameters can take different values :
* **folds** : this parameter determine the number of folds to create into the dataset. If you choose 5 for example, the dataset will be divided in 5, train a model on 4 folds and validate on the last (folds 0, 1, 2, and 3 for training and 4 for validation. Then, it'll train on folds 0, 1, 2, 4 and validate on 3 etc...).
* **model** : You can choose any model that is in the models/ folder, name must be typed in MAJ like in the example above.
* **metric** : The training procedure return loss throughout training and validation, and accuracy by default, but you can choose to return another validation metric if needed (the metrics you can choose from are in **utils/metrics.py**)

## **Image_dataset**
---
The image dataset folder contain a dataset class that loads images and corresponding labels and return them as tensors ready to be used by your model.

## **Models**
---
The models folder contain the different models that can be used on any task. \
The models i use follow a similar structure, they are pretrained models to which i add a final linear layer for classification.

The model output is directly coming from the final layer. If you want to obtain probabilities you should pass the output through a softmax/sigmoid function.
## **Trainer** 
---
The trainer is a class that is used to train the model for one epoch and validate afterward. It contains a training function and an evaluation function.

## **To do** 
---
* Add more models
* Add augmentation files so adding or removing image augmentation is done outside the run function. 
* Add TPU support for the dataset and training. 
* Add metrics
* Debug the AUROC metric (not usable atm)